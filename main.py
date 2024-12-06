import pandas as pd
import sklearn
from text_encoding import Word2Vec
from sampling_strategy import sample_and_clean
import tensorflow as tf
from tensorflow.keras.layers import Embedding

from typing import Tuple

import sklearn.metrics
import sklearn.model_selection


class RNNTextClassifier:

    def __init__(self, w2v_embedding_layer: Embedding = None):
        self.model = None
        self.embedding_layer = w2v_embedding_layer

    def split_data(self, data, split_size):

        # Separate features and labels
        features = data.iloc[:, :-1]  # All columns except the last one
        labels = data.iloc[:, -1]  # Last column

        # Split into training and testing sets
        train_features, test_features, train_labels, test_labels = sklearn.model_selection.train_test_split(
            features, labels, shuffle=True, test_size=split_size
        )
        return train_features, train_labels, test_features, test_labels

    def build(self, input_dimension):
        self.model = tf.keras.Sequential()
        if self.embedding_layer is None:
            self.model.add(tf.keras.layers.Embedding(input_dim=input_dimension, output_dim=64, mask_zero=True))
            self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
            self.model.add(tf.keras.layers.Dense(64), activation='relu')

        else:
            embed_layer = self.embedding_layer()
            print('output layer dim', embed_layer.output_dim)
            print('input layer dim', embed_layer.input_dim)
            print('passed input layer', input_dimension)
            self.model.add(embed_layer)
            self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embed_layer.output_dim)))
            self.model.add(tf.keras.layers.Dense(embed_layer.output_dim, activation='relu'))

        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, train_text, train_label, epoch, batch_size):
        self.model.fit(train_text, train_label, epochs=epoch, batch_size=batch_size, verbose=2)

    def predict(self, test_text):
        results = self.model.predict(test_text)
        predicted_labels = []
        for value in results:
            if value > 0.5:
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)
        return predicted_labels


def tpr_fpr(predictions, real) -> Tuple[int,int,int,int]:
    """Function to look at accuracy"""

    # create counters
    tp_count = 0  # true positives
    fp_count = 0  # false positives
    tn_count = 0  # true negatives
    fn_count = 0  # false negatives

    # tpr/fpr defaults to 0
    tpr = 0
    fpr = 0

    # count results
    for i in range(len(predictions)):

        # true positives
        if predictions[i] == 1 and real[i] == 1:
            tp_count += 1

        # false positives
        elif predictions[i] == 1 and real[i] == 0:
            fp_count += 1

        # true negatives
        elif predictions[i] == 0 and real[i] == 0:
            tn_count += 1

        # false negatives
        elif predictions[i] == 0 and real[i] == 1:
            fn_count += 1

    # calculate tpr
    try:
        tpr = tp_count / (tp_count + fn_count)

    except ZeroDivisionError:
        print("\nError!!! Cant compute following TPR due to insufficient data")

    # calculate fpr
    try:
        fpr = fp_count / (fp_count + tn_count)

    except ZeroDivisionError:
        print("\nError!!! Cant compute following FPR due to insufficient data")

    return tp_count, fp_count, tn_count, fn_count


def preprocess(dataset, samples_per_class):
    class Dataset(pd.DataFrame):
        _metadata = ["w2v"]  # Ensures `w2v` persists through Pandas operations

        def __init__(self, data, w2v=None):
            super().__init__(data)
            self.w2v = w2v

        @property
        def _constructor(self):
            return Dataset

    dataset = sample_and_clean(dataset, samples_per_class=samples_per_class)

    # convert to numerical representation
    w2v = Word2Vec(dataset=dataset)
    features, labels = w2v.encode_text_dataset()

    # in case any words are not in our downloaded words
    features = features.clip(0, len(w2v.word_indices) - 1)

    # Combine train_text and train_label for filtering
    data = pd.concat((features, labels), axis=1)

    # Drop rows where train_label is NaN
    data = data.dropna(subset=[labels.name])

    # Separate train_text and train_label
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]

    return Dataset(features, w2v), labels


if __name__ == '__main__':
    train = pd.read_csv('data/final_train.csv')
    test = pd.read_csv('data/final_test.csv')

    train_features, train_labels = preprocess(train, samples_per_class=500)
    test_features, test_labels = preprocess(test, samples_per_class=200)

    rnn = RNNTextClassifier(w2v_embedding_layer=train_features.w2v.embed_layer)

    print("total number of training points")
    print(len(train_labels))
    print("number of positive training points")
    print(sum(train_labels))

    # input dimension is number of words we have downloaded
    # input_dim = len(w2v.word_indices)
    input_dim = len(train_features.w2v.word_indices)
    print(f"Setting input_dim to {input_dim}")
    rnn.build(input_dimension=input_dim)

    # train the model
    rnn.train(train_features, train_labels, epoch=5, batch_size=10)

    # predict
    predicted_labels = rnn.predict(test_features)
    actual_labels = test_labels.values.tolist()

    tp, fp, tn, fn = tpr_fpr(predicted_labels, actual_labels)

    print("tp:", tp)
    print("fp:", fp)
    print("tp:", tn)
    print("tp:", fn)
    print("accuracy:", (tp+tn)/(tn+fn+fp+tp))
