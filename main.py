import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from sklearn.metrics import confusion_matrix

from typing import Tuple, Optional

from text_encoding import Word2Vec
from sampling_strategy import sample_and_clean


class RNNTextClassifier:
    """
    Basic RNN text classifier, as outlined in Tensorflow's
    "Text classification with an RNN" and "RNN Demo" from AvenueToLearn.
    """

    def __init__(self, w2v_embedding_layer: Optional[Embedding] = None) -> None:
        self.model = None
        self.embedding_layer = w2v_embedding_layer

    def build(self, input_dimension: int) -> None:
        """
        Build the model.
        Layers:
            * Embedding layer (for word2vec)
            * Bidirectional layer
            * Dense layer (fully connected layer with relu activation)

        Use embedded layer from word2vec object if available.
        """
        self.model = tf.keras.Sequential()
        if self.embedding_layer is None:
            self.model.add(
                tf.keras.layers.Embedding(
                    input_dim=input_dimension, output_dim=64, mask_zero=True
                )
            )
            self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
            self.model.add(tf.keras.layers.Dense(64), activation="relu")

        else:
            embed_layer = self.embedding_layer()
            self.model.add(embed_layer)
            self.model.add(
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(embed_layer.output_dim)
                )
            )
            self.model.add(
                tf.keras.layers.Dense(embed_layer.output_dim, activation="relu")
            )

        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        self.model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

    def train(self, train_text: pd.DataFrame, train_label: pd.DataFrame, epoch: int, batch_size: int) -> None:
        self.model.fit(
            train_text, train_label, epochs=epoch, batch_size=batch_size, verbose=2
        )

    def predict(self, test_text: pd.DataFrame) -> list[int]:
        results = self.model.predict(test_text)
        predicted_labels = []
        for value in results:
            if value > 0.5:
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)
        return predicted_labels

    def prediction_metrics(self, y_predicted: list[int], y_actual: pd.DataFrame) -> Tuple[int, int, int, int]:
        """Return confusion matrix metrics"""
        tn, fp, fn, tp = confusion_matrix(y_actual, y_predicted).ravel()

        return tn, fp, fn, tp


def preprocess(dataset: pd.DataFrame, samples_per_class: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    class Dataset(pd.DataFrame):
        """Subclass of DataFrame to carry on the Word2Vec object as an attribute, for use later"""
        _metadata = ["w2v"]  # Ensures `w2v` persists through Pandas operations

        def __init__(self, data: pd.DataFrame, w2v: Optional[Word2Vec] = None) -> None:
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


if __name__ == "__main__":
    train = pd.read_csv("data/final_train.csv")
    test = pd.read_csv("data/final_test.csv")

    train_features, train_labels = preprocess(train, samples_per_class=500)
    test_features, test_labels = preprocess(test, samples_per_class=200)

    rnn = RNNTextClassifier(w2v_embedding_layer=train_features.w2v.embed_layer)

    print("total number of training points")
    print(len(train_labels))
    print("number of positive training points")
    print(sum(train_labels))

    # input dimension is number of words we have downloaded
    input_dim = len(train_features.w2v.word_indices)
    print(f"Setting input_dim to {input_dim}")
    rnn.build(input_dimension=input_dim)

    # train the model
    rnn.train(train_features, train_labels, epoch=5, batch_size=10)

    # predict
    predicted_labels = rnn.predict(test_features)
    actual_labels = test_labels.values.tolist()

    tn, fp, fn, tp = rnn.prediction_metrics(
        y_predicted=predicted_labels, y_actual=test_labels
    )

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print("test accuracy: ", accuracy)
    print("test sensitivity: ", sensitivity)
    print("test specificity: ", specificity)
