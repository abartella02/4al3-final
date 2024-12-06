import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
import gensim.downloader as api
from gensim.models import KeyedVectors
import re
import pandas as pd
from typing import Tuple
from pathlib import Path
import json
import sklearn
import sklearn.metrics
import sklearn.model_selection
import tensorflow as tf


class Word2Vec:
    """
    Encode a dataframe where the samples are selections of text (essays, sentences, etc)
    using pre-trained word2vec model. Default model is "google-news-300".
    """
    def __init__(
        self, dataset: pd.DataFrame, model: str = "word2vec-google-news-300"
    ) -> None:
        self.dataset = dataset
        __dataset_same = False  # has a new dataset been entered?

        # paths to stored csv, weights, and index_to_key dict
        prev_data = Path(__file__).parent / ".word2vec"
        prev_csv = prev_data / "prev.csv"
        prev_weights = prev_data / "prev_weights.npy"
        prev_index_to_key = prev_data / "index_to_key.json"

        # check if the previous dataframe exists
        # to avoid loading the (very large) word2vec model if possible
        if not prev_csv.exists():
            prev_data.mkdir(exist_ok=True)  # make containing folder
            self.dataset.to_csv(prev_csv)  # write current dataframe to folder
        else:
            prev_df = pd.read_csv(prev_csv)  # load previous df
            prev_df = prev_df.drop(
                columns=prev_df.columns[0]
            )  # drop first column (row numbers)
            if prev_df.equals(dataset):  # check if current and previous df are the same
                print("Same dataset entered.")
                __dataset_same = True  # set flag
            else:
                print("New dataset entered.")
                self.dataset.to_csv(prev_csv)  # write current df to folder

        # if current dataframe == previous dataframe and all important data exists
        if __dataset_same and prev_weights.exists() and prev_index_to_key.exists():
            print("Loading saved weights and indices...")
            # load weights and word indices from folder instead of recalculating
            self.weights = np.load(prev_weights)
            with open(prev_index_to_key, "r") as f:
                self.word_indices = json.load(f)

        else:
            # load word2vec model and calculate weights and word indices as normal
            print("Loading model...")
            keyed_vectors = api.load(model)
            keyed_vectors.index_to_key = keyed_vectors.index_to_key[:50000]  # Slice the vocabulary
            keyed_vectors.vectors = keyed_vectors.vectors[:50000]  # Slice the embedding vectors
            self.weights = keyed_vectors.vectors
            np.save(prev_weights, self.weights)

            self.word_indices = {
                word.lower(): idx for idx, word in enumerate(keyed_vectors.index_to_key)
            }
            with open(prev_index_to_key, "w") as f:
                json.dump(self.word_indices, f)

    @property
    def embed_layer(self) -> Embedding:
        """Convert word2vec embeddings to tensorflow embedding layer"""
        return Embedding(
            input_dim=self.weights.shape[0],
            output_dim=self.weights.shape[1],
            weights=[self.weights],
            trainable=False,
        )


    def __words_to_indices(self, sentence: str) -> list[int]:
        """Convert a sentence to a list of indices, readable by a NN's word2vec layer"""
        words = re.findall(r"\w+", sentence)  # split the sentence into a list of words
        return [
            self.word_indices.get(word.lower(), 0) for word in words
        ]  # convert to list of indices


    def encode_text_dataset(self, label_type = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Encode words in dataset as integers corresponding to weights in the embedding layer"""
        encoded_sentences = list(
            self.dataset["Text"].apply(self.__words_to_indices)
        )  # convert to indices
        max_len_sentence = max(
            len(sentence) for sentence in encoded_sentences
        )  # find sample with most words
        encoded_sentences = pad_sequences(
            encoded_sentences, maxlen=max_len_sentence, padding="post"
        )  # pad all other samples

        features = pd.DataFrame(encoded_sentences)  # create feature matrix

        # if label in the csv is 0/1
        if label_type == 0:
            labels = self.dataset["Label"]

        # if lavel in the csv is "student"/"ai"
        elif label_type == 1:
            labels = self.dataset["Label"].map(
                {"student": 0, "ai": 1}
            )  # create label matrix

        return features, labels




class rnn_textclassifier:

    def __init__(self):
        self.model = None

    def split_data(self, data, split_size):

        # Separate features and labels
        features = data.iloc[:, :-1]  # All columns except the last one
        labels = data.iloc[:, -1]    # Last column

        # Split into training and testing sets
        train_features, test_features, train_labels, test_labels = sklearn.model_selection.train_test_split(
            features, labels, shuffle=True, test_size=split_size
        )
        return train_features, train_labels, test_features, test_labels
    

    def build(self, input_dimension):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Embedding(input_dim=input_dimension, output_dim=64,mask_zero=True))
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
        self.model.add(tf.keras.layers.Dense(64, activation='relu'))
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


# small function to look at accuract
def tpr_fpr(predictions, real):
    # create counters
    tp_count = 0 # true positives
    fp_count = 0 # false positives
    tn_count = 0 # true negatives
    fn_count = 0 # false negatives

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
    if tp_count + fn_count > 0:
        tpr = tp_count/(tp_count + fn_count)
    
    # in case divisor is 0
    else:
        print("\nError!!! Cant compute following TPR due to insufficent data")

    # calculate fpr
    if fp_count + tn_count > 0:
        fpr = fp_count/(fp_count + tn_count)

    # in case divisor is 0
    else:
        print("\nError!!! Cant compute following FPR due to insufficent data")

    return tp_count, fp_count, tn_count, fn_count






def main():

    '''
    ADD CODE HERE TO CLEAN INITIAL CSV
    
    '''



    # load in data
    df = pd.read_csv('edited_train.csv')

    # convert to numerical representation
    w2v = Word2Vec(dataset=df)
    features, labels = w2v.encode_text_dataset()

    # in case any words are not in our downloaded words
    features = features.clip(0, len(w2v.word_indices) - 1)

    # concatonate before passing into lstm
    processed_data = pd.concat([features, labels], axis=1)


    # intialize LSTM
    rnn = rnn_textclassifier()

    # split data in to train and test and their text and label columns
    train_text, train_label, test_text, test_label = rnn.split_data(processed_data, 0.2)


    # Combine train_text and train_label for filtering
    data = pd.concat([train_text, train_label], axis=1)
    
    # Drop rows where train_label is NaN
    data = data.dropna(subset=[train_label.name])
    
    # Separate train_text and train_label
    train_text = data.iloc[:, :-1]
    train_label = data.iloc[:, -1]
    

    print("total number of training points")
    print(len(train_label))
    print("number of positive training points")
    print(sum(train_label))

    '''   
    # code to check for illegal words
    max_index = train_text.max().max()  # Max index in training data
    vocab_size = len(w2v.word_indices)  # Size of vocabulary
    if max_index >= vocab_size:
        raise ValueError(f"Train data contains out-of-vocabulary indices: {max_index} >= {vocab_size}")

    print(f"Train data max index: {max_index}, Vocabulary size: {vocab_size}")'''

    # input dimension is number of words we have downloaded
    input_dim = len(w2v.word_indices)
    print(f"Setting input_dim to {input_dim}")
    rnn.build(input_dimension=input_dim)

    # train the model
    rnn.train(train_text, train_label, epoch=5, batch_size=10)

    # predict
    predicted_labels = rnn.predict(test_text)
    actual_labels = test_label.values.tolist()

    true_positives, false_positives, true_negatives, false_negatives =tpr_fpr(predicted_labels, actual_labels)

    print(true_positives)
    print(true_negatives)
    print(false_positives)
    print(false_negatives)

main()
