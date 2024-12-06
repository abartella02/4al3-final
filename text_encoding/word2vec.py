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
            self.weights = keyed_vectors.vectors
            np.save(prev_weights, self.weights)

            self.word_indices = {
                word.lower(): idx for idx, word in enumerate(keyed_vectors.index_to_key)
            }
            with open(prev_index_to_key, "w") as f:
                json.dump(self.word_indices, f)

    def embed_layer(self, output_dim=None) -> Embedding:
        """Convert word2vec embeddings to tensorflow embedding layer"""
        return Embedding(
            input_dim=self.weights.shape[0],
            output_dim=self.weights.shape[1] if output_dim is None else output_dim,
            weights=[self.weights],
            trainable=False,
        )

    def __words_to_indices(self, sentence: str) -> list[int]:
        """Convert a sentence to a list of indices, readable by a NN's word2vec layer"""
        words = re.findall(r"\w+", sentence)  # split the sentence into a list of words
        return [
            self.word_indices.get(word.lower(), 0) for word in words
        ]  # convert to list of indices

    def encode_text_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        labels = self.dataset["Label"]#.map({"student": 0, "ai": 1})  # create label matrix

        return features, labels
