import pandas as pd
import numpy as np
import kagglehub
import gensim.downloader as api
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
from tensorflow.keras import Model
from sklearn.metrics import confusion_matrix

import os
import shutil
from typing import Tuple, Optional
from pathlib import Path
import json
import re


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
        labels = self.dataset["Label"]

        return features, labels


class SamplingStrategy:
    def __init__(self):
        pass

    def data_cleanup(self, dataset: pd.DataFrame, amount_per_class: int) -> pd.DataFrame:
        # Drop rows with NaN in the 'text' column
        dataset = dataset.dropna(subset=["text"])

        # Drop any rows from the train data that has emojis
        emoji_rows = dataset[
            dataset["text"].str.contains(r"[\u263a-\U0001f645]", na=False)
        ]
        dataset = dataset.drop(emoji_rows.index, axis=0)

        # Create dataframe for new dataset
        shortened_data = pd.DataFrame()

        # Get an equal amount of random samples from each class and add it to the new dataset df
        np.random.seed(123)
        label_1 = dataset[dataset["label"] == 1].sample(
            amount_per_class, random_state=42
        )
        shortened_data = shortened_data._append(label_1)
        label_0 = dataset[dataset["label"] == 0].sample(
            amount_per_class, random_state=42
        )
        shortened_data = shortened_data._append(label_0)

        # Shuffle and reset the index for the new df
        shortened_data = shortened_data.sample(frac=1)
        shortened_data = shortened_data.reset_index(drop=True)

        return shortened_data

    def condense_text(self, data: pd.DataFrame, text_column_name: str) -> pd.DataFrame:
        """Function to convert each text sample into single paragraphs with no spacing, new lines, etc"""

        # Replace multiple line breaks with a space for each item in the column
        data[text_column_name] = data[text_column_name].str.replace(
            r"\s*\n\s*", " ", regex=True
        )

        # Normalize spaces (remove multiple spaces)
        data[text_column_name] = (
            data[text_column_name].str.replace(r"\s+", " ", regex=True).str.strip()
        )

        return data

    def sample_and_clean(
        self,
        data: pd.DataFrame,
        samples_per_class: int,
        to_csv: bool = False,
        output_csv_path: str = "data/preprocessed.csv",
    ) -> pd.DataFrame:
        """Preprocess using data cleanup and condense text"""

        # shorten the size of the dataset
        train_shortened_data = self.data_cleanup(data, samples_per_class)

        # get rid of whitespace and other stuff from the texts
        train_no_space_data = self.condense_text(train_shortened_data, "text")

        # rename columns
        train_no_space_data.columns = ["Text", "Label"]

        # put this new train df into a csv
        if to_csv:
            train_no_space_data.to_csv(output_csv_path, index=False)
        return train_no_space_data


class RNNTextClassifier:
    """
    Basic RNN text classifier, as outlined in Tensorflow's
    "Text classification with an RNN" and "RNN Demo" from AvenueToLearn.
    """

    def __init__(
        self,
        model: Optional[Model] = None,
        w2v_embedding_layer: Optional[Embedding] = None,
    ) -> None:
        self.model = model
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
        if isinstance(self.model, Model):
            print("Warning: overwriting pretrained model")

        self.model = tf.keras.Sequential()
        if self.embedding_layer is None:
            self.model.add(
                tf.keras.layers.Embedding(
                    input_dim=input_dimension, output_dim=64, mask_zero=True
                )
            )
            self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
            self.model.add(tf.keras.layers.Dense(64, activation="relu"))

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

    def train(
        self,
        train_text: pd.DataFrame,
        train_label: pd.DataFrame,
        epoch: int,
        batch_size: int,
    ) -> None:
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

    def prediction_metrics(
        self, y_predicted: list[int], y_actual: pd.DataFrame
    ) -> Tuple[int, int, int, int]:
        """Return confusion matrix metrics"""
        tn, fp, fn, tp = confusion_matrix(y_actual, y_predicted).ravel()

        return tn, fp, fn, tp


def preprocess(
    dataset: pd.DataFrame, samples_per_class: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    class Dataset(pd.DataFrame):
        """Subclass of DataFrame to carry on the Word2Vec object as an attribute, for use later"""

        _metadata = ["w2v"]  # Ensures `w2v` persists through Pandas operations

        def __init__(self, data: pd.DataFrame, w2v: Optional[Word2Vec] = None) -> None:
            super().__init__(data)
            self.w2v = w2v

        @property
        def _constructor(self):
            return Dataset

    ss = SamplingStrategy()
    dataset = ss.sample_and_clean(dataset, samples_per_class=samples_per_class)

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


def download_dataset() -> Path:
    """Download the dataset to ./data/ if the dataset has not already been downloaded"""
    # path to dataset in kagglehub
    dataset = "jdragonxherrera/augmented-data-for-llm-detect-ai-generated-text"

    # path to local folder
    dest_path = Path(__file__).parent / "data"

    # if the dataset files we need do not exist locally
    if not (
        (dest_path / "final_test.csv").exists()
        and (dest_path / "final_train.csv").exists()
    ):
        # delete /data/ if it exists but the csv's we need are not inside
        if dest_path.exists():
            shutil.rmtree(dest_path)
        # remake /data/ folder
        dest_path.mkdir(parents=True, exist_ok=True)

        # Download latest version
        # if the kaggle folder is present but no files are inside, the download will not initiate
        path = kagglehub.dataset_download(dataset)
        if len(os.listdir(path)) == 0:
            # if the folder is empty from a previous move operation, delete the folder
            shutil.rmtree(Path(path).parent)
            # attempt to download the dataset again
            path = kagglehub.dataset_download(dataset)

        for item in os.listdir(path):
            # move dataset csv's to repo folder
            shutil.move(os.path.join(path, item), dest_path)

    return dest_path


if __name__ == "__main__":
    download_dataset()

    train = pd.read_csv("data/final_train.csv")

    train_features, train_labels = preprocess(train, samples_per_class=25000)

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
    rnn.train(train_features, train_labels, epoch=15, batch_size=10)

    rnn.model.save("saved_model.h5")