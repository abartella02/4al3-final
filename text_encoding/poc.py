import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
import gensim.downloader as api
from gensim.models import KeyedVectors
import re
import pandas as pd
import time

start_time = time.time()
keyed_vectors = api.load('word2vec-google-news-300')
# keyed_vectors = api.load('word2vec-ruscorpora-300')
# keyed_vectors = KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)

weights = keyed_vectors.vectors

embed_layer = Embedding(
    input_dim=weights.shape[0],
    output_dim=weights.shape[1],
    weights=[weights],
    trainable=False
)  # maps the words to weights


word_indices = {word.lower(): idx for idx, word in enumerate(keyed_vectors.index_to_key)}  # dict with word: index
def words_to_indices(sentence: str):
    """Convert a sentence to a list of indices, readable by a NN's word2vec layer"""
    words = re.findall(r'\w+', sentence)  # split the sentence into a list of words
    return [word_indices.get(word.lower(), 0) for word in words]  # convert to list of indices

df = pd.read_csv('LLM.csv')
encoded_sentences = list(df['Text'].apply(words_to_indices))
max_len_sentence = max(len(sentence) for sentence in encoded_sentences)
encoded_sentences = pad_sequences(encoded_sentences, maxlen=max_len_sentence, padding='post')

features = pd.DataFrame(encoded_sentences)
labels = df['Label'].map({'student':0, 'ai':1})

print(f"Total execution time: {time.time()-start_time}")

a = 1