import tensorflow as tf
import gensim.downloader as api
from gensim.models import KeyedVectors
from tensorflow.keras.layers import Embedding




keyed_vectors = api.load('word2vec-google-news-300')
weights = keyed_vectors.vectors
index_to_key = keyed_vectors.index_to_key

embed_layer = Embedding(
    input_dim=weights.shape[0],
    output_dim=weights.shape[1],
    weights=[weights],
    trainable=False
)


