'''
NOTES:

- References
    - Used RNN-demo.py from Class examples (Week 12) as a reference
    - Geeks for Geeks: RNN for Text Classifications in NLP
      https://www.geeksforgeeks.org/rnn-for-text-classifications-in-nlp/
    - Tensorflow: Text Classification with an RNN
      https://www.tensorflow.org/text/tutorials/text_classification_rnn

- This model is heavily based on the references above
- Classifies the text as student or AI (AI set as label 1, student set as label 0)
- The dataset is LLM.csv and comes from Kaggle
    - This is a really small dataset though (1 sentence each, little over 1000 data points)
    - Also it is very obvious which sentences are student and which are AI (the words used in the "student" samples are almost the same every time)

- Potentially could be used as a starting point
'''



import pandas as pd
import numpy as np
import sklearn
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing

import tensorflow as tf




class rnn_textclassifier:

    def __init__(self):
        self.model = None

    def split_data(self, data, split_size):
        train_set, test_set = sklearn.model_selection.train_test_split(data, shuffle=True, test_size=split_size)
        train_text = train_set["Text"]
        train_label = train_set["Label"]
        test_text = test_set["Text"]
        test_label = test_set["Label"]

        return train_text, train_label, test_text, test_label
    

    def make_tensor(self, train_text, train_label, test_text, test_label):
        train_text_tensor = tf.convert_to_tensor(train_text)
        train_label_tensor = tf.convert_to_tensor(train_label)
        test_text_tensor = tf.convert_to_tensor(test_text)
        test_label_tensor = tf.convert_to_tensor(test_label)
        return train_text_tensor, train_label_tensor, test_text_tensor, test_label_tensor


    def encode(self, train_text_tensor, vocab_size):
        encoder = tf.keras.layers.TextVectorization(max_tokens=vocab_size)
        encoder.adapt(train_text_tensor)
        return encoder
    

    def build(self, encoder):
        self.model = tf.keras.Sequential()
        self.model.add(encoder)
        self.model.add(tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=64,mask_zero=True))
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
        self.model.add(tf.keras.layers.Dense(64, activation='relu'))
        self.model.add(tf.keras.layers.Dense(1))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    def train(self, train_text_tensor, train_label_tensor, epoch, batch_size):
        self.model.fit(train_text_tensor, train_label_tensor, epochs=epoch, batch_size=batch_size, verbose=2)


    def predict(self, test_text_tensor):
        results = self.model.predict(test_text_tensor)
        predicted_labels = []
        for value in results:
            if value > 0:
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)
        return predicted_labels






# load the csv
dataset = pd.read_csv('LLM.csv')

# change the labels from "ai" and "student" to 1 and 0
# 1 for "ai" and 0 or "student"
dataset["Label"] = np.where(dataset["Label"]=="ai", 1, 0)

# intialize RNN
rnn = rnn_textclassifier()

# split data in to train and test and their text and label columns
train_text, train_label, test_text, test_label = rnn.split_data(dataset, 0.2)

# convert the data to tensors
train_text_tensor, train_label_tensor, test_text_tensor, test_label_tensor = rnn.make_tensor(train_text, train_label, test_text, test_label)

# set the vocabulary size and create the encoder
vocabulary_size = 400
encoder = rnn.encode(train_text_tensor, vocabulary_size)

# build the encoder
rnn.build(encoder)

# train the model
rnn.train(train_text_tensor, train_label_tensor, epoch=10, batch_size=10)

# predict
predicted_labels = rnn.predict(test_text_tensor)

correct_predictions = 0
total_predictions = 0

# count how many correct predictions there are
for prediction, actual, text in zip(predicted_labels, test_label, test_text):
    if prediction != actual:
        print("text: ", text, "predicted: ", prediction, "actual: ", actual)
    if prediction == actual:
        correct_predictions += 1
    total_predictions += 1

print("correct predictions: ", correct_predictions)
print("total_predictions: ", total_predictions)












