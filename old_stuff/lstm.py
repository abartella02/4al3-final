import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Example: Dummy binary sequence data
X = [
    [1, 2, 3, 4],
    [1, 3, 4],
    [2, 3, 4, 5, 6],
    [1, 2],
    [3, 4, 5]
]
y = [0, 1, 1, 0, 1]  # Binary labels

# Pad sequences to ensure equal length
X = pad_sequences(X, maxlen=5)  # Pad to a max length of 5
y = np.array(y)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



model = Sequential()

# Embedding layer (optional, for sequences like text data)
model.add(Embedding(input_dim=10, output_dim=8, input_length=5))  # Adjust input_dim and output_dim as needed

# LSTM layer
model.add(LSTM(units=16, return_sequences=False))  # Set return_sequences=True if stacking more LSTMs

# Dense layer for binary classification
model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary output

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',  # Binary cross-entropy for binary classification
    metrics=['accuracy']
)

# Print model summary
model.summary()


history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")


predictions = model.predict(X_test)
binary_predictions = (predictions > 0.5).astype(int)
print(binary_predictions)
