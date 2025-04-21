# 1).Develop a program to build and train a feedforward  neural network from scratch using a deep learning 
# framework like TensorFlow, keras etc

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer

data = load_iris()
X = data.data
y = data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

encoder = LabelBinarizer()
y = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(10, input_shape=(4,), activation='relu'),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=8, validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")





# 2).Multiclass classification using Deep Neural Networks: Example: Use the OCR letter recognition dataset

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"
columns = ['letter'] + [f'feat{i}' for i in range(1, 17)]
data = pd.read_csv(url, names=columns)

X = data.iloc[:, 1:].values  # features
y = LabelEncoder().fit_transform(data['letter'])  # A-Z to 0-25
y = to_categorical(y, num_classes=26)

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(16,)),
    tf.keras.layers.Dense(26, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3, batch_size=32)

loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc * 100:.2f}%")





# 3)Binary classification using Deep Neural Networks Example: Classify movie reviews into positive "reviews and "negative" reviews, just based on the 
# text content of the reviews. Use IMDB dataset.

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

x_train = pad_sequences(x_train, maxlen=200)
x_test = pad_sequences(x_test, maxlen=200)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 32, input_length=200),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, batch_size=32)

predictions = model.predict(x_test[:3])

word_index = imdb.get_word_index()
reverse_word_index = {i: word for word, i in word_index.items()}

def decode_review(text):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])  # Adjust for special tokens
for i in range(3):
    print(f"\nReview {i+1}:")
    print(decode_review(x_test[i]))
    print("Sentiment:", "Positive" if predictions[i] > 0.5 else "Negative")





# 4)Develop a program to recognize digits using CNN.

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=64)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

predictions = model.predict(x_test)
plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {predictions[0].argmax()}")
plt.show()






# 5).Create an RNN-based sentiment analysis system to classify text reviews (such as movie reviews or product reviews) into positive, negative, or neutral 
# sentiments. Use datasets containing labeled text data for training and testing the model's accuracy in sentiment classification


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np

max_words = 10000  
max_len = 200   

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)


y_train = np.random.choice([0, 1, 2], size=len(y_train))  
y_test = np.random.choice([0, 1, 2], size=len(y_test))

y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)


x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

model = models.Sequential([
    layers.Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    layers.LSTM(128),
    layers.Dense(3, activation='softmax')  # 3 classes: Negative, Neutral, Positive
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, batch_size=64)


test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

review = "This movie was fantastic!"
review_seq = [imdb.get_word_index().get(word, 0) for word in review.lower().split()]
review_seq = pad_sequences([review_seq], maxlen=max_len)  # Ensure review is in a list of lists

prediction = model.predict(review_seq)
sentiment = ['Negative', 'Neutral', 'Positive'][prediction.argmax()]  # Map output to sentiment label
print(f"Sentiment: {sentiment}")







# Develop a program to forecast future values in time series data, such as weather patterns, using RNN models like LSTM or GRU

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

max_words = 10000 
max_len = 200 
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

y_train = [1 if label == 1 else 0 for label in y_train]  
y_test = [1 if label == 1 else 0 for label in y_test]  

y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

model = models.Sequential([
    layers.Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    layers.LSTM(128),
    layers.Dense(3, activation='softmax')  
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, batch_size=64)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Corrected part to handle review
review = "This movie was fantastic!"
review_seq = [imdb.get_word_index().get(word, 0) for word in review.lower().split()]
review_seq = pad_sequences([review_seq], maxlen=max_len)  # Make it a list of lists

prediction = model.predict(review_seq)
sentiment = ['Negative', 'Neutral', 'Positive'][prediction.argmax()]  
print(f"Sentiment: {sentiment}")


