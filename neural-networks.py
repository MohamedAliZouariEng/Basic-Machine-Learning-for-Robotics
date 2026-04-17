import tensorflow as tf
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt 
import numpy as np
import random 
import tkinter as tk 
from tkinter import *

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train, X_test, = X_train / 255.0, X_test / 255.0

model = Sequential([Input(shape=(28,28)),
                    Flatten(),
                    Dense(128, activation='relu'),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(64, activation='relu'),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(10, activation='softmax')])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
callbacks = [EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True),
             ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')]

model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2, callbacks=callbacks, verbose=2)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc:.2f}")

model.save('final_model.keras')

plot_model(model, to_file='model_architecture.png', show_shaps=True, show_layer_names=True)

img = plt.imread('model_architecture.png')
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.axis('off')
plt.show()


def visualize_prediction(index=None):
    if index is None:
        index = random.randint(0, X_test.shape[0] - 1)
    
    img = X_test[index]
    actual_label = y_test[index]

    prediction = model.predict(np.expand_dims(img, axis=0))
    predicted_label = np.argmax(prediction)

    plt.figure(figsize=(4,4))
    plt.imshow(img, cmap='gray')
    plt.title(f"Actual: {actual_label}, Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()

visualize_prediction()