import numpy as np
import pandas as pd
import os
import re
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from keras import callbacks
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, MaxPool2D, GlobalAveragePooling2D, Concatenate
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard

from load_data import y_train, y_test, y_val, X_test, X_val, X_train


path = 'data'

tensorboard = TensorBoard(
  log_dir='.\logs',
  histogram_freq=1,
  write_images=True
)
keras_callbacks = [
  tensorboard
]



callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
inputs = Input(shape = (84,150, 3))
y = Conv2D(6, 5, activation='relu')(inputs)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)
y = Conv2D(16, 5, activation='relu')(y)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)

x = tf.keras.layers.Flatten()(y)
x = Dense(128, activation= 'relu', kernel_regularizer= regularizers.l2(0.01))(x)
outputs = Dense(11, activation='softmax', kernel_regularizer= regularizers.l2(0.01))(x)

ConvMod_D = Model(inputs, outputs)
ConvMod_D.summary()
ConvMod_D.compile( optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

D_fit = ConvMod_D.fit(X_train, D_train_labels, epochs=25, batch_size=32, validation_data= (X_val, D_val_labels), callbacks = [callback])
results_D = ConvMod_D.evaluate(X_test, D_test_labels,  batch_size=16)
ConvMod_D.save('D.h5')

plt.plot(D_fit.history['accuracy'], label='Training accuracy')
plt.plot(D_fit.history['val_accuracy'], label='Validation accuracy')
plt.plot(D_fit.history['accuracy'], label='Training accuracy (regularized)')
plt.plot(D_fit.history['val_accuracy'], label='Validation accuracy (regularized)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

