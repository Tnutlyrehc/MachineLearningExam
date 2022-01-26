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
path = 'data'

from main import X_train, CC_train_labels, X_val, CC_val_labels, X_test, CC_test_labels, D_train_labels, D_test_labels, Y_val_labels, Y_train_labels, Y_test_labels, D_val_labels



callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
inputs = Input(shape = (84,150, 3))
y = Conv2D(6, 5, activation='relu')(inputs)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)
y = Conv2D(16, 5, activation='relu')(y)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)

x = tf.keras.layers.Flatten()(y)
x = Dense(128, activation= 'relu', kernel_regularizer= regularizers.l2(0.01))(x)
outputs = Dense(11, activation='softmax', kernel_regularizer= regularizers.l2(0.01))(x)

ConvMod_Y = Model(inputs, outputs)
ConvMod_Y.summary()
ConvMod_Y.compile( optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

Y_fit = ConvMod_Y.fit(X_train, Y_train_labels, epochs=25, batch_size=32, validation_data= (X_val, Y_val_labels), callbacks = [callback])
results_Y = ConvMod_Y.evaluate(X_test, Y_test_labels,  batch_size=16)
ConvMod_Y.save('Y.h5')