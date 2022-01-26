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

# Defining the model for CC
inputs = Input(shape = (84,150, 3))
y = Conv2D(6, 5, activation='relu')(inputs)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)
y = Conv2D(16, 5, activation='relu')(y)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)

x = keras.layers.Flatten()(y)
x = Dense(128, activation= 'relu', kernel_regularizer= regularizers.l2(0.01))(x)
outputs = Dense(1, activation='sigmoid', kernel_regularizer= regularizers.l2(0.01))(x)

ConvMod_CC = Model(inputs, outputs)
ConvMod_CC.summary()
ConvMod_CC.compile( optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

CC_fit = ConvMod_CC.fit(X_train, CC_train_labels, epochs=25, batch_size=32, validation_data= (X_val, CC_val_labels), callbacks = [callback])
results = ConvMod_CC.evaluate(X_test, CC_test_labels,  batch_size=16)
ConvMod_CC.save('CC.h5')


# Defining the model for D
inputs_D = Input(shape = (84,150, 3))
y = Conv2D(6, 5, activation='relu')(inputs_D)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)
y = Conv2D(16, 5, activation='relu')(y)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)

x = keras.layers.Flatten()(y)
x = Dense(128, activation= 'relu', kernel_regularizer= regularizers.l2(0.01))(x)
outputs_D = Dense(11, activation='softmax', kernel_regularizer= regularizers.l2(0.01))(x)

ConvMod_D = Model(inputs_D, outputs_D)
ConvMod_D.summary()
ConvMod_D.compile( optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

D_fit = ConvMod_D.fit(X_train, D_train_labels, epochs=25, batch_size=32, validation_data= (X_val, D_val_labels), callbacks = [callback])
results = ConvMod_D.evaluate(X_test, D_test_labels,  batch_size=16)
ConvMod_D.save('D.h5')

# Defining the model for Y
inputs_Y = Input(shape = (84,150, 3))
y = Conv2D(6, 5, activation='relu')(inputs_Y)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)
y = Conv2D(16, 5, activation='relu')(y)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)

x = keras.layers.Flatten()(y)
x = Dense(128, activation= 'relu', kernel_regularizer= regularizers.l2(0.01))(x)
outputs_Y = Dense(11, activation='softmax', kernel_regularizer= regularizers.l2(0.01))(x)

ConvMod_Y = Model(inputs_Y, outputs_Y)
ConvMod_Y.summary()
ConvMod_Y.compile( optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

Y_fit = ConvMod_Y.fit(X_train, Y_train_labels, epochs=25, batch_size=32, validation_data= (X_val, Y_val_labels), callbacks = [callback])
results = ConvMod_Y.evaluate(X_test, Y_test_labels,  batch_size=16)
ConvMod_Y.save('Y.h5')

len(CC_fit.history['loss'])
len(D_fit.history['loss'])
len(Y_fit.history['loss'])


print("test loss, test acc:", results)
def plot_loss_acc(model_fit):
    history_dict = model_fit.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    train_acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    epochs = range(1, len(loss_values) + 1)

    fig, axs = plt.subplots(2)
    axs[0].plot(epochs, loss_values, 'bo', label='Training loss')
    axs[0].plot(epochs, val_loss_values, 'b', label='Validation loss')
    axs[0].set_title('Training and validation loss')
    axs[1].plot(epochs, train_acc, 'ro', label='Training accuracy')
    axs[1].plot(epochs, val_acc, 'r', label='Validation accuracy')
    axs[1].set_title('Training and validation accuracy')
    plt.show()
plot_loss_acc(CC_fit)
plot_loss_acc(D_fit)
plot_loss_acc(Y_fit)


