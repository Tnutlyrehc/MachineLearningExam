import tensorflow as tf
from keras import Input, regularizers, Model
from keras.layers import Concatenate
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from keras.preprocessing import image
# Set path to this file location
from main import path
from os import listdir
from os.path import isfile, join
import glob



filenames = []
filenames = glob.glob(path +"train/*.jpg")

train_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=30,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    brightness_range=[0.5, 1.3])

batch_size=32
train_datagenerator = image.ImageDataGenerator(
                             rescale=1./255,
                             rotation_range=20,
                             zoom_range=0.2,
                             horizontal_flip=True)

train_generator_CC= train_datagenerator.flow_from_dataframe(
                            path + '/train',
                            labels = filenames,
                            batch_size=batch_size,
                            class_mode="binary",
                            target_size=(150,84),
                            color_mode='rgb')


# setting global variabels
val_size = 0.15
test_size = 0.1
train_size = 1 - val_size - test_size

inputs = Input(shape = (150,84,3))
x1 = Conv2D(8, (3,3), activation='relu', padding='same')(inputs)
x2 = Conv2D(8, (5,5), activation='relu', padding='same')(inputs)
x3 = Conv2D(16, (2,2), activation='relu', padding='same')(inputs)
y = Concatenate()([x1, x2, x3])
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)
print(y.shape)
x = keras.layers.Flatten()(y)
print(x.shape)
x = Dropout(0.5)(x)
x = Dense(32, activation= 'relu', kernel_regularizer=regularizers.l2(0.001))(x)
x = Dense(64, activation= 'relu', kernel_regularizer=regularizers.l2(0.001))(x)
outputs = Dense(1, activation='sigmoid')(x)

Incep_simple_model = Model(inputs, outputs)
Incep_simple_model.compile( optimizer='adam',
			   loss='binary_crossentropy',
			   metrics=['accuracy'])

Incep_simple_model.summary()

Incep_simple_model.fit(
    train_generator_CC,
    steps_per_epoch= 7680 * train_size // batch_size,
    epochs=1)


