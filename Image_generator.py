import numpy as np
import pandas as pd
import os
import re
from PIL import Image
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import keras
import sklearn
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, MaxPool2D, GlobalAveragePooling2D, Concatenate
from tensorflow.keras import regularizers
from load_data import labels, X_train, X_test, X_val, y_train,y_test, y_val

'''
# SMOTE technique for oversampling, does not work well for image data, because it needs one big feature array
smote = imblearn.over_sampling.SMOTE('minority')
print(X_train.shape, y_train.shape)

X_sm_train, y_sm_train = smote.fit_resample(X_train, labels.Y[y_train])

print(X_sm_train.shape, y_sm_train.shape)
print(X_sm_train.shape, y_sm_train.shape)
X_train = X_sm_train.reshape(-1,84,150,3)

print(X_train.shape)
print(X_train[0].shape)
img = Image.fromarray(X_train[4], 'RGB')
img.show()

unique, counts = np.unique(y_sm_train, return_counts=True)
print(dict(zip(unique, counts)))
'''
path = 'data'

batch_size=16
train_datagenerator = image.ImageDataGenerator(
                             rescale=1./255,
                             rotation_range=20,
                             zoom_range=0.2,
                            shear_range= 0.2)

train_generator_CC= train_datagenerator.flow_from_dataframe(
                            dataframe= labels.loc[y_train, :],
                            directory= path + '/train',
                            x_col= 'filenames',
                            y_col='CC_string',
                            batch_size=batch_size,
                            class_mode='binary',
                            target_size=(150,84),
                            color_mode='rgb')

train_generator_D= train_datagenerator.flow_from_dataframe(
                            dataframe= labels.loc[y_train, :],
                            directory= path + '/train',
                            x_col= 'filenames',
                            y_col='D_string',
                            batch_size=batch_size,
                            class_mode='categorical',
                            target_size=(150,84),
                            color_mode='rgb')

train_generator_Y= train_datagenerator.flow_from_dataframe(
                            dataframe= labels.loc[y_train, :],
                            directory= path + '/train',
                            x_col= 'filenames',
                            y_col='Y_string',
                            batch_size=batch_size,
                            class_mode='categorical',
                            target_size=(150,84),
                            color_mode='rgb')
# Same for the validation data
val_datagenerator = image.ImageDataGenerator(rescale=1/255)

val_generator_CC = val_datagenerator.flow_from_dataframe(
                            dataframe= labels.loc[y_val, :],
                            directory= path + '/validation',
                            x_col= 'filenames',
                            y_col='CC_string',
                            batch_size=batch_size,
                            class_mode='binary',
                            target_size=(150,84),
                            color_mode='rgb')

val_generator_D = val_datagenerator.flow_from_dataframe(
                            dataframe= labels.loc[y_val, :],
                            directory= path + '/validation',
                            x_col= 'filenames',
                            y_col='D_string',
                            batch_size=batch_size,
                            class_mode='categorical',
                            target_size=(150,84),
                            color_mode='rgb')

val_generator_Y = val_datagenerator.flow_from_dataframe(
                            dataframe= labels.loc[y_val, :],
                            directory= path + '/validation',
                            x_col= 'filenames',
                            y_col='Y_string',
                            batch_size=batch_size,
                            class_mode='categorical',
                            target_size=(150,84),
                            color_mode='rgb')

val_size = 0.16
test_size = 0.2
train_size = 1 - val_size - test_size


# Defining the model for D
inputs = Input(shape = (84,150, 3))
y = Conv2D(32, 5, activation='relu')(inputs)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)
y = Conv2D(64, 5, activation='relu')(y)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)

x = keras.layers.Flatten()(y)
x = Dense(128, activation= 'relu')(x)
outputs = Dense(5, activation='softmax')(x)

ConvMod_D = Model(inputs, outputs)
ConvMod_D.summary()
ConvMod_D.compile( optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

D_data_augmentation_fit = ConvMod_D.fit(train_generator_D,
                                            steps_per_epoch= 12000 * train_size // batch_size,
                                            epochs=25,
                                            validation_data =val_generator_D,
                                            validation_steps = 12000 * val_size // batch_size)

ConvMod_D.save('models/data_augmentation_D.h5')
np.save('data_augmentation_D_training.npy', D_data_augmentation_fit.history)

# Defining the model for CC
inputs = Input(shape = (84,150, 3))
y = Conv2D(32, 5, activation='relu')(inputs)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)
y = Conv2D(64, 5, activation='relu')(y)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)

x = keras.layers.Flatten()(y)
x = Dense(128, activation= 'relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

ConvMod_CC = Model(inputs, outputs)
ConvMod_CC.summary()
ConvMod_CC.compile( optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

CC_data_augmentation_fit = ConvMod_CC.fit(train_generator_CC,
                                            steps_per_epoch= 12000 * train_size // batch_size,
                                            epochs=15,
                                            validation_data =val_generator_CC,
                                            validation_steps = 12000 * val_size // batch_size)
ConvMod_CC.save('models/data_augmentation_CC.h5')
np.save('data_augmentation_CC_training.npy', CC_data_augmentation_fit.history)


# Defining the model for Y
inputs = Input(shape = (84,150, 3))
y = Conv2D(32, 5, activation='relu')(inputs)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)
y = Conv2D(64, 5, activation='relu')(y)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)

x = keras.layers.Flatten()(y)
x = Dense(128, activation= 'relu')(x)
outputs = Dense(11, activation='softmax')(x)

ConvMod_Y = Model(inputs, outputs)
ConvMod_Y.summary()
ConvMod_Y.compile( optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

Y_data_augmentation_fit = ConvMod_Y.fit(train_generator_Y,
                                            steps_per_epoch= 12000 * train_size // batch_size,
                                            epochs=15,
                                            validation_data =val_generator_Y,
                                            validation_steps = 12000 * val_size // batch_size)

ConvMod_Y.save('models/data_augmentation_Y.h5')
np.save('data_augmentation_Y_training.npy', Y_data_augmentation_fit.history)

'''

# Defining the test data generators
test_datagen_D = val_datagenerator.flow_from_dataframe(
                            dataframe= labels.loc[y_test, :],
                            directory= path + '/test',
                            x_col= 'filenames',
                            y_col='D_string',
                            batch_size=batch_size,
                            shuffle=False,
                            class_mode='categorical',
                            target_size=(150,84),
                            color_mode='rgb')
test_datagen_Y = val_datagenerator.flow_from_dataframe(
                            dataframe= labels.loc[y_test, :],
                            directory= path + '/test',
                            x_col= 'filenames',
                            y_col='Y_string',
                            batch_size=batch_size,
                            shuffle=False,
                            class_mode='categorical',
                            target_size=(150,84),
                            color_mode='rgb')
test_datagen_CC = val_datagenerator.flow_from_dataframe(
                            dataframe= labels.loc[y_test, :],
                            directory= path + '/test',
                            x_col= 'filenames',
                            y_col='CC_string',
                            batch_size=batch_size,
                            shuffle=False,
                            class_mode='binary',
                            target_size=(150,84),
                            color_mode='rgb')

'''