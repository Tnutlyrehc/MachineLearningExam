import numpy as np
import pandas as pd
import os
import re
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, MaxPool2D, GlobalAveragePooling2D, SpatialDropout2D
from tensorflow.keras.regularizers import l2
from load_data import labels, X_train, X_test, X_val, y_train, y_test, y_val

path = 'data'

# data augmentation
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

wdCC = 0.001
wdD = 0.001
wdY = 0.001

# Defining the preferred model for CC
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

callbacks_list_CC = [
	keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
]

CC_fit =ConvMod_CC.fit(train_generator_CC,
						steps_per_epoch= 12000 * train_size // batch_size,
						epochs=10,
					   callbacks=callbacks_list_CC,
						validation_data =val_generator_CC,
						validation_steps = 12000 * val_size // batch_size)
ConvMod_CC.save('models/pref_mod_CC.h5')
np.save('pref_mod_CC_training.npy', CC_fit.history)

'''
# Defining the model for D
inputs = Input(shape = (84,150, 3))
y = Conv2D(32, 5, activation='relu', kernel_regularizer=l2(wdD))(inputs)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)
y = Conv2D(64, 5, activation='relu', kernel_regularizer=l2(wdD))(y)
y = SpatialDropout2D(0.2)(y)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)

x = keras.layers.Flatten()(y)
x = Dense(128, activation= 'relu', kernel_regularizer=l2(wdD))(x)
x = Dropout(0.5)(x)
outputs = Dense(5, activation='softmax')(x)

ConvMod_D = Model(inputs, outputs)
ConvMod_D.summary()
# one hot encode labels for categorical_crossentropy loss
cat = OneHotEncoder()
one_hot_D_train = cat.fit_transform(np.array(labels.D[y_train]).reshape(-1, 1)).toarray()
print(one_hot_D_train.shape)
one_hot_D_val = cat.fit_transform(np.array(labels.D[y_val]).reshape(-1, 1)).toarray()
ConvMod_D.compile( optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

callbacks_list_D = [
	keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
]
D_fit = ConvMod_D.fit(train_generator_D,
						steps_per_epoch= 12000 * train_size // batch_size,
						epochs=10,
					   callbacks=callbacks_list_D,
						validation_data =val_generator_D,
						validation_steps = 12000 * val_size // batch_size)
ConvMod_D.save('models/pref_mod_D.h5')
np.save('pref_mod_D_training.npy', D_fit.history)

# Defining the model for Y
inputs = Input(shape = (84,150, 3))
y = Conv2D(32, 5, activation='relu', kernel_regularizer=l2(wdY))(inputs)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)
y = Conv2D(64, 5, activation='relu', kernel_regularizer=l2(wdY))(y)
y = SpatialDropout2D(0.3)(y)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)

x = keras.layers.Flatten()(y)
x = Dense(128, activation= 'relu', kernel_regularizer=l2(wdY))(x)
x = Dropout(0.5)(x)
outputs = Dense(11, activation='softmax')(x)

ConvMod_Y = Model(inputs, outputs)
ConvMod_Y.summary()
ConvMod_Y.compile( optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

callbacks_list_Y = [
	keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
]

Y_fit = ConvMod_Y.fit(train_generator_Y,
						steps_per_epoch= 12000 * train_size // batch_size,
						epochs=10,
					   callbacks=callbacks_list_Y,
						validation_data =val_generator_Y,
						validation_steps = 12000 * val_size // batch_size)
ConvMod_Y.save('models/pref_mod_Y.h5')
np.save('pref_mod_Y_training.npy', Y_fit.history)'''