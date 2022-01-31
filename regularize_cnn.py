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


# Defining the early stopping
callbacks_list = [
	keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
]
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

CC_fit = ConvMod_CC.fit(X_train, labels.CC[y_train], epochs=25, batch_size=32,
                        callbacks=callbacks_list,
                        validation_data= (X_val, labels.CC[y_val]))
ConvMod_CC.save('models/regularize_early_CC.h5')
np.save('regularize_early_CC_training.npy', CC_fit.history)

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
# one hot encode labels for categorical_crossentropy loss
cat = OneHotEncoder()
one_hot_D_train = cat.fit_transform(np.array(labels.D[y_train]).reshape(-1, 1)).toarray()
print(one_hot_D_train.shape)
one_hot_D_val = cat.fit_transform(np.array(labels.D[y_val]).reshape(-1, 1)).toarray()
ConvMod_D.compile( optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

D_fit = ConvMod_D.fit(X_train,one_hot_D_train, epochs=25, batch_size=32,
                      callbacks=callbacks_list,
                      validation_data= (X_val, one_hot_D_val))
ConvMod_D.save('models/regularize_early_D.h5')
np.save('regularize_early_D_training.npy', D_fit.history)

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
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

Y_fit = ConvMod_Y.fit(X_train, labels.Y[y_train], epochs=25, batch_size=32,
                      callbacks=callbacks_list,
                      validation_data= (X_val, labels.Y[y_val]))
ConvMod_Y.save('models/regularize_early_Y.h5')
np.save('regularize_early_Y_training.npy', Y_fit.history)


# Dropout
# Defining the model for CC
inputs = Input(shape = (84,150, 3))
y = Conv2D(32, 5, activation='relu')(inputs)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)
y = Conv2D(64, 5, activation='relu')(y)
y = SpatialDropout2D(0.2)(y)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)

x = keras.layers.Flatten()(y)
x = Dense(128, activation= 'relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)

ConvMod_CC = Model(inputs, outputs)
ConvMod_CC.summary()
ConvMod_CC.compile( optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

CC_fit = ConvMod_CC.fit(X_train, labels.CC[y_train], epochs=25, batch_size=32,
                        validation_data= (X_val, labels.CC[y_val]))
ConvMod_CC.save('models/regularize_dropout_CC.h5')
np.save('regularize_dropout_CC_training.npy', CC_fit.history)


# Defining the model for D
inputs = Input(shape = (84,150, 3))
y = Conv2D(32, 5, activation='relu')(inputs)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)
y = Conv2D(64, 5, activation='relu')(y)
y = SpatialDropout2D(0.2)(y)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)

x = keras.layers.Flatten()(y)
x = Dense(128, activation= 'relu')(x)
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

D_fit = ConvMod_D.fit(X_train,one_hot_D_train, epochs=25, batch_size=32,
                      validation_data= (X_val, one_hot_D_val))
ConvMod_D.save('models/regularize_dropout_D.h5')
np.save('regularize_dropout_D_training.npy', D_fit.history)

# Defining the model for Y
inputs = Input(shape = (84,150, 3))
y = Conv2D(32, 5, activation='relu')(inputs)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)
y = Conv2D(64, 5, activation='relu')(y)
y = SpatialDropout2D(0.2)(y)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)

x = keras.layers.Flatten()(y)
x = Dense(128, activation= 'relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(11, activation='softmax')(x)

ConvMod_Y = Model(inputs, outputs)
ConvMod_Y.summary()
ConvMod_Y.compile( optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

Y_fit = ConvMod_Y.fit(X_train, labels.Y[y_train], epochs=25, batch_size=32,
                      validation_data= (X_val, labels.Y[y_val]))
ConvMod_Y.save('models/regularize_dropout_Y.h5')
np.save('regularize_dropout_Y_training.npy', Y_fit.history)

# weight decay l2
weight_decay = 0.001


# Defining the model for D
inputs = Input(shape = (84,150, 3))
y = Conv2D(32, 5, activation='relu', kernel_regularizer=l2(weight_decay))(inputs)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)
y = Conv2D(64, 5, activation='relu', kernel_regularizer=l2(weight_decay))(y)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)

x = keras.layers.Flatten()(y)
x = Dense(128, activation= 'relu', kernel_regularizer=l2(weight_decay))(x)
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

D_fit = ConvMod_D.fit(X_train,one_hot_D_train, epochs=25, batch_size=32, validation_data= (X_val, one_hot_D_val))
ConvMod_D.save('models/regularize_wd1_D.h5')
np.save('regularize_wd1_D_training.npy', D_fit.history)

# Defining the model for CC
inputs = Input(shape = (84,150, 3))
y = Conv2D(32, 5, activation='relu', kernel_regularizer=l2(weight_decay))(inputs)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)
y = Conv2D(64, 5, activation='relu', kernel_regularizer=l2(weight_decay))(y)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)

x = keras.layers.Flatten()(y)
x = Dense(128, activation= 'relu', kernel_regularizer=l2(weight_decay))(x)
outputs = Dense(1, activation='sigmoid')(x)

ConvMod_CC = Model(inputs, outputs)
ConvMod_CC.summary()
ConvMod_CC.compile( optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

CC_fit = ConvMod_CC.fit(X_train, labels.CC[y_train], epochs=25, batch_size=32, validation_data= (X_val, labels.CC[y_val]))
ConvMod_CC.save('models/regularize_wd1.h5')
np.save('regularize_wd1_CC_training.npy', CC_fit.history)

# Defining the model for Y
inputs = Input(shape = (84,150, 3))
y = Conv2D(32, 5, activation='relu', kernel_regularizer=l2(weight_decay))(inputs)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)
y = Conv2D(64, 5, activation='relu', kernel_regularizer=l2(weight_decay))(y)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)

x = keras.layers.Flatten()(y)
x = Dense(128, activation= 'relu', kernel_regularizer=l2(weight_decay))(x)
outputs = Dense(11, activation='softmax')(x)

ConvMod_Y = Model(inputs, outputs)
ConvMod_Y.summary()
ConvMod_Y.compile( optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

Y_fit = ConvMod_Y.fit(X_train, labels.Y[y_train], epochs=25, batch_size=32, validation_data= (X_val, labels.Y[y_val]))
ConvMod_Y.save('models/regularize_wd1_Y.h5')
np.save('regularize_wd1_Y_training.npy', Y_fit.history)