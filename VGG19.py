import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
import tensorflow as tf
from load_data import X_train, X_test, X_val, y_train, y_test, y_val, labels



vgg19 = VGG19(weights = 'imagenet',
              include_top = False,
              input_shape=(84, 150, 3)
              )

CC_model = Sequential()
CC_model.add(vgg19)
CC_model.add(Flatten())
CC_model.add(Dense(1, activation='sigmoid'))
CC_model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

CC_model.summary()

history = CC_model.fit(X_train, labels.CC[y_train],
                    epochs=20,
                    batch_size=32,
                    validation_data=(X_test, y_test))

fig, axes=plt.subplots(nrows=1, ncols=2, figsize=(16,6))

axes[0].plot(history.history['accuracy'])
axes[0].plot(history.history['val_accuracy'])
axes[0].set_title('Model Accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend(['train', 'validation'], loc='upper left')

axes[1].plot(history.history['loss'])
axes[1].plot(history.history['val_loss'])
axes[1].set_title('Model Loss')
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend(['train', 'validation'], loc='upper left')

plt.show()


"""
(7680, 84, 150, 3)
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 vgg19 (Functional)          (None, 2, 4, 512)         20024384  
                                                                 
 flatten (Flatten)           (None, 4096)              0         
                                                                 
 dense (Dense)               (None, 1)                 4097      
                                                                 
=================================================================
Total params: 20,028,481
Trainable params: 20,028,481
Non-trainable params: 0
_________________________________________________________________
Epoch 1/20
240/240 [==============================] - 837s 3s/step - loss: 0.0933 - accuracy: 0.9717 - val_loss: 37984.3633 - val_accuracy: 4.1667e-04
Epoch 2/20
240/240 [==============================] - 813s 3s/step - loss: 0.0378 - accuracy: 0.9900 - val_loss: 40820.7930 - val_accuracy: 4.1667e-04
Epoch 3/20
240/240 [==============================] - 799s 3s/step - loss: 0.0234 - accuracy: 0.9936 - val_loss: 54834.8398 - val_accuracy: 4.1667e-04
Epoch 4/20
  7/240 [..............................] - ETA: 11:58 - loss: 0.0020 - accuracy: 1.0000
  
  
  VGG19 WTF 


"""