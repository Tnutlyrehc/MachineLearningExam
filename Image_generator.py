import numpy as np
import pandas as pd
import os
import re
from PIL import Image
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, MaxPool2D, GlobalAveragePooling2D, Concatenate
from tensorflow.keras import regularizers
path = 'C:/Users/felix/Documents/_FWM/Master/Semester 3/Applied Machine Learning/Exam/data'

# creating the labels from CSV file
labels = pd.read_csv(path + '/DIDA_12000_String_Digit_Labels.csv', names=['Index', 'Label'])

CC = []
D = []
Y = []
CC_string = []
for i in range(0, len(labels)):
    current_year = labels.iat[i, 1]
    digits = [int(x) for x in str(current_year)]
    # check for century
    if len(digits) > 4:
        CC.append(1)
        CC_string.append('not 18')
        D.append(10)
        Y.append(10)
    elif len(digits) < 4:
        CC.append(1)
        CC_string.append('not 18')
        D.append(digits[-2])
        Y.append(digits[-1])
    else:
        if digits[0] == 1 and digits[1] == 8:
            CC.append(0)
            CC_string.append('18')
        else:
            CC.append(1)
            CC_string.append('not 18')
        D.append(digits[-2])
        Y.append(digits[-1])
filenames = []
for i in labels['Index']:
    filenames.append(str(i) + '.jpg')
labels['CC'] = CC
labels['D'] = D
labels['Y'] = Y
labels['filenames'] = filenames
labels['CC_string'] = labels['CC'].astype(str)
labels['D_string'] = labels['D'].astype(str)
labels['Y_string'] = labels['Y'].astype(str)
print(labels.head())
raw_imgs = []

for i in range(1, 12001):
    current_image = image.load_img(os.path.join(path + '/original_data/' + str(i) + '.jpg'))
    raw_imgs.append(current_image)

# splitting the data randomly
indices = np.array(range(0, 12000))
X_train, X_test, y_train, y_test = train_test_split(raw_imgs, indices, random_state=42, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train ,random_state=42, test_size=0.2)

# writing it to the three directories
def write_to_dir (data, type, filenames):
    path = 'C:/Users/felix/Documents/_FWM/Master/Semester 3/Applied Machine Learning/Exam/data' + '/' + type
    for i in range(len(data)):
        image.save_img(path + '/' + str(filenames[i]) + '.jpg', data[i])

test_filename = np.array(y_test) + 1
train_filename = np.array(y_train) + 1
val_filename = np.array(y_val) + 1
write_to_dir(X_test, 'test', test_filename)
write_to_dir(X_train, 'train', train_filename)
write_to_dir(X_val, 'validation', val_filename)

batch_size=16
train_datagenerator = image.ImageDataGenerator(
                             rescale=1./255,
                             rotation_range=20,
                             zoom_range=0.2,
                             horizontal_flip=True)

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
'''
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

CC_data_augmentation_fit = ConvMod_CC.fit(train_generator_CC,
                                            steps_per_epoch= 12000 * train_size // batch_size,
                                            epochs=15,
                                            validation_data =val_generator_CC,
                                            validation_steps = 12000 * val_size // batch_size)

ConvMod_CC.save('CC_data_augmentation.h5')
'''
'''
# Defining the model for D
inputs = Input(shape = (84,150, 3))
y = Conv2D(6, 5, activation='relu')(inputs)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)
y = Conv2D(16, 5, activation='relu')(y)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)

x = keras.layers.Flatten()(y)
x = Dense(128, activation= 'relu', kernel_regularizer= regularizers.l2(0.01))(x)
outputs = Dense(11, activation='softmax', kernel_regularizer= regularizers.l2(0.01))(x)

ConvMod_D = Model(inputs, outputs)
ConvMod_D.summary()
ConvMod_D.compile( optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

D_data_augmentation_fit = ConvMod_D.fit(train_generator_D,
                                            steps_per_epoch= 12000 * train_size // batch_size,
                                            epochs=15,
                                            validation_data =val_generator_D,
                                            validation_steps = 12000 * val_size // batch_size)

ConvMod_D.save('D_data_augmentation.h5')
'''

# Defining the model for Y
inputs = Input(shape = (84,150, 3))
y = Conv2D(6, 5, activation='relu')(inputs)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)
y = Conv2D(16, 5, activation='relu')(y)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)

x = keras.layers.Flatten()(y)
x = Dense(128, activation= 'relu', kernel_regularizer= regularizers.l2(0.01))(x)
outputs = Dense(11, activation='softmax', kernel_regularizer= regularizers.l2(0.01))(x)

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

ConvMod_Y.save('D_data_augmentation.h5')