import numpy as np
import pandas as pd
import os
import re
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, MaxPool2D, GlobalAveragePooling2D, Concatenate
from tensorflow.keras import regularizers
from load_data import labels, X_train, X_test, X_val, y_train,y_test, y_val
from cnn_model import build_cnn_model_default
path = 'data'
'''
# creating the labels from CSV file
labels = pd.read_csv(path + '/DIDA_12000_String_Digit_Labels.csv', names=['Index', 'Label'])

CC = []
D = []
Y = []

for i in range(0, len(labels)):
    current_year = labels.iat[i, 1]
    digits = [int(x) for x in str(current_year)]
    # check for century
    if len(digits) > 4:
        CC.append(1)
        D.append(10)
        Y.append(10)
    elif len(digits) < 4:
        CC.append(1)
        D.append(digits[-2])
        Y.append(digits[-1])
    else:
        if digits[0] == 1 and digits[1] == 8:
            CC.append(0)
        else:
            CC.append(1)
        D.append(digits[-2])
        Y.append(digits[-1])

labels['CC'] = CC
labels['D'] = D
labels['Y'] = Y

# exploratory data analysis - summary statistics

ax = sns.countplot(labels['Y'])
plt.bar_label(ax.containers[0], label_type='edge')
plt.title('Amount of representations of classes for the Y variable')
plt.savefig('Y_barplot.jpg')
plt.title('Amount of representations of classes for the Y variable')
plt.savefig('Y_barplot.jpg')
# reading the data to a list
raw_imgs = []

for i in range(1, 12001):
    current_image = image.load_img(os.path.join(path + '/original_data/' + str(i) + '.jpg'))
    current_image = tf.image.resize(current_image, (84, 150))
    raw_imgs.append(current_image)

#scaling the data
raw_imgs = np.array(raw_imgs) / 255
# splitting the data randomly
indices = np.array(range(0, 12000))
X_train, X_test, y_train, y_test = train_test_split(raw_imgs, indices, random_state=42, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train ,random_state=42, test_size=0.2)

CC_train_labels = labels.loc[y_train, 'CC']
CC_val_labels = labels.loc[y_val, 'CC']
CC_test_labels = labels.loc[y_test, 'CC']

D_train_labels = labels.loc[y_train, 'D']
D_val_labels = labels.loc[y_val, 'D']
D_test_labels = labels.loc[y_test, 'D']

Y_train_labels = labels.loc[y_train, 'Y']
Y_val_labels = labels.loc[y_val, 'Y']
Y_test_labels = labels.loc[y_test, 'Y']
'''
"""
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

CC_fit = ConvMod_CC.fit(X_train, CC_train_labels, epochs=15, batch_size=32, validation_data= (X_val, CC_val_labels))
results = ConvMod_CC.evaluate(X_test, CC_test_labels,  batch_size=16)
ConvMod_CC.save('CC.h5')


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
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

D_fit = ConvMod_D.fit(X_train, D_train_labels, epochs=15, batch_size=32, validation_data= (X_val, D_val_labels))
results = ConvMod_D.evaluate(X_test, D_test_labels,  batch_size=16)
ConvMod_D.save('D.h5')

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
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

Y_fit = ConvMod_Y.fit(X_train, Y_train_labels, epochs=15, batch_size=32, validation_data= (X_val, Y_val_labels))
results = ConvMod_Y.evaluate(X_test, Y_test_labels,  batch_size=16)
ConvMod_Y.save('Y.h5')

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
plot_loss_acc(Y_fit)"""

# Defining the model for CC
inputs = Input(shape = (84,150, 3))
y = Conv2D(6, 5, activation='relu')(inputs)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)
y = Conv2D(16, 5, activation='relu')(y)
y = MaxPool2D(pool_size=(2,2), strides=(2,2))(y)

x = keras.layers.Flatten()(y)
x = Dense(128, activation= 'relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

ConvMod_CC = Model(inputs, outputs)
ConvMod_CC.summary()
ConvMod_CC.compile( optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

CC_fit = ConvMod_CC.fit(X_train, labels.CC[y_train], epochs=15, batch_size=32, validation_data= (X_val, labels.CC[y_val]))

np.save('my_history.npy', CC_fit.history)

'''
CC_model = build_cnn_model_default(True, False, False)
hist = CC_model.fit(X_train, labels.CC[y_train], epochs=15, validation_data=(X_val,labels.CC[y_val]), batch_size=16)'''