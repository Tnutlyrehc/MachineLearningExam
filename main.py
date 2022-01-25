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
path = 'C:/Users/felix/Documents/_FWM/Master/Semester 3/Applied Machine Learning/Exam/data'

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
sns.countplot(labels['CC'])
#plt.show()
# reading the data to a list
raw_imgs = []
CC_labels = labels['CC'].to_numpy()
print(CC_labels[7])
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
plot_loss_acc(Y_fit)
'''
# writing it to the three directories
def write_to_dir (data, type):
    path = 'C:/Users/felix/Documents/_FWM/Master/Semester 3/Applied Machine Learning/Exam/data'  + '/' + type
    for i in range(len(data)):
        image.save_img(os.path.join(path, data[i][1]), data[i][0])
write_to_dir(X_test, 'test')
write_to_dir(X_train, 'train')
write_to_dir(X_val, 'validation')'''
'''
batch_size=16
train_datagenerator = image.ImageDataGenerator(
                             rescale=1./255,
                             rotation_range=20,
                             zoom_range=0.2,
                             horizontal_flip=True)

train_generator_CC= train_datagenerator.flow_from_dataframe(
                            dataframe= train_labels['CC'],
                            directory= path + '/train',
                            x_col= "Filenames",
                            y_col=[0,1],
                            batch_size=batch_size,
                            class_mode="binary",
                            target_size=(150,84),
                            color_mode='rgb')
train_generator_CC = train_datagenerator.flow_from_directory(
                                                    path + '/train',
                                                    save_to_dir=path + '/data_augmentation',
                                                    batch_size=batch_size,
                                                    target_size=(150,84),
                                                    color_mode='rgb',
                                                    class_mode='binary')


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
    epochs=10)
'''

