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


path = 'data'
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
        D.append(10)
        Y.append(10)
    elif len(digits) < 4:
        CC.append(1)
        if digits[-2] == 0 or digits[-2] == 1 or digits[-2] == 2 or digits[-2] == 3:
            D.append(digits[-2])
        else:
            D.append(10)
        Y.append(digits[-1])
    else:
        if digits[0] == 1 and digits[1] == 8:
            CC.append(0)
        else:
            CC.append(1)
        if digits[-2] == 0 or digits[-2] == 1 or digits[-2] == 2 or digits[-2] == 3:
            D.append(digits[-2])
        else:
            D.append(10)
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


raw_imgs = []

for i in range(1, 12001):
    current_image = image.load_img(os.path.join(path + '/original_data/' + str(i) + '.jpg'))
    raw_imgs.append(current_image)

# splitting the data randomly
indices = np.array(range(0, 12000))
X_train, X_test, y_train, y_test = train_test_split(raw_imgs, indices, random_state=42, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train ,random_state=42, test_size=0.2)
# writing it to three directories for data generators
def write_to_dir (data, type, filenames):
    path = 'data' + '/' + type
    for i in range(len(data)):
        image.save_img(path + '/' + str(filenames[i]) + '.jpg', data[i])

test_filename = np.array(y_test) + 1
train_filename = np.array(y_train) + 1
val_filename = np.array(y_val) + 1
write_to_dir(X_test, 'test', test_filename)
write_to_dir(X_train, 'train', train_filename)
write_to_dir(X_val, 'validation', val_filename)

# preparing numpy versions for 'normal' training
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
print(X_train.shape)
