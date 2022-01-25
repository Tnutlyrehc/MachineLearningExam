import glob

import numpy as np
import pandas as pd
import os
import re

import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, MaxPool2D, GlobalAveragePooling2D, Concatenate
from tensorflow.keras import regularizers
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import asarray

path = 'data'

# creating the labels from CSV file
#labels = pd.read_csv(path + '/DIDA_12000_String_Digit_Labels.csv', names=['Index', 'Label'])
"""
CC = []
D = []
Y = []

for i in range(0,len(labels)):
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
print('18 or not \n', labels['CC'].value_counts())
print('Decade \n', labels['D'].value_counts())
print('Year \n', labels['Y'].value_counts())"""

# reading the data to a list

raw_imgs = []

for i in os.listdir(path + '/original_data'):
    current_image = image.load_img(os.path.join(path + '/original_data', i))
    raw_imgs.append([current_image, i])

# splitting the data randomly
X_train, X_test = train_test_split(raw_imgs, random_state=42, test_size=0.2)
X_train, X_val= train_test_split(X_train, random_state=42, test_size=0.2)
print(len(X_train), len(X_val), len(X_test))

y_index_train = [int(re.sub("[^0-9]", "",item[1])) for item in X_train]
y_index_test = [int(re.sub("[^0-9]", "",item[1])) for item in X_test]
y_index_val = [int(re.sub("[^0-9]", "",item[1])) for item in X_val]
print(y_index_train)
#y_train = labels['Index' == X_train]
# writing it to the three directories
def write_to_dir (data, type):
    path = 'data'  + '/' + type
    for i in range(len(data)):
        image.save_img(os.path.join(path, data[i][1]), data[i][0])
write_to_dir(X_test, 'test')
write_to_dir(X_train, 'train')
write_to_dir(X_val, 'validation')

path = "data/test"
files = os.listdir(path)

images_test = glob.glob('*/*.jpg')
image_strings = [str(p) for p in images_test]

print(image_strings)