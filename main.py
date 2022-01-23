import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import keras
from keras import utils

from PIL import Image
import glob
import os
import random
import os
import shutil
import splitfolders
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.preprocessing import image

# Assuming we're handed a CSV
df_labels = pd.read_csv('Data\DIDA_12000_String_Digit_Labels.csv', encoding='utf-8', delimiter=(','), header = None, names=["index", "string"])

df_labels['CC'] = 0
df_labels['D'] = 0
df_labels['Y'] = 0
df_labels = df_labels.astype(str)

for i, row in df_labels.iterrows():
    if len(row['string']) != 4:
        row['CC'] = '1'
        row['D'] = '10'
        row['Y'] = '10'
    else:
        row['D'] = row['string'][2]
        row['Y'] = row['string'][3]
        if row['string'][0:2] == '18':
            row['CC']='0'
        else:
            row['CC']='1'

print(df_labels)

df_labels.to_csv('12k_labeled.csv', index = False, sep = ',', header = True)


images = glob.glob("Data/DIDA_1/*.jpg")
for image in images:
    with open(image, 'rb') as file:
        img = Image.open(file)
        #img.show()

(x_train, y_train), (x_test, y_test) = images
print(x_train.shape, y_train.shape)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, df_labels)
y_test = keras.utils.to_categorical(y_test, df_labels)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#print(len(x_train))