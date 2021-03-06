import numpy as np
import pandas as pd

import os
import re

import sklearn.metrics
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

from load_data import labels, y_test
path = 'data'
val_datagenerator = image.ImageDataGenerator(rescale=1/255)
batch_size= 16
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
print('test')
def conf_matrix(filename_model, filename_fig, test_data, true_labels, variable, data_gen = False):

    model = keras.models.load_model(filename_model)
    if data_gen is True:
        test_data.reset()

    pred_prob = model.predict(test_data)
    if variable == 'CC':
        pred_prob[pred_prob <= 0.5] = 0
        pred_prob[pred_prob > 0.5] = 1
        predicted_classes = pred_prob
    elif variable == 'D':
        predicted_class_indices = np.argmax(pred_prob, axis=1)
        predicted_classes = np.where(predicted_class_indices == 4, 10, predicted_class_indices)
    else:
        predicted_classes = np.argmax(pred_prob, axis=1)

    conf_mat = sklearn.metrics.confusion_matrix(true_labels, predicted_classes)
    if variable != 'CC' and data_gen==True:
        wild_card = conf_mat[:,2]
        conf_mat = np.column_stack((np.delete(conf_mat,2,1), wild_card))
    sns.heatmap(conf_mat, annot=True, fmt="d")
    plt.xlabel('Prediction')
    plt.ylabel('True')
    plt.savefig(filename_fig)

conf_matrix(filename_model='models/D_data_augmentation.h5', test_data=test_datagen_D, true_labels=labels.D[y_test],
            filename_fig= 'plots/D_data_aug_pred_conf_matrix.jpg', variable='D', data_gen=True)

def seq_acc (true_labels_CC, true_labels_D, true_labels_Y, pred_CC, pred_D, pred_Y):
    score = 0
    n = len(true_labels_CC)
    for i in range(0, n):
        if true_labels_CC[i] == pred_CC[i] and true_labels_D[i] == pred_D[i] and true_labels_Y[i] == pred_Y[i]:
            score = score + 1
    acc = score / n
    return acc

def char_acc (true_labels_CC, true_labels_D, true_labels_Y, pred_CC, pred_D, pred_Y):
    score = 0
    n = len(true_labels_CC)
    for i in range(0, n):
        if true_labels_CC[i] == pred_CC[i]:
            score = score + 1/3
        if true_labels_D[i] == pred_D[i]:
            score = score + 1/3
        if true_labels_Y[i] == pred_Y[i]:
            score = score + 1/3
    acc = score / n
    return acc

