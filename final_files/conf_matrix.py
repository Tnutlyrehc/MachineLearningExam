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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, MaxPool2D, GlobalAveragePooling2D, Concatenate
from tensorflow.keras import regularizers
from load_data import labels, y_test, y_train, y_val

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
test_datagen_Y = val_datagenerator.flow_from_dataframe(
                            dataframe= labels.loc[y_test, :],
                            directory= path + '/test',
                            x_col= 'filenames',
                            y_col='Y_string',
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

conf_matrix(filename_model='models/data_augmentation_CC.h5', test_data=test_datagen_CC, true_labels=labels.CC[y_test],
            filename_fig= 'plots/CC_data_aug_pred_conf_matrix.jpg', variable='CC', data_gen=True)
conf_matrix(filename_model='models/data_augmentation_D.h5', test_data=test_datagen_D, true_labels=labels.D[y_test],
            filename_fig= 'plots/D_data_aug_pred_conf_matrix.jpg', variable='D', data_gen=True)
conf_matrix(filename_model='models/data_augmentation_Y.h5', test_data=test_datagen_Y, true_labels=labels.Y[y_test],
            filename_fig= 'plots/Y_data_aug_pred_conf_matrix.jpg', variable='Y', data_gen=True)

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

# code for getting the accuracy metrics
CC_mod = keras.models.load_model('models/data_augmentation_CC.h5')
pred_prob_CC = CC_mod.predict(test_datagen_CC)
pred_prob_CC[pred_prob_CC <= 0.5] = 0
pred_prob_CC[pred_prob_CC > 0.5] = 1

D_mod = keras.models.load_model('models/data_augmentation_D.h5')
pred_prob_D = D_mod.predict(test_datagen_D)
predicted_class_indices_D = np.argmax(pred_prob_D, axis=1)
for i in range(0, len(predicted_class_indices_D)):
    if predicted_class_indices_D[i] == 4:
        predicted_class_indices_D[i] = 10

Y_mod = keras.models.load_model('models/data_augmentation_Y.h5')
pred_prob_Y = Y_mod.predict(test_datagen_Y)
predicted_class_indices_Y = np.argmax(pred_prob_Y, axis=1)

predicted_classes_CC = pred_prob_CC
predicted_classes_D = predicted_class_indices_D
predicted_classes_Y = predicted_class_indices_Y

true_labels_CC = np.array(labels.CC[y_test])
true_labels_D = np.array(labels.D[y_test])
true_labels_Y =  np.array(labels.Y[y_test])

seq_acc = seq_acc(true_labels_CC, true_labels_D, true_labels_Y, predicted_classes_CC, predicted_classes_D, predicted_classes_Y)
char_acc =  char_acc(true_labels_CC, true_labels_D, true_labels_Y, predicted_classes_CC, predicted_classes_D, predicted_classes_Y)


print('sequence acc: ', seq_acc, '\n character acc:', char_acc)
score_CC, acc_CC = CC_mod.evaluate(test_datagen_CC)
print('acc for CC:', acc_CC)
score_D , acc_D = D_mod.evaluate(test_datagen_D)
print('acc for D:', acc_D)
score_Y , acc_Y = Y_mod.evaluate(test_datagen_Y)
print('acc for Y:', acc_Y)

# code for evaluating RF
RF_CC = np.load('working_npy/CC_RF_PRED_TEST.npy')
RF_D = np.load('working_npy/D_RF_PRED_TEST.npy')
RF_Y = np.load('working_npy/Y_RF_PRED_TEST.npy')
true_labels_CC = np.array(labels.CC[y_test])
true_labels_D = np.array(labels.D[y_test])
true_labels_Y =  np.array(labels.Y[y_test])

seq_acc_RF_test = seq_acc(true_labels_CC, true_labels_D, true_labels_Y, RF_CC, RF_D, RF_Y)
char_acc_RF_test = char_acc(true_labels_CC, true_labels_D, true_labels_Y, RF_CC, RF_D, RF_Y)
