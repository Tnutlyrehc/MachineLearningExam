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

from Image_generator import labels, test_datagen_Y, test_datagen_D, test_datagen_CC, y_test
from main import X_test as X_test_np

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
conf_matrix(filename_model='models/Y_data_augmentation.h5', test_data=test_datagen_Y, true_labels=labels.Y[y_test],
            filename_fig= 'plots/Y_data_aug_pred_conf_matrix.jpg', variable='Y', data_gen=True)