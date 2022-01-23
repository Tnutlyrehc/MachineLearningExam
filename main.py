import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import keras
from PIL import Image
import glob
import os

# Assuming we're handed a CSV
labels_csv = pd.read_csv('Data\DIDA_12000_String_Digit_Labels.csv', encoding='utf-8', delimiter=(','))

filelist = []
path = 'Data/10000'

for root, dirs, files in os.walk(path):
    for file in files:
        # append the file name to the list
        filelist.append(os.path.join(root, file))

# print all the file names
# for name in filelist:
#    print(name)

print(len(filelist))
