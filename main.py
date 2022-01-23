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
import random

# Assuming we're handed a CSV
labels_csv = pd.read_csv('Data\DIDA_12000_String_Digit_Labels.csv', encoding='utf-8', delimiter=(','))

#Getting the 0 to 9 data
numblist = []
path = 'Data/10000'

#Getting the images in the subfolders
for root, dirs, files in os.walk(path):
    for file in files:
        # append the file name to the list
        numblist.append(os.path.join(root, file))

# print all the file names
# for name in filelist:
#    print(name)

print(len(numblist))

#Split the images into 10 arrays (1000 in each)
splits = np.array_split(numblist, 10)

for array in splits:
    print(list(array))

print(len(splits))


#Getting the DIDA CCDY data
path_dida = 'Data/DIDA_1'
images = glob.glob(random.choice(path_dida))
random_image = random.choice(images)
