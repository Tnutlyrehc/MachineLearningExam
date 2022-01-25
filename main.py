import numpy as np
import pandas as pd
import os
import re
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

train_labels = labels.loc[labels['Index'].isin(y_index_train)]
test_labels = labels.loc[labels['Index'].isin(y_index_test)]
val_labels = labels.loc[labels['Index'].isin(y_index_val)]

# writing it to the three directories
def write_to_dir (data, type):
    path = 'C:/Users/felix/Documents/_FWM/Master/Semester 3/Applied Machine Learning/Exam/data'  + '/' + type
    for i in range(len(data)):
        image.save_img(os.path.join(path, data[i][1]), data[i][0])
write_to_dir(X_test, 'test')
write_to_dir(X_train, 'train')
write_to_dir(X_val, 'validation')

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
'''train_generator_CC = train_datagenerator.flow_from_directory(
                                                    path + '/train',
                                                    save_to_dir=path + '/data_augmentation',
                                                    batch_size=batch_size,
                                                    target_size=(150,84),
                                                    color_mode='rgb',
                                                    class_mode='binary')
'''

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


