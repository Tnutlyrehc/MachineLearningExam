import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing import image


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
print(labels.head())


# preparing numpy versions for 'normal' training
raw_imgs = []

for i in range(1, 12001):
    current_image = image.load_img(os.path.join(path + '/original_data/' + str(i) + '.jpg'))
    current_image = tf.image.resize(current_image, (84, 150))
    current_image = np.array(current_image) / 255
    raw_imgs.append(current_image)

#scaling the data
raw_imgs = np.array(raw_imgs)

# splitting the data randomly
indices = np.array(range(0, 12000))
X_train, X_test, y_train, y_test = train_test_split(raw_imgs, indices, random_state=42, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=42, test_size=0.2)
print(X_train.shape)

X_train = X_train.reshape(7680, 84 * 150 * 3)
X_val = X_val.reshape(1920, 84 * 150 * 3)

# Parameter grid to sample from during fitting, chooses a different combination on each iteration
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Instantiate the random search and fit it
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(
    estimator = rf,
    param_distributions = random_grid,
    n_iter = 10,
    cv = 3,
    verbose=2,
    random_state=42,
    n_jobs = -1)

# Fit the random search model
rf_random.fit(X_train, labels.CC[y_train])
print('Done fitting')
# review the best parameters from fitting the random search
rf_random.best_params_
print("Best parameters from random search: ")
print(rf_random.best_params_)

# Fitting and evaluating the model with the best parameters
X_train_val = np.concatenate((X_train, X_val))
print(X_train_val.shape)

Y_train_val = np.concatenate((y_train, y_val))
print(Y_train_val.shape)

final_forest_CC = RandomForestClassifier(
    n_estimators=555,
    min_samples_split=10,
    min_samples_leaf=1,
    max_features='auto',
    max_depth=80,
    bootstrap=False
)


final_forest_CC.fit(X_train_val, labels.CC[Y_train_val])

predictions = final_forest_CC.predict(X_test)
accuracy = accuracy_score(predictions, labels.CC[y_test])
print(accuracy)

