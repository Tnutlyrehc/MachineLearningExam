import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

#Assuming we're handed a CSV
data = pd.read_csv('thepath\csv.csv', encoding='utf-8', delimiter=(','), index_col=[0])

}