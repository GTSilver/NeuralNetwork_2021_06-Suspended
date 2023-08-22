print("Импорт библиотек...")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as mpl
import tensorflow as tf
import random as rand
import pickle as pic
import numpy as np
import pylab as pl
import cv2
import os
from keras.optimizers.optimizer_v2.gradient_descent import SGD
from keras.optimizers.optimizer_v2.adam import Adam
from keras.losses import MeanSquaredError
from keras.losses import MeanAbsoluteError
from keras.losses import mean_squared_error
from keras.models import Sequential
from keras.layers import Input, Dense, Conv1D, Conv2D, MaxPool1D, AvgPool1D, Dropout, Activation, Flatten, MaxPool2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score
from numpy import array, argmax, arange

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
