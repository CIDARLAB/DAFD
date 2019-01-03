"""
Created on Fri Nov 23 19:05:38 2018

@author: noushinm
"""
from keras import metrics
from keras.layers import Dense, Activation
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
# Reading an excel file using Python
from sklearn import model_selection
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
#from keras.utils.vis_utils import plot_model
import sys
from sklearn.neural_network import MLPRegressor
import numpy as np


# root mean squared error (rmse) for regression
def rmse(y_true, y_pred):
	from keras import backend
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


# mean squared error (mse) for regression
def mse(y_true, y_pred):
	from keras import backend
	return backend.mean(backend.square(y_pred - y_true), axis=-1)


# coefficient of determination (R^2) for regression
def r_square(y_true, y_pred):
	from keras import backend as K
	SS_res =  K.sum(K.square(y_true - y_pred))
	SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
	return (1 - SS_res/(SS_tot + K.epsilon()))


def r_square_loss(y_true, y_pred):
	from keras import backend as K
	SS_res =  K.sum(K.square(y_true - y_pred))
	SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
	return 1 - ( 1 - SS_res/(SS_tot + K.epsilon()))


class NeuralNetModel_keras:

	regression_model = None

	def __init__(self, features, labels):
		# Initialising the ANN
		self.regression_model = Sequential()

		# Adding the input layer and the first hidden layer
		self.regression_model.add(Dense(16, activation = 'relu', input_dim = 8))

		# Adding the second hidden layer
		self.regression_model.add(Dense(units = 16, activation = 'relu'))

		# Adding the third hidden layer
		self.regression_model.add(Dense(units = 16, activation = 'relu'))

		# Adding the 4th hidden layer
		self.regression_model.add(Dense(units = 8, activation = 'relu'))

		# Adding the output layer
		self.regression_model.add(Dense(units = 1))

		# Compiling the NN
		self.regression_model.compile(optimizer = 'nadam', loss = 'mean_squared_error',metrics=['mean_squared_error', rmse, r_square] )#metrics=[metrics.mae, metrics.categorical_accuracy]

		earlystopping=EarlyStopping(monitor="mean_squared_error", patience=20, verbose=1, mode='auto')

		# Fitting the NN to the Training set
		train_features = np.stack(features)
		train_labels = np.stack(labels)
		self.regression_model.fit(train_features, train_labels, batch_size = 10, epochs = 500, callbacks=[earlystopping])#

	def predict(self, features):
		return self.regression_model.predict(np.asarray(features).reshape(1, -1))[0]

