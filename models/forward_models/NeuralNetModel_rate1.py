"""
Created on Fri Nov 23 19:05:38 2018

@author: noushinm
"""
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import model_from_json
import numpy as np
import os


# root mean squared error (rmse) for regression
def rmse(y_true, y_pred):
	from tensorflow.keras import backend
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


# mean squared error (mse) for regression
def mse(y_true, y_pred):
	from tensorflow.keras import backend
	return backend.mean(backend.square(y_pred - y_true), axis=-1)


# coefficient of determination (R^2) for regression
def r_square(y_true, y_pred):
	from tensorflow.keras import backend as K
	SS_res =  K.sum(K.square(y_true - y_pred))
	SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
	return (1 - SS_res/(SS_tot + K.epsilon()))


def r_square_loss(y_true, y_pred):
	from tensorflow.keras import backend as K
	SS_res =  K.sum(K.square(y_true - y_pred))
	SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
	return 1 - ( 1 - SS_res/(SS_tot + K.epsilon()))


class NeuralNetModel_rate1:

	regression_model = None

	def train_model(self, output_name, regime, features, labels):
		model_name = output_name + str(regime)
		print(model_name)

		# Initialising the ANN
		self.regression_model = Sequential()

		# Adding the input layer and the first hidden layer
		self.regression_model.add(Dense(32, activation='relu', input_dim=8))

		# Adding the second hidden layer
		self.regression_model.add(Dense(units=16, activation='relu'))
		# self.regression_model.add(Dropout(0.1))
		# Adding the third hidden layer
		# model.add(Dense(units = 32, activation = 'relu'))
		# model.add(Dropout(0.1))
		# Adding the third hidden layer
		self.regression_model.add(Dense(units=16, activation='relu'))
		# model.add(Dropout(0.1))

		# Adding the 4th hidden layer
		self.regression_model.add(Dense(units=8, activation='relu'))

		# Adding the output layer
		self.regression_model.add(Dense(units=1))

		# Compiling the NN
		self.regression_model.compile(optimizer='nadam', loss='mean_squared_error', metrics=['mean_squared_error', rmse,
																			 r_square])  # metrics=[metrics.mae, metrics.categorical_accuracy]

		earlystopping = EarlyStopping(monitor="mean_squared_error", patience=30, verbose=1, mode='auto')


		# Fitting the NN to the Training set
		train_features = np.stack(features)
		train_labels = np.stack(labels)


		# Fitting the NN to the Training set
		# 20, 500 for all others
		self.regression_model.fit(train_features, train_labels, batch_size = 10, epochs = 400, callbacks=[earlystopping],verbose=0)#

		# serialize model to JSON
		model_json = self.regression_model.to_json()
		with open(os.path.dirname(os.path.abspath(__file__)) + "/saved/" + model_name + ".json", "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		self.regression_model.save_weights(os.path.dirname(os.path.abspath(__file__)) + "/saved/" + model_name + ".h5")

	def load_model(self, output_name, regime):
		model_name = output_name + str(regime)

		# load json and create model
		json_file = open(os.path.dirname(os.path.abspath(__file__)) + "/saved/" + model_name + ".json", 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		loaded_model.load_weights(os.path.dirname(os.path.abspath(__file__)) + "/saved/" + model_name + ".h5")
		self.regression_model = loaded_model

	def predict(self, features):
		return self.regression_model.predict(np.asarray(features).reshape(1, -1))[0]

