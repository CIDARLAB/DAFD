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


class NeuralNetModel_regime:

	classifier_model = None

	def train_model(self, features, labels):
		self.classifier_model = Sequential()
		self.classifier_model.add(Dense(32, input_dim=8, activation='relu'))
		self.classifier_model.add(Dense(16, activation='relu'))
		self.classifier_model.add(Dense(16, activation='relu'))
		self.classifier_model.add(Dense(1, activation='sigmoid'))

		self.classifier_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
		earlystopping = EarlyStopping(monitor="binary_crossentropy", patience=20, verbose=1, mode='auto')

		train_features = np.stack(features)
		train_labels = np.stack(labels)-1

		# Fitting the NN to the Training set
		self.classifier_model.fit(train_features, train_labels, batch_size=30, epochs=300, callbacks=[earlystopping],verbose=1)#

		# serialize model to JSON
		model_json = self.classifier_model.to_json()
		with open(os.path.dirname(os.path.abspath(__file__)) + "/saved/classifier.json", "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		self.classifier_model.save_weights(os.path.dirname(os.path.abspath(__file__)) + "/saved/classifier.h5")

	def load_model(self):
		# load json and create model
		json_file = open(os.path.dirname(os.path.abspath(__file__)) + "/saved/classifier.json", 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		loaded_model.load_weights(os.path.dirname(os.path.abspath(__file__)) + "/saved/classifier.h5")
		self.classifier_model = loaded_model

	def predict(self, features):
		return self.classifier_model.predict_classes(np.asarray(features).reshape(1, -1))[0][0]+1

