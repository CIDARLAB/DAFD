
class NearestDataPointModel:

	train_features = []
	train_labels = []

	def __init__(self, features, labels):
		self.train_features = features
		self.train_labels = labels

	def predict(self, features):
		min_dev = float("inf")
		min_index = -1
		for i, pot_match in enumerate(self.train_features):
			dev = sum([abs(pot_match[i] - features[i]) for i in range(len(features))])
			if dev < min_dev:
				min_dev = dev
				min_index = i

		return self.train_labels[min_index]
