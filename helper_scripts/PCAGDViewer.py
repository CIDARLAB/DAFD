""" Displays the results of gradient descent in the reverse model as a 2D figure via PCA
		Deprecated and not used in the final paper. Visualizations were not very informative.
"""

from helper_scripts.ModelHelper import ModelHelper
from sklearn.decomposition import PCA
from core_logic.InterModel import InterModel
import matplotlib.pyplot as plt
import numpy as np
import random

class PCAGDViewer:
	def __init__(self):
		self.MH = ModelHelper.get_instance()  # type: ModelHelper
		self.pca = PCA(n_components=2)
		self.pca.fit(self.MH.train_features_dat)

	def make_plot(self):
		plt.figure()

	def plot_point(self, gd_feature, pcolor="blue"):
		coords = self.pca.transform(np.asarray(gd_feature).reshape(1,-1))
		plt.plot(coords[0][0],coords[0][1], marker="o",color=pcolor)

	def plot_arrow(self, point_a, point_b):
		coords1 = self.pca.transform(np.asarray(point_a).reshape(1,-1))[0]
		coords2 = self.pca.transform(np.asarray(point_b).reshape(1,-1))[0]
		print(coords1)
		plt.arrow(coords1[0],coords1[1], coords2[0]-coords1[0], coords2[1] - coords1[1])


	def finish_plot(self):
		plt.show()

	def display_data(self, gd_features, gd_errors):
		plt.figure()

		#for features in self.MH.train_features_dat:
		#	print(features)
		#	coords = self.pca.transform(np.asarray(features).reshape(1,-1))
		#	print(coords)
		#	plt.plot(coords[0][0],coords[0][1], marker="o",color="blue")



		plt.show()


def distance(pointA, pointB):
	sum_dist = 0
	print(pointA)
	print(pointB)
	for i in range(len(pointA)):
		sum_dist += (pointA[i] - pointB[i])**2
	return sum_dist



viewer = PCAGDViewer()
inter_model = InterModel()
viewer.make_plot()

MH = ModelHelper.get_instance()	# type: ModelHelper
for i in MH.train_features_dat:
	viewer.plot_point(i,pcolor="green")

for i in range(15):
	gen_rate = random.randrange(50,450)
	drop_size = random.randrange(50,450)
	inter_model.interpolate({"generation_rate":gen_rate,"droplet_size":drop_size},{})
	viewer.plot_point(inter_model.first_val,pcolor="blue")
	viewer.plot_point(inter_model.last_point,pcolor="red")
	viewer.plot_arrow(inter_model.first_val, inter_model.last_point)
	print(distance(inter_model.first_val,inter_model.last_point))

print(distance(MH.train_features_dat[12], MH.train_features_dat[209]))

viewer.finish_plot()

