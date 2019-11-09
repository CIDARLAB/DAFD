""" This stand-alone script was used to get summary statistics from the model outputs """

import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt



def disp_graphs(csv_file):
	""" The CSV file is derived from the ForwardModelTester.py script. It essentially has the observed and predicted
	values for all points in a dataset (normally obtained via cross validation)"""
	print(csv_file)
	validations = []
	with open(csv_file,"r") as f:
		headers = [x for x in f.readline().strip().split(",")]
		for line in f:
			validations.append([float(x) for x in line.strip().split(",")])
		
	grs = sorted(validations, key=lambda x: x[5])

	total_num = len(grs)
	set_mean = sum([x[0] for x in grs])/total_num

	total_right = 0
	for x in grs:
		if int(x[4]) == int(x[5]):
			total_right+=1

	print("Classifier Accuracy: " + str(total_right/total_num))

	total_squared_pred_error = 0
	total_squared_mean_error = 0
	for x in grs:
		deviation = abs(x[0]-x[1])
		total_squared_pred_error += (deviation)**2
		total_squared_mean_error += (x[1]-set_mean)**2


	r2 = 1 - (total_squared_pred_error/total_squared_mean_error)
	print("Regressor R2 (no bias): " + str(r2))


	pred_vals = np.asarray([x[0] for x in grs])
	actual_vals = np.asarray([x[1] for x in grs])

	reg = LinearRegression().fit(pred_vals.reshape(-1,1),actual_vals.reshape(-1,1))
	print("Regressor R2 (bias): " + str(reg.score(pred_vals.reshape(-1,1),actual_vals.reshape(-1,1))))

	deviation_mean = sum([x[2] for x in grs])/total_num
	deviation_median = sorted(grs, key = lambda x: x[2])[total_num//2][2]
	print("Deviation mean: " + str(deviation_mean))
	print("Deviation median: " + str(deviation_median))

	deviation_percent_mean = sum([x[3] for x in grs])/total_num * 100
	deviation_percent_median = sorted(grs, key = lambda x: x[3])[total_num//2][3] * 100
	print("Deviation percent mean: " + str(deviation_percent_mean))
	print("Deviation percent median: " + str(deviation_percent_median))

	plt.figure()
	plt.title(csv_file)
	plt.plot(pred_vals,actual_vals,'o')


if __name__ == "__main__":
	csv_files = sys.argv[1:]
	for csv_file in csv_files:
		disp_graphs(csv_file)
		print()
	#plt.show()
