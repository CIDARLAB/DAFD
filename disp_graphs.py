import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt



def disp_graphs(csv_file):
	validations = []
	with open(csv_file,"r") as f:
		headers = [x for x in f.readline().strip().split(",")]
		for line in f:
			validations.append([float(x) for x in line.strip().split(",")])
		
	grs = sorted(validations,key = lambda x: x[5])

	total_num = len(grs)
	set_mean = sum([x[0] for x in grs])/total_num

	total_right = 0
	for x in grs:
		if int(x[3]) == int(x[4]):
			total_right+=1

	print("SVM Accuracy: " + str(total_right/total_num))

	total_squared_pred_error = 0
	total_squared_mean_error = 0
	for x in grs:
		deviation = abs(x[0]-x[1])
		total_squared_pred_error += (deviation)**2
		total_squared_mean_error += (x[1]-set_mean)**2


	r2 = 1 - (total_squared_pred_error/total_squared_mean_error)
	print("SVR R2 (no bias): " + str(r2))


	pred_vals = np.asarray([x[0] for x in grs])
	actual_vals = np.asarray([x[1] for x in grs])

	reg = LinearRegression().fit(pred_vals.reshape(-1,1),actual_vals.reshape(-1,1))
	print("SVR R2 (bias): " + str(reg.score(pred_vals.reshape(-1,1),actual_vals.reshape(-1,1))))


	plt.figure()
	plt.title(csv_file)
	plt.plot(pred_vals,actual_vals,'o')

	plt.show()

if __name__ == "__main__":
	csv_file = sys.argv[1]
	disp_graphs(csv_file)
