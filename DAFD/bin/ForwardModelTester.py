from DAFD.core_logic.ForwardModel import ForwardModel
from DAFD.helper_scripts.ModelHelper import ModelHelper
import random
import os


class ForwardModelTester:
	""" This class is used to test the accuracy of the forward models. It is simply a helpful utility and not a requirement
		for the function of the system"""

	def __init__(self):
		self.MH = ModelHelper.get_instance() # type: ModelHelper

	def train(self):
		""" Train the model and stop. Uses all data."""
		data_size = len(self.MH.train_features_dat_wholenorm)
		self.MH.make_train_data([x for x in range(data_size)])
		self.forward_model = ForwardModel()

	def cross_validate(self, folds):
		""" Typical cross-validation of data to determine accuracy
				folds is the division of the dataset (such as 10 for 10-fold CV)
		"""
		data_size = len(self.MH.all_dat)

		if folds == -1:
			folds = data_size # Leave one out cross validation

		group_size = int(data_size/folds)

		rand_indices = random.sample([x for x in range(data_size)], data_size)

		validations = {}
		for header in self.MH.output_headers:
			validations[header] = []

		# Go through each fold, train on everything else, and test it
		for i in range(folds):
			train_indices = [x for x in (rand_indices[:i*group_size] + rand_indices[(i+1)*group_size:])]
			self.MH.make_train_data(train_indices)
			self.forward_model = ForwardModel()
			test_indices = rand_indices[i*group_size:(i+1)*group_size]
			test_dat = [self.MH.all_dat[x] for x in test_indices]
			for i,dat_point in enumerate(test_dat):
				ret_vals = self.validate_model(dat_point)
				for header in ret_vals:
					validations[header].append(ret_vals[header]) # Validations dict says how well we did for the point

		# Data for the cross validation is written out at all_preds_droplet_size.csv and all_preds_generation_rate.csv
		# This data only has the shown file headers. For bulk statistics (like coefficient of determination), you will
		#  need to run the data through DAFD/model_data/disp_graphs.py. See that file for more information.
		for header in validations:
			file_headers = ["actual_val" ,"pred_val" ,"deviation", "deviation_percent","actual_regime" ,"pred_regime" ,"chip_number"]
			with open("all_preds_" + header + ".csv" ,"w") as f:
				f.write(",".join(file_headers) + "\n")
				for x in validations[header]:
					f.write(",".join([str(xi) for xi in x]) + "\n")


	def cross_validate_regime(self, folds, fileprefix=""):
		""" This class is pretty much the same as the normal cross validation class, but we assume that the regime
				classifier is 100% accurate. This allows us to determine the accuracy of our regressors more precisely"""
		data_size = len(self.MH.all_dat)

		regime1_points = [i for i in range(data_size) if self.MH.all_dat[i]["regime"] == 1]
		regime2_points = [i for i in range(data_size) if self.MH.all_dat[i]["regime"] == 2]


		random.shuffle(regime1_points)
		random.shuffle(regime2_points)

		validations = {}
		for header in self.MH.output_headers:
			validations[header] = []

		# Regime 1 validations
		for i in range(folds):
			group_size = int(len(regime1_points)/folds)
			train_indices = [x for x in (regime1_points[:i*group_size] + regime1_points[(i+1)*group_size:])]
			test_indices = regime1_points[i*group_size:(i+1)*group_size]
			self.MH.make_train_data(train_indices)
			self.MH.make_test_data(test_indices)
			self.forward_model = ForwardModel(should_generate_regime_classifier=False)
			test_dat = [self.MH.all_dat[x] for x in test_indices]
			for i,dat_point in enumerate(test_dat):
				ret_vals = self.validate_model(dat_point, given_regime=1)
				for header in ret_vals:
					validations[header].append(ret_vals[header])

		for header in validations:
			file_headers = ["actual_val" ,"pred_val" ,"deviation", "deviation_percent","actual_regime" ,"pred_regime" ,"chip_number"]
			with open("model_data/"+fileprefix + "r1_all_preds_" + header + ".csv" ,"w") as f:
				f.write(",".join(file_headers) + "\n")
				for x in validations[header]:
					f.write(",".join([str(xi) for xi in x]) + "\n")

		# I added the call to disp_graphs.py here to speed up testing
		os.system("python3 model_data/disp_graphs.py model_data/r1_all_preds_generation_rate.csv >> model_data/r1rate.txt")
		os.system("python3 model_data/disp_graphs.py model_data/r1_all_preds_droplet_size.csv >> model_data/r1size.txt")



		validations = {}
		for header in self.MH.output_headers:
			validations[header] = []

		# Now it is time for regime 2 validations
		for i in range(folds):
			group_size = int(len(regime2_points)/folds)
			train_indices = [x for x in (regime2_points[:i*group_size] + regime2_points[(i+1)*group_size:])]
			test_indices = regime2_points[i*group_size:(i+1)*group_size]
			self.MH.make_train_data(train_indices)
			self.MH.make_test_data(test_indices)
			self.forward_model = ForwardModel(should_generate_regime_classifier=False)
			test_dat = [self.MH.all_dat[x] for x in test_indices]
			for i,dat_point in enumerate(test_dat):
				ret_vals = self.validate_model(dat_point, given_regime=2)
				for header in ret_vals:
					validations[header].append(ret_vals[header])

		for header in validations:
			file_headers = ["actual_val" ,"pred_val" ,"deviation", "deviation_percent","actual_regime" ,"pred_regime" ,"chip_number"]
			with open("model_data/"+fileprefix + "r2_all_preds_" + header + ".csv" ,"w") as f:
				f.write(",".join(file_headers) + "\n")
				for x in validations[header]:
					f.write(",".join([str(xi) for xi in x]) + "\n")

		os.system("python3 model_data/disp_graphs.py model_data/r2_all_preds_generation_rate.csv >> model_data/r2rate.txt")
		os.system("python3 model_data/disp_graphs.py model_data/r2_all_preds_droplet_size.csv >> model_data/r2size.txt")

	def hold_out_classifier(self, hold_out_percent):
		""" Hold out accuracy tests for the regime classifier"""
		data_size = len(self.MH.all_dat)
		all_indices = [x for x in range(data_size)]
		random.seed(400)
		random.shuffle(all_indices)
		train_indices = all_indices[int(data_size*hold_out_percent):]
		test_indices = all_indices[:int(data_size*hold_out_percent)]

		self.MH.make_train_data(train_indices)
		self.MH.make_test_data(test_indices)

		validations = {}
		for header in self.MH.output_headers:
			validations[header] = []

		self.forward_model = ForwardModel(should_generate_regime_classifier=True)
		test_dat = [self.MH.all_dat[x] for x in test_indices]
		for i,dat_point in enumerate(test_dat):
			ret_vals = self.validate_model(dat_point)
			for header in ret_vals:
				validations[header].append(ret_vals[header])

		# We still print out everything, but we really only care about classifier accuracy for this section
		# You should use the method hold_out if you care about regressor accuracy
		for header in validations:
			file_headers = ["actual_val" ,"pred_val" ,"deviation", "deviation_percent","actual_regime" ,"pred_regime" ,"chip_number"]
			with open("model_data/all_preds_" + header + ".csv" ,"w") as f:
				f.write(",".join(file_headers) + "\n")
				for x in validations[header]:
					f.write(",".join([str(xi) for xi in x]) + "\n")

		os.system("python3 model_data/disp_graphs.py model_data/all_preds_generation_rate.csv >> model_data/rate.txt")
		os.system("python3 model_data/disp_graphs.py model_data/all_preds_droplet_size.csv >> model_data/size.txt")



	def hold_out(self, hold_out_percent, fileprefix=""):
		""" Hold out accuracy for our regressors. Assumes 100% accurate classifier"""
		data_size = len(self.MH.all_dat)

		regime1_points = [i for i in range(data_size) if self.MH.all_dat[i]["regime"] == 1]
		regime2_points = [i for i in range(data_size) if self.MH.all_dat[i]["regime"] == 2]


		random.shuffle(regime1_points)
		random.shuffle(regime2_points)

		validations = {}
		for header in self.MH.output_headers:
			validations[header] = []

		# Regime 1 accuracy tests
		train_indices = regime1_points[int(len(regime1_points)*hold_out_percent):]
		test_indices = regime1_points[:int(len(regime1_points)*hold_out_percent)]
		self.MH.make_train_data(train_indices)
		self.MH.make_test_data(test_indices)
		self.forward_model = ForwardModel(should_generate_regime_classifier=False)
		test_dat = [self.MH.all_dat[x] for x in test_indices]
		for i,dat_point in enumerate(test_dat):
			ret_vals = self.validate_model(dat_point, given_regime=1)
			for header in ret_vals:
				validations[header].append(ret_vals[header])

		for header in validations:
			file_headers = ["actual_val" ,"pred_val" ,"deviation", "deviation_percent","actual_regime" ,"pred_regime" ,"chip_number"]
			with open("model_data/"+fileprefix + "r1_all_preds_" + header + ".csv" ,"w") as f:
				f.write(",".join(file_headers) + "\n")
				for x in validations[header]:
					f.write(",".join([str(xi) for xi in x]) + "\n")

		os.system("python3 model_data/disp_graphs.py model_data/r1_all_preds_generation_rate.csv >> model_data/r1rate.txt")
		os.system("python3 model_data/disp_graphs.py model_data/r1_all_preds_droplet_size.csv >> model_data/r1size.txt")



		validations = {}
		for header in self.MH.output_headers:
			validations[header] = []

		# Regime 2 accuracy tests
		train_indices = regime2_points[int(len(regime2_points)*hold_out_percent):]
		test_indices = regime2_points[:int(len(regime2_points)*hold_out_percent)]
		self.MH.make_train_data(train_indices)
		self.MH.make_test_data(test_indices)
		self.forward_model = ForwardModel(should_generate_regime_classifier=False)
		test_dat = [self.MH.all_dat[x] for x in test_indices]
		for i,dat_point in enumerate(test_dat):
			ret_vals = self.validate_model(dat_point, given_regime=2)
			for header in ret_vals:
				validations[header].append(ret_vals[header])

		for header in validations:
			file_headers = ["actual_val" ,"pred_val" ,"deviation", "deviation_percent","actual_regime" ,"pred_regime" ,"chip_number"]
			with open("model_data/"+fileprefix + "r2_all_preds_" + header + ".csv" ,"w") as f:
				f.write(",".join(file_headers) + "\n")
				for x in validations[header]:
					f.write(",".join([str(xi) for xi in x]) + "\n")

		os.system("python3 model_data/disp_graphs.py model_data/r2_all_preds_generation_rate.csv >> model_data/r2rate.txt")
		os.system("python3 model_data/disp_graphs.py model_data/r2_all_preds_droplet_size.csv >> model_data/r2size.txt")


	def hold_out_double_test(self, hold_out_percent, fileprefix=""):
		""" This class was built for a very specific experiment

			I wanted to prove that if we ran the training a million times and chose the model with the best test
			accuracy, we wouldn't necessarily have chosen the best overall model, just one that happened to fit our
			specific test data really nicely.

			The experiment goes as follows:
				Choose 10% of data to hold out
				Choose another 10% of the data to hold out as a second set
				Train on the remaining 80%
				Check if there is rank correlation between models that do the best on hold out set 1 vs hold out set 2

			If the models that do best on hold out set 1 are also the models that do best on hold out set 2, then I was
			wrong and we should try to pick out the model that does best on testing.

			If there is no rank correlation (ie a model's accuracy on set 1 does not correlate with its accuracy on
			set 2), then it does not matter which model we choose as long as it converged during training.

			Turns out that there is not really any rank correlation, so we don't have to worry about choosing one model
			over another GIVEN THEY HAVE THE SAME HYPERPARAMETERS
			"""
		data_size = len(self.MH.all_dat)

		regime1_points = [i for i in range(data_size) if self.MH.all_dat[i]["regime"] == 1]
		regime2_points = [i for i in range(data_size) if self.MH.all_dat[i]["regime"] == 2]

		random.shuffle(regime1_points)
		random.shuffle(regime2_points)

		validations = {}
		for header in self.MH.output_headers:
			validations[header] = []

		train_indices = regime1_points[int(len(regime1_points)*hold_out_percent):]
		test_indices1 = regime1_points[:int(len(regime1_points)*hold_out_percent*0.5)]
		test_indices2 = regime1_points[int(len(regime1_points)*hold_out_percent*0.5):int(len(regime1_points)*hold_out_percent)]
		self.MH.make_train_data(train_indices)
		self.MH.make_test_data(test_indices1)
		self.forward_model = ForwardModel(should_generate_regime_classifier=False)

		test_dat = [self.MH.all_dat[x] for x in test_indices1]
		for i,dat_point in enumerate(test_dat):
			ret_vals = self.validate_model(dat_point, given_regime=1)
			for header in ret_vals:
				validations[header].append(ret_vals[header])

		for header in validations:
			file_headers = ["actual_val" ,"pred_val" ,"deviation", "deviation_percent","actual_regime" ,"pred_regime" ,"chip_number"]
			with open("model_data/"+fileprefix + "r1_all_preds_set1_" + header + ".csv" ,"w") as f:
				f.write(",".join(file_headers) + "\n")
				for x in validations[header]:
					f.write(",".join([str(xi) for xi in x]) + "\n")

		os.system("python3 model_data/disp_graphs.py model_data/r1_all_preds_set1_generation_rate.csv >> model_data/r1rate_set1.txt")
		os.system("python3 model_data/disp_graphs.py model_data/r1_all_preds_set1_droplet_size.csv >> model_data/r1size_set1.txt")

		validations = {}
		for header in self.MH.output_headers:
			validations[header] = []

		test_dat = [self.MH.all_dat[x] for x in test_indices2]
		for i,dat_point in enumerate(test_dat):
			ret_vals = self.validate_model(dat_point, given_regime=1)
			for header in ret_vals:
				validations[header].append(ret_vals[header])

		for header in validations:
			file_headers = ["actual_val" ,"pred_val" ,"deviation", "deviation_percent","actual_regime" ,"pred_regime" ,"chip_number"]
			with open("model_data/"+fileprefix + "r1_all_preds_set2_" + header + ".csv" ,"w") as f:
				f.write(",".join(file_headers) + "\n")
				for x in validations[header]:
					f.write(",".join([str(xi) for xi in x]) + "\n")

		os.system("python3 model_data/disp_graphs.py model_data/r1_all_preds_set2_generation_rate.csv >> model_data/r1rate_set2.txt")
		os.system("python3 model_data/disp_graphs.py model_data/r1_all_preds_set2_droplet_size.csv >> model_data/r1size_set2.txt")


		validations = {}
		for header in self.MH.output_headers:
			validations[header] = []

		train_indices = regime2_points[int(len(regime2_points)*hold_out_percent):]
		test_indices1 = regime2_points[:int(len(regime2_points)*hold_out_percent*0.5)]
		test_indices2 = regime2_points[int(len(regime2_points)*hold_out_percent*0.5):int(len(regime2_points)*hold_out_percent)]
		self.MH.make_train_data(train_indices)
		self.MH.make_test_data(test_indices1)
		self.forward_model = ForwardModel(should_generate_regime_classifier=False)

		test_dat = [self.MH.all_dat[x] for x in test_indices1]
		for i,dat_point in enumerate(test_dat):
			ret_vals = self.validate_model(dat_point, given_regime=2)
			for header in ret_vals:
				validations[header].append(ret_vals[header])

		for header in validations:
			file_headers = ["actual_val" ,"pred_val" ,"deviation", "deviation_percent","actual_regime" ,"pred_regime" ,"chip_number"]
			with open("model_data/"+fileprefix + "r2_all_preds_set1_" + header + ".csv" ,"w") as f:
				f.write(",".join(file_headers) + "\n")
				for x in validations[header]:
					f.write(",".join([str(xi) for xi in x]) + "\n")

		os.system("python3 model_data/disp_graphs.py model_data/r2_all_preds_set1_generation_rate.csv >> model_data/r2rate_set1.txt")
		os.system("python3 model_data/disp_graphs.py model_data/r2_all_preds_set1_droplet_size.csv >> model_data/r2size_set1.txt")

		validations = {}
		for header in self.MH.output_headers:
			validations[header] = []

		test_dat = [self.MH.all_dat[x] for x in test_indices2]
		for i,dat_point in enumerate(test_dat):
			ret_vals = self.validate_model(dat_point, given_regime=2)
			for header in ret_vals:
				validations[header].append(ret_vals[header])

		for header in validations:
			file_headers = ["actual_val" ,"pred_val" ,"deviation", "deviation_percent","actual_regime" ,"pred_regime" ,"chip_number"]
			with open("model_data/"+fileprefix + "r2_all_preds_set2_" + header + ".csv" ,"w") as f:
				f.write(",".join(file_headers) + "\n")
				for x in validations[header]:
					f.write(",".join([str(xi) for xi in x]) + "\n")

		os.system("python3 model_data/disp_graphs.py model_data/r2_all_preds_set2_generation_rate.csv >> model_data/r2rate_set2.txt")
		os.system("python3 model_data/disp_graphs.py model_data/r2_all_preds_set2_droplet_size.csv >> model_data/r2size_set2.txt")



	def validate_model(self, dat_point, given_regime=0):
		""" Get test accuracies for a data point """
		features = [dat_point[x] for x in self.MH.input_headers]
		labels = {x:dat_point[x] for x in self.MH.output_headers}
		regime = dat_point["regime"]
		chip_number = dat_point["chip_number"]

		pred_vals = self.forward_model.predict(features, regime=given_regime)

		ret_val = {}
		for header in labels:
			actual_val = labels[header]
			pred_val = pred_vals[header]
			actual_regime = regime
			pred_regime = pred_vals["regime"]
			ret_val[header] = [actual_val, pred_val, abs(pred_val-actual_val), abs(pred_val-actual_val)/actual_val,
								actual_regime, pred_regime, chip_number]

		return ret_val


# Put experiments here

tester = ForwardModelTester()
#tester.hold_out_classifier(0.2)
#tester.hold_out(0.2)
tester.train()
#tester.cross_validate_regime(folds=20)

