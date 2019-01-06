from core_logic.ForwardModel import ForwardModel
from helper_scripts.ModelHelper import ModelHelper
import random


class ForwardModelTester:
	def __init__(self):
		self.MH = ModelHelper.get_instance() # type: ModelHelper

	def train(self):
		data_size = len(self.MH.train_features_dat)
		self.MH.make_train_data([x for x in range(data_size)])
		self.forward_model = ForwardModel()

	def cross_validate(self, folds):
		data_size = len(self.MH.train_features_dat)

		if folds == -1:
			folds = data_size

		group_size = int(data_size/folds)

		rand_indices = random.sample([x for x in range(data_size)], data_size)

		validations = {}
		for header in self.MH.output_headers:
			validations[header] = []
		for i in range(folds):
			train_indices = [x for x in (rand_indices[:i*group_size] + rand_indices[(i+1)*group_size:])]
			self.MH.make_train_data(train_indices)
			self.forward_model = ForwardModel()
			test_indices = rand_indices[i*group_size:(i+1)*group_size]
			test_dat = [self.MH.all_dat[x] for x in test_indices]
			for i,dat_point in enumerate(test_dat):
				ret_vals = self.validate_model(dat_point)
				for header in ret_vals:
					if ret_vals[header][1] > 1000:
						print(test_indices[i])
						print(ret_vals)
					validations[header].append(ret_vals[header])

		for header in validations:
			file_headers = ["actual_val" ,"pred_val" ,"deviation", "deviaiton_percent","actual_regime" ,"pred_regime" ,"chip_number"]
			with open("all_preds_" + header + ".csv" ,"w") as f:
				f.write(",".join(file_headers) + "\n")
				for x in validations[header]:
					f.write(",".join([str(xi) for xi in x]) + "\n")


	def validate_model(self, dat_point):
		""" Get test accuracies for a data point """
		features = [dat_point[x] for x in self.MH.input_headers]
		labels = {x:dat_point[x] for x in self.MH.output_headers}
		regime = dat_point["regime"]
		chip_number = dat_point["chip_number"]

		pred_vals = self.forward_model.predict(features)

		ret_val = {}
		for header in labels:
			actual_val = labels[header]
			pred_val = pred_vals[header]
			actual_regime = regime
			pred_regime = pred_vals["regime"]
			ret_val[header] = [actual_val, pred_val, abs(pred_val-actual_val), abs(pred_val-actual_val)/actual_val,
								actual_regime, pred_regime, chip_number]

		return ret_val

tester = ForwardModelTester()
#tester.cross_validate(2)
tester.train()

