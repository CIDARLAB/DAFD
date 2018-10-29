import tensorflow as tf
import pickle
import os
import numpy as np
import pandas as pd



def min_max_normalize(x, set_min, set_max):
	return (x-set_min)/(set_max-set_min)

def min_max_denormalize(x, set_min, set_max):
	return (x*(set_max-set_min)) + set_min


class ForwardModel:
	def __init__(self):
		if os.path.isfile("NN_build/data_meta.p"):
			pickle_dict = pickle.load(open("NN_build/data_meta.p", "rb"))
			self.maxes_dict = pickle_dict["maxes_dict"]
			self.mins_dict = pickle_dict["mins_dict"]
			self.label_names = pickle_dict["label_names"]
			self.feature_names = pickle_dict["feature_names"]
		else:
			self.init_train()
			

	def normalize_array(self, np_arr, is_features):
		df_nonnormal = pd.DataFrame(np_arr)
		df_nonnormal.columns = self.feature_names if is_features else self.label_names
		return np.array(self.normalize_dataframe(df_nonnormal))

	def denormalize_array(self, np_arr, is_features):
		df_nonnormal = pd.DataFrame(np_arr)
		df_nonnormal.columns = self.feature_names if is_features else self.label_names
		return np.array(self.denormalize_dataframe(df_nonnormal))

	def normalize_dataframe(self, df):
		global normalize_dict
		new_df = pd.DataFrame()
		for col in df.columns:
			new_df[col] = df[col].apply(min_max_normalize, set_min=self.mins_dict[col], set_max=self.maxes_dict[col])
		return new_df

	def denormalize_dataframe(self, df):
		global denormalize_dict
		new_df = pd.DataFrame()
		for col in df.columns:
			new_df[col] = df[col].apply(min_max_denormalize, set_min=self.mins_dict[col], set_max=self.maxes_dict[col])
		return new_df

	def init_train(self):
		label_ct = 2
		feature_ct = 8
		test_number = 100

		batch_num_train = 1024
		neuron_ct = 100
		learning_rate = 0.01
		steps_to_train = 10000




		dat_df = pd.read_csv("ExperimentalResults.csv")
		dat_df = dat_df.sample(frac=1).reset_index(drop=True)
		labels = dat_df.iloc[test_number:,-label_ct:]
		features = dat_df.iloc[test_number:,:-label_ct]


		self.maxes_dict = {}
		self.mins_dict = {}

		maxes = np.max(labels,axis=0)
		mins = np.min(labels,axis=0)
		for i,col in enumerate(labels.columns):
			self.maxes_dict[col] = maxes[i]
			self.mins_dict[col] = mins[i]

		maxes = np.max(features,axis=0)
		mins = np.min(features,axis=0)
		for i,col in enumerate(features.columns):
			self.maxes_dict[col] = maxes[i]
			self.mins_dict[col] = mins[i]



		self.label_names = list(labels.columns)
		self.feature_names = list(features.columns)

		pickle_dict = {"maxes_dict": self.maxes_dict,
				"mins_dict": self.mins_dict,
				"label_names": self.label_names,
				"feature_names": self.feature_names}


		labels = self.normalize_dataframe(labels)
		features = self.normalize_dataframe(features)

		train_ds = tf.data.Dataset.from_tensor_slices((labels,features))
		train_iterator = train_ds.shuffle(1000).repeat().batch(batch_num_train).make_one_shot_iterator()
		train_next_element = train_iterator.get_next()


		labels = dat_df.iloc[:test_number,-label_ct:]
		features = dat_df.iloc[:test_number,:-label_ct]

		test_ds = tf.data.Dataset.from_tensor_slices((labels,features))
		test_iterator = test_ds.shuffle(1000).repeat().batch(test_number).make_one_shot_iterator()
		test_next_element = test_iterator.get_next()



		# INPUT FEATURES VECTOR
		x = tf.placeholder(tf.float32, [None, feature_ct],name="in_feats")

		# INPUT LABELS VECTOR
		y_ = tf.placeholder(tf.float32, [None, label_ct])



		h1 = tf.layers.dense(x, neuron_ct, activation=tf.nn.relu)

		h2 = tf.layers.dense(h1, neuron_ct, activation=tf.nn.relu)

		h3 = tf.layers.dense(h2, neuron_ct, activation=tf.nn.relu)

		y = tf.layers.dense(h3, label_ct, activation=None, name="predictions")


		# LOSS CALCULATIONS
		MSE_loss = tf.losses.mean_squared_error(predictions=y,labels=y_)
		loss_val = MSE_loss
		#loss_val = tf.norm(y_-y)

		train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(MSE_loss)

		# SET UP SESSION
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True

		sess = tf.Session(config=config)
		sess.run(tf.global_variables_initializer())


		for i in range(steps_to_train):
			ys_in,xs_in = sess.run(train_next_element)
			_, loss_out = sess.run([train_step,loss_val], feed_dict={x: xs_in, y_: ys_in})
			if i % 100 == 0:
				y_out = sess.run(y,feed_dict={x: xs_in, y_: ys_in})
				MSE_out = sess.run(MSE_loss,feed_dict={x: xs_in, y_: ys_in})

				##a = [75,1.5,3,1.5,2.5,2.5,0.045,10]
				#a = [100,2.5,6,1,3,2.5,0.36,10]
				#a_norm = self.normalize_array([a],True)
				#y_out2,MSE_out2 = sess.run([y,MSE_loss],feed_dict={x: a_norm, y_:[[0.5]*2]})
				#y2_res = str(self.denormalize_array(y_out2,False))

				print(str(i) + " " + str(loss_out) + " " + str(np.linalg.norm(ys_in-y_out)))


		# create model builder
		builder = tf.saved_model.builder.SavedModelBuilder("NN_build")

		# create tensors info
		predict_tensor_inputs_info = tf.saved_model.utils.build_tensor_info(x)
		predict_tensor_outs_info = tf.saved_model.utils.build_tensor_info(y)

		# build prediction signature
		prediction_signature = (
			tf.saved_model.signature_def_utils.build_signature_def(
				inputs={'input_feats': predict_tensor_inputs_info},
				outputs={'output_vals': predict_tensor_outs_info},
				method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
				)
		)

		# save the model
		legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
		builder.add_meta_graph_and_variables(
		sess, [tf.saved_model.tag_constants.SERVING],
		signature_def_map={
		    'predict_feats': prediction_signature
		},
		legacy_init_op=legacy_init_op)

		builder.save()

		pickle.dump(pickle_dict, open("NN_build/data_meta.p", "wb"))
	
	def predict(self,val_dict):
		if isinstance(val_dict,list):
			val_list = val_dict
		else:
			val_list = [val_dict[x] for x in self.feature_names]

		with tf.Session(graph=tf.Graph()) as sess:
			# restore save model
			export_dir = './NN_build'
			model = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
			# print(model)
			loaded_graph = tf.get_default_graph()

			# get necessary tensors by name
			input_tensor_name = model.signature_def['predict_feats'].inputs['input_feats'].name
			input_tensor = loaded_graph.get_tensor_by_name(input_tensor_name)
			output_tensor_name = model.signature_def['predict_feats'].outputs['output_vals'].name
			output_tensor = loaded_graph.get_tensor_by_name(output_tensor_name)

			scores = sess.run(output_tensor, {input_tensor: self.normalize_array([val_list],True)})
			reg_scores = self.denormalize_array(scores,False)
			reg_dict = {x:reg_scores[0][i] for i,x in enumerate(self.label_names)}
			return reg_dict



#fwd_model = ForwardModel()
#fwd_model.init_train()
#ys_in,xs_in = sess.run(test_next_element)
