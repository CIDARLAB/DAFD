import tensorflow as tf
import numpy as np
import pandas as pd

label_ct = 2
feature_ct = 8
test_number = 100

batch_num_train = 1024
neuron_ct = 100
learning_rate = 0.01




dat_df = pd.read_csv("ExperimentalResults.csv")
dat_df = dat_df.sample(frac=1).reset_index(drop=True)
labels = dat_df.iloc[test_number:,-label_ct:]
features = dat_df.iloc[test_number:,:-label_ct]


maxes_dict = {}
mins_dict = {}

maxes = np.max(labels,axis=0)
mins = np.min(labels,axis=0)
for i,col in enumerate(labels.columns):
	maxes_dict[col] = maxes[i]
	mins_dict[col] = mins[i]

maxes = np.max(features,axis=0)
mins = np.min(features,axis=0)
for i,col in enumerate(features.columns):
	maxes_dict[col] = maxes[i]
	mins_dict[col] = mins[i]



label_names = list(labels.columns)
feature_names = list(features.columns)


def normalize_array(np_arr, is_features):
	df_nonnormal = pd.DataFrame(np_arr)
	df_nonnormal.columns = feature_names if is_features else label_names
	return np.array(normalize_dataframe(df_nonnormal))

def denormalize_array(np_arr, is_features):
	df_nonnormal = pd.DataFrame(np_arr)
	df_nonnormal.columns = feature_names if is_features else label_names
	return np.array(denormalize_dataframe(df_nonnormal))

def normalize_dataframe(df):
	global normalize_dict
	new_df = pd.DataFrame()
	for col in df.columns:
		new_df[col] = df[col].apply(min_max_normalize, set_min=mins_dict[col], set_max=maxes_dict[col])
	return new_df

def denormalize_dataframe(df):
	global denormalize_dict
	new_df = pd.DataFrame()
	for col in df.columns:
		new_df[col] = df[col].apply(min_max_denormalize, set_min=mins_dict[col], set_max=maxes_dict[col])
	return new_df

def min_max_normalize(x, set_min, set_max):
	return (x-set_min)/(set_max-set_min)

def min_max_denormalize(x, set_min, set_max):
	return (x*(set_max-set_min)) + set_min



labels = normalize_dataframe(labels)
features = normalize_dataframe(features)

train_ds = tf.data.Dataset.from_tensor_slices((labels,features))
train_iterator = train_ds.shuffle(1000).repeat().batch(batch_num_train).make_one_shot_iterator()
train_next_element = train_iterator.get_next()


labels = dat_df.iloc[:test_number,-label_ct:]
features = dat_df.iloc[:test_number,:-label_ct]

test_ds = tf.data.Dataset.from_tensor_slices((labels,features))
test_iterator = test_ds.shuffle(1000).repeat().batch(test_number).make_one_shot_iterator()
test_next_element = test_iterator.get_next()



# INPUT FEATURES VECTOR
x = tf.placeholder(tf.float32, [None, feature_ct])

# INPUT LABELS VECTOR
y_ = tf.placeholder(tf.float32, [None, label_ct])



h1 = tf.layers.dense(x, neuron_ct, activation=tf.nn.relu)

h2 = tf.layers.dense(h1, neuron_ct, activation=tf.nn.relu)

h3 = tf.layers.dense(h2, neuron_ct, activation=tf.nn.relu)

y = tf.layers.dense(h3, label_ct, activation=None)


# LOSS CALCULATIONS
MSE_loss = tf.losses.mean_squared_error(predictions=y,labels=y_)
#loss_val = MSE_loss
loss_val = tf.norm(y_-y)

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(MSE_loss)

# SET UP SESSION
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


for i in range(100000):
	ys_in,xs_in = sess.run(train_next_element)
	_, loss_out = sess.run([train_step,loss_val], feed_dict={x: xs_in, y_: ys_in})
	if i % 100 == 0:
		y_out = sess.run(y,feed_dict={x: xs_in, y_: ys_in})
		MSE_out = sess.run(MSE_loss,feed_dict={x: xs_in, y_: ys_in})
		print(str(i) + " " + str(loss_out) + " " + str(np.linalg.norm(ys_in-y_out)))


#ys_in,xs_in = sess.run(test_next_element)
