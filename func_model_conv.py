#! /data2/public/chonghui_backup/anaconda3/bin/python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Concatenate, Conv1D, Lambda, Add
from tensorflow.keras.metrics import *
import numpy as np
import os
from pprint import pprint
from tensorflow.keras import backend as K
import pandas as pd

# npz_file = '/data2/public/chonghui/MGNify_12.5w/subset_Soil_matrices_1462_features_coef_0.001.npz'
npz_file = '/data2/public/chonghui/MGNify_12.5w/matrices_1462_features_coef_0.001.npz'
gpu = True
split_idx = 45000
end_idx = 50000


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)

if gpu:
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)


def get_data_label():
	data = np.load(npz_file)
	matrices = data['matrices']
	l0, l1, l2, l3, l4 = (data['label_0'], data['label_1'], data['label_2'],
							  data['label_3'], data['label_4'])
	
	l0 = np.concatenate( (l0, (1-l0.sum(axis=1)).reshape(l0.shape[0], 1)), axis=1) # Unknown
	l1 = np.concatenate( (l1, (1-l1.sum(axis=1)).reshape(l1.shape[0], 1)), axis=1) # Unknown
	l2 = np.concatenate( (l2, (1-l2.sum(axis=1)).reshape(l2.shape[0], 1)), axis=1) # Unknown
	l3 = np.concatenate( (l3, (1-l3.sum(axis=1)).reshape(l3.shape[0], 1)), axis=1) # Unknown
	l4 = np.concatenate( (l4, (1-l4.sum(axis=1)).reshape(l4.shape[0], 1)), axis=1) # Unknown
		
	idx = np.arange(matrices.shape[0])
	np.random.seed(1)
	np.random.shuffle(idx)
	nidx = idx[0:end_idx]
	labels = [l0[nidx], l1[nidx], l2[nidx], l3[nidx], l4[nidx]]
	return matrices[nidx], labels

matrices, labels = get_data_label()

l_root = np.ones((labels[0].shape[0], 1))

#matrices = matrices.reshape(matrices.shape[0], 1462, 7, 1)

X = matrices
Y = [l_root] + labels
print(X.shape)
for i in Y:
	print(i.shape)
layer_weights = [Y[0].shape[1], Y[1].shape[1], Y[2].shape[1], Y[3].shape[1], Y[4].shape[1], Y[5].shape[1]]

def r_s(y_true, y_pred):
	SS_res =  K.sum(K.square(y_true - y_pred)) 
	SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
	return ( 1 - SS_res/(SS_tot + K.epsilon()) )


def layer_block_inter(tensor, out_units, layer):
	# already flattened !!!!!!!
	
	filter_by_size = lambda x: x[ (x >= out_units * 4) & (x <= out_units * 32)][::-1]
	num_units = filter_by_size(2 ** np.arange(11))
	tmp = Dense(num_units[0], name=layer+'_dense_inter_0', activation=tf.nn.relu, use_bias=False, 
				kernel_regularizer=tf.keras.regularizers.l2(0.012))(tensor)
	tmp = BatchNormalization()(tmp)
	for i, num_unit in enumerate(num_units[1:]):
		tmp = Dense(num_unit, name=layer+'_dense_inter_'+str(i+1), activation=tf.nn.relu, use_bias=False)(tmp)
		tmp = BatchNormalization()(tmp)
	return tmp


def layer_block_out(tensor, out_units, layer):
	n_units = 2 ** np.arange(0, 11)[::-1]
	n_units = n_units[(n_units > (4 * out_units)).sum() + 1]
	tmp = Dense(n_units, name=layer+'_dense_out_0', activation=tf.nn.relu, use_bias=False)(tensor)
	tmp = Dense(n_units, name=layer+'_dense_out_1', activation=tf.nn.relu, use_bias=False)(tmp)
	tmp = Dense(out_units, name=layer+'_y', activation='sigmoid', use_bias=False)(tmp)
	return tmp


strategy = tf.distribute.MirroredStrategy()  
with strategy.scope(): 
#if True:
	x = tf.keras.Input(shape=(1462, 7))

	base = Conv1D(64, kernel_size=1, use_bias=False, activation=tf.nn.relu)(x)
	base = BatchNormalization()(base)
	base = Conv1D(128, kernel_size=1, use_bias=False, activation=tf.nn.relu)(base)
	base = BatchNormalization()(base)
	#base0_3 = Dense(base0_2, 1024, activation=tf.nn.relu)
	base = Conv1D(128, kernel_size=1, use_bias=False, activation=tf.nn.relu)(base)
	base = BatchNormalization()(base)
	base = Conv1D(256, kernel_size=1, use_bias=False, activation=tf.nn.relu)(base)
	base = BatchNormalization()(base)
	base = Conv1D(256, kernel_size=1, use_bias=False, activation=tf.nn.relu)(base)
	base = BatchNormalization()(base)
	base = Conv1D(512, kernel_size=1, use_bias=False, activation=tf.nn.relu)(base)
	base = BatchNormalization()(base)
	base = Conv1D(512, kernel_size=1, use_bias=False, activation=tf.nn.relu)(base)
	base = BatchNormalization()(base)
	base = Conv1D(256, kernel_size=1, use_bias=False, activation=tf.nn.relu)(base)
	base = BatchNormalization()(base)
	base = Conv1D(128, kernel_size=1, use_bias=False, activation=tf.nn.relu)(base)
	base = BatchNormalization()(base)
	base = Conv1D(64, kernel_size=1, use_bias=False, activation=tf.nn.relu)(base)
	base = BatchNormalization()(base)
	base = Conv1D(32, kernel_size=1, use_bias=False, activation=tf.nn.relu)(base)
	base = BatchNormalization()(base)
	base = Conv1D(16, kernel_size=1, use_bias=False, activation=tf.nn.relu)(base)
	base = BatchNormalization()(base)
	base = Conv1D(8, kernel_size=1, use_bias=False, activation=tf.nn.relu)(base)
	base = BatchNormalization()(base)
	base = Conv1D(4, kernel_size=1, use_bias=False, activation=tf.nn.relu)(base)
	base = BatchNormalization()(base)
	base = Conv1D(1024, kernel_size=1462, use_bias=False, activation=tf.nn.relu)(base)
	base = BatchNormalization()(base)
	base = Conv1D(1024, kernel_size=1, use_bias=False, activation=tf.nn.relu)(base)
	base = BatchNormalization()(base)
	base = Conv1D(1024, kernel_size=1, use_bias=False, activation=tf.nn.relu)(base)
	base = BatchNormalization()(base)

	root = Flatten()(base)
	root = Dense(32, activation=tf.nn.relu, use_bias=False)(root)
	root = BatchNormalization()(root)
	root_y = Dense(1, activation='sigmoid', name='l0')(root)
	
	l1 = Flatten()(base)
	l1 = tf.concat([root, l1], axis=1)
	l1 = layer_block_inter(tensor=l1, layer='l1', out_units=4)
	l1_copy = tf.identity(l1)
	l1_y = layer_block_out(tensor=l1_copy, out_units=4, layer='l1')

	l2 = Flatten()(base)
	l2 = tf.concat([l2, l1], axis=1)
	l2 = layer_block_inter(tensor=l2, layer='l2', out_units=7)
	l2_copy = tf.identity(l2)
	l2_y = layer_block_out(tensor=l2_copy, out_units=7, layer='l2')

	l3 = Flatten()(base)
	l3 = tf.concat([l3, l2], axis=1)
	l3 = layer_block_inter(tensor=l3, layer='l3', out_units=22)
	l3_copy = tf.identity(l3)
	l3_y = layer_block_out(tensor=l3_copy, out_units=22, layer='l3')

	l4 = Flatten()(base)
	l4 = tf.concat([l4, l3], axis=1)
	l4 = layer_block_inter(tensor=l4, layer='l4', out_units=56)
	l4_copy = tf.identity(l4)
	l4_y = layer_block_out(tensor=l4_copy, out_units=56, layer='l4')

	l5 = Flatten()(base)
	l5 = tf.concat([l5, l4], axis=1)
	l5 = layer_block_inter(tensor=l5, layer='l5', out_units=43)
	l5_copy = tf.identity(l5)
	l5_y = layer_block_out(tensor=l5_copy, out_units=43, layer='l5')

	def scale_output(x):
		total_contrib = tf.constant([[1]], dtype=tf.float32, shape=(1, 1))
		unknown_contrib = tf.subtract(total_contrib, tf.keras.backend.sum(x, keepdims=True, axis=1))
		contrib = tf.keras.backend.relu(tf.keras.backend.concatenate( (x, unknown_contrib), axis=1))
		scaled_contrib = tf.divide(contrib, tf.keras.backend.sum(contrib, keepdims=True, axis=1))
		return scaled_contrib


	l1_y = Lambda(scale_output, name='l1')(l1_y)
	l2_y = Lambda(scale_output, name='l2')(l2_y)
	l3_y = Lambda(scale_output, name='l3')(l3_y)
	l4_y = Lambda(scale_output, name='l4')(l4_y)
	l5_y = Lambda(scale_output, name='l5')(l5_y)


	model = tf.keras.Model(inputs=x, outputs=[root_y, l1_y, l2_y, l3_y, l4_y, l5_y])
	model.compile(optimizer=tf.keras.optimizers.Adam(lr=5e-5), #, clipnorm=5), # how about mommentum? beta1 and beta 2?
			  	  loss={layer: tf.keras.losses.CategoricalCrossentropy(from_logits=False) 
				  for layer in ['l' + str(i) for i in range(6)]},
			      #loss_weights={'l' + str(i): np.log(i + 2) / np.log(np.arange(2, 8)).sum() for i in range(6)},
			  	  loss_weights={'l' + str(i): i / 21 for i in range(6)},
			  	  metrics=[TruePositives(0.5, name='TP'), FalsePositives(0.5, name='FP'), TrueNegatives(0.5, name='TN'), 
				  FalseNegatives(0.5, name='FN'), CategoricalAccuracy(name='acc'), AUC(curve='ROC', name='auROC'), 
				  AUC(curve='PR', name='auPRC')])

history = model.fit(x=X[0:split_idx], y=[y[0:split_idx] for y in Y], epochs=180, batch_size=128, validation_split=0.1, verbose=1) 
# use sample_weight !!!!!!!!!!!!
model.evaluate(x=X[split_idx:], y=[y[split_idx:] for y in Y], verbose=1)
#res = model.predict(X[split_idx:])
#pprint('Prediction:------------------')
#pprint(res)
#pprint('Truth:-----------------------')
#t = [y[split_idx:] for y in Y]
#pprint(t)
model.summary()
model.save('/data2/public/chonghui_backup/onn_model_7.30', save_format='tf')
pd.DataFrame(history.history).to_csv('/data2/public/chonghui_backup/training_history.csv')
#np.savez('/data2/public/chonghui_backup/prediction', list(res))
#np.savez('/data2/public/chonghui_backup/truth', t)


