#! /data2/public/chonghui_backup/anaconda3/CLI/python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Concatenate, Conv1D, Lambda, Add, Activation
from tensorflow.keras.metrics import *
#import tensorflow_addons as tfa
import numpy as np
import os
from pprint import pprint
from tensorflow.keras import backend as K
import pandas as pd

# npz_file = '/data2/public/chonghui/MGNify_12.5w/subset_Soil_matrices_1462_features_coef_0.001.npz'
npz_file = '/data2/public/chonghui/MGNify_12.5w/matrices_1462_features_coef_0.001.npz'
gpu = True
split_idx = 102400
end_idx = 125823
np.random.seed(0)
lr = None
m = None


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)

if gpu:
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
	np.random.shuffle(idx)
	nidx = idx[0:end_idx]
	labels = [l0[nidx], l1[nidx], l2[nidx], l3[nidx], l4[nidx]]
	return matrices[nidx], labels

matrices, labels = get_data_label()

l_root = np.array(list(zip([1] * labels[0].shape[0], [0] * labels[0].shape[0])))

#matrices = matrices.reshape(matrices.shape[0], 1462, 7, 1)

X = matrices
Y = [l_root] + labels
print(X.shape)
for i in Y:
	print(i.shape)
layer_weights = [Y[0].shape[1], Y[1].shape[1], Y[2].shape[1], Y[3].shape[1], Y[4].shape[1], Y[5].shape[1]]

ln_init = tf.keras.initializers.HeNormal(seed=0)

class Model(tf.keras.Model):

	def __init__(self):
		super(Model, self).__init__()
		self.base_block1 = self.init_base_block1()
		self.base_block2 = self.init_base_block2()
		self.base_block3 = self.init_base_block3()
		self.base_block4 = self.init_base_block4()
		self.base_batchnorms = [self.init_bn_relu('base'+str(i)) for i in range(1, 5)]
		self.concat = tf.concat
		self.copy = tf.identity
		#self.add = Add()
		self.num_units = [1, 4, 7, 22, 56, 43]
		self.inter_blocks = [self.init_kth_inter_block(k, self.num_units[k]) for k in range(6)]
		self.out_batchnorms = [self.init_bn_relu(k) for k in range(6)]
		#self.dropouts = [Dropout(0.5) for k in range(6)]
		self.out_blocks = [self.init_kth_out_block(k, self.num_units[k]) for k in range(6)]
		self.postproc_layers = [self.init_kth_post_proc_layer(k) for k in range(6)]
	
	def init_base_block1(self):
		block = tf.keras.Sequential(name='base_block1')
		block.add(Conv1D(64, kernel_size=2, strides=1, padding='same', 
						  kernel_initializer=ln_init, use_bias=True, input_shape=(1462, 7)))
		block.add(Conv1D(64, kernel_size=2, strides=2, kernel_initializer=ln_init, use_bias=True))
		return block

	def init_base_block2(self):
		block = tf.keras.Sequential(name='base_block2')
		block.add(Conv1D(128, kernel_size=3, strides=2, kernel_initializer=ln_init, use_bias=True))
		block.add(Conv1D(128, kernel_size=3, strides=2, kernel_initializer=ln_init, use_bias=True))	
		return block

	def init_base_block3(self):
		block = tf.keras.Sequential(name='base_block3')
		block.add(Conv1D(256, kernel_size=2, strides=2, kernel_initializer=ln_init, use_bias=True))
		block.add(Conv1D(256, kernel_size=4, strides=3, kernel_initializer=ln_init, use_bias=True))
		block.add(Conv1D(256, kernel_size=2, strides=2, kernel_initializer=ln_init, use_bias=True))
		return block
		
	def init_base_block4(self):
		block = tf.keras.Sequential(name='base_block4')
		block.add(Conv1D(512, kernel_size=3, strides=2, kernel_initializer=ln_init, use_bias=True))
		block.add(Conv1D(512, kernel_size=7, strides=1, kernel_initializer=ln_init, use_bias=True))
		block.add(Conv1D(1024, kernel_size=1, strides=1, kernel_initializer=ln_init, use_bias=True))
		#block.add(BatchNormalization())
		#block.add(tfa.layers.GroupNormalization(32))
		#block.add(Activation('relu'))
		#block.add(Dropout(0.2))
		block.add(Flatten())
		return block

	def init_bn_relu(self, k):
		if type(k) == str:
		#if k.startswith('base'):
			block = tf.keras.Sequential(name='base_bn_relu_block_'+k[-1])
		else:
			block = tf.keras.Sequential(name=str(k)+'th_bn_relu_block')
		block.add(BatchNormalization(momentum=0.9))
		block.add(Activation('relu'))
		#block.add(Dropout(0.5))
		return block

	def init_kth_inter_block(self, k: int, out_units):
		block = tf.keras.Sequential(name=str(k)+'th_inter_block')
		block.add(Dense(256, name='l'+str(k)+'_inter_dense_0', input_shape=(1024,), use_bias=True, kernel_initializer=ln_init))
		block.add(Activation('relu'))
		#block.add(Dropout(0.5))
		block.add(Dense(128, name='l'+str(k)+'_inter_dense_1', use_bias=True, kernel_initializer=ln_init))
		#block.add(BatchNormalization())
		block.add(Activation('relu'))
		block.add(Dense(64, name='l'+str(k)+'_inter_dense_2', use_bias=True, kernel_initializer=ln_init))
		#block.add(BatchNormalization())
		#block.add(Activation('relu'))
		#block.add(Dense(num_units[1], name='l'+str(k)+'_inter_dense_3', use_bias=True, kernel_initializer=he_normal))
		return block

	def init_kth_out_block(self, k: int, out_units):
		filter_by_size = lambda x: x[ (x > out_units * 4) & (x <= out_units * 16)][::-1]
		num_units = filter_by_size(2 ** np.arange(11))
		# modify here
		block = tf.keras.Sequential(name=str(k)+'th_out_block')
		#block.add(Dropout(0.2))
		block.add(Dense(num_units[0], name='l'+str(k)+'_out_dense_0', use_bias=True, kernel_initializer=ln_init))
		block.add(BatchNormalization())
		block.add(Activation('relu'))
		#block.add(Dropout(0.3))
		#block.add(Dense(num_units[0], name='l'+str(k)+'_out_dense_1', use_bias=True, kernel_initializer=he_normal))
		#block.add(BatchNormalization())
		#block.add(Activation('relu'))
		#block.add(Dropout(0.3))
		block.add(Dense(num_units[1], name='l'+str(k)+'_out_dense_2', use_bias=True, kernel_initializer=ln_init))
		block.add(BatchNormalization())
		block.add(Activation('relu'))
		block.add(Dropout(0.5))
		block.add(Dense(out_units, name='l'+str(k), activation='sigmoid', use_bias=True))
		return block

	def init_kth_post_proc_layer(self, k):
		def scale_output(x):
			total_contrib = tf.constant([[1]], dtype=tf.float32, shape=(1, 1))
			unknown_contrib = tf.subtract(total_contrib, tf.keras.backend.sum(x, keepdims=True, axis=1))
			contrib = tf.keras.backend.relu(tf.keras.backend.concatenate( (x, unknown_contrib), axis=1))
			scaled_contrib = tf.divide(contrib, tf.keras.backend.sum(contrib, keepdims=True, axis=1))
			return scaled_contrib
		return Lambda(scale_output, name='l'+str(k)+'_y')

	def call(self, input, training=False):
		base = self.base_block1(input, training=training)
		base = self.base_batchnorms[0](base, training=training)
		base = self.base_block2(base, training=training)
		base = self.base_batchnorms[1](base, training=training)
		base = self.base_block3(base, training=training)
		base = self.base_batchnorms[2](base, training=training)
		base = self.base_block4(base, training=training)
		base = self.base_batchnorms[3](base, training=training)

		#base_copy = self.copy(base)
		inter_factors = [self.inter_blocks[i](base, training=training) for i in range(6)]
		concat_factors = [self.concat(inter_factors[i-1:i+1], axis=1) if i > 0 else inter_factors[i] for i in range(6)]
		#concat_factors = [self.add(inter_factors[0:i]) for i in range(1, 7)]
		concat_factors = [self.out_batchnorms[i](concat_factors[i], training=training) for i in range(6)]
		#concat_factors = [self.dropouts[i](concat_factors[i], training=training) for i in range(6)]
		y_s = [self.out_blocks[i](concat_factors[i], training=training) for i in range(6)]
		(l0_y, l1_y, l2_y, l3_y, l4_y, l5_y) = (self.postproc_layers[i](y_s[i], training=training) for i in range(6))
		
		return l0_y, l1_y, l2_y, l3_y, l4_y, l5_y


print('lr: {}, momentum: {}'.format(lr, m))
strategy = tf.distribute.MirroredStrategy()  
with strategy.scope(): 
#if True:
	model = Model()
	model.compile(optimizer=tf.keras.optimizers.SGD(lr=lr, momentum=m, nesterov=True), # how about mommentum? beta1 and beta 2?
				  loss=tf.keras.losses.CategoricalCrossentropy(),
				  loss_weights={'output_' + str(i+1): layer_weights[i] for i in range(6)},
				  #loss_weights={'output_' + str(i+1): layer_weights[i] / sum(layer_weights) for i in range(6)},
				  metrics=[#TruePositives(0.5, name='TP'), FalsePositives(0.5, name='FP'), TrueNegatives(0.5, name='TN'), 
						   #FalseNegatives(0.5, name='FN'), 
						   'acc'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, verbose=2, restore_best_weights=True)
history = '/data2/public/chonghui_backup/training_history_by_csv_logger.csv'
csv_logger = tf.keras.callbacks.CSVLogger(history, separator=',', append=False)

history = model.fit(x=X[0:512], y=[y[0:512] for y in Y], epochs=1, batch_size=512,# shuffle=False, 
					validation_data=(X[split_idx:], [y[split_idx:] for y in Y]), verbose=1) 
	# use sample_weight !!!!!!!!!!!!
model.summary()
	
model.fit(x=X[0:split_idx], y=[y[0:split_idx] for y in Y], epochs=500, batch_size=512, #shuffle=False, 
						callbacks=[csv_logger, early_stopping], validation_data=(X[split_idx:], [y[split_idx:] for y in Y]))

#res = model.predict(X[split_idx:])
model.save('/data2/public/chonghui_backup/onn_model_8.6_tunning', save_format='tf')
#pd.DataFrame(history.history).to_csv('/data2/public/chonghui_backup/training_history_tunning.csv')
#np.savez('/data2/public/chonghui_backup/prediction', list(res))
#np.savez('/data2/public/chonghui_backup/truth', t)


