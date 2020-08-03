#! /data2/public/chonghui_backup/anaconda3/bin/python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Concatenate, Conv1D, Lambda, Add, Activation
from tensorflow.keras.metrics import *
import numpy as np
import os
from pprint import pprint
from tensorflow.keras import backend as K
import pandas as pd

# npz_file = '/data2/public/chonghui/MGNify_12.5w/subset_Soil_matrices_1462_features_coef_0.001.npz'
npz_file = '/data2/public/chonghui/MGNify_12.5w/matrices_1462_features_coef_0.001.npz'
gpu = True
split_idx = 100000
end_idx = 125823


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

ln_init = tf.keras.initializers.GlorotUniform()

class Model(tf.keras.Model):

    def __init__(self):
        super(Model, self).__init__()
        self.base_blocks = self.init_base_block()
        self.concat = tf.concat
        self.copy = tf.identity
        self.num_units = [1, 4, 7, 22, 56, 43]
        self.inter_blocks = [self.init_kth_inter_block(k, self.num_units[k]) for k in range(6)]
        self.out_blocks = [self.init_kth_out_block(k, self.num_units[k]) for k in range(6)]
        self.postproc_layers = [self.init_kth_post_proc_layer(k) for k in range(6)]

    def init_kth_inter_block(self, k: int, out_units):
        filter_by_size = lambda x: x[ (x > out_units * 8) & (x <= out_units * 32)][::-1]
        num_units = filter_by_size(2 ** np.arange(11))
        # modify here
        block = tf.keras.Sequential()
        block.add(Dense(num_units[0], name='l'+str(k)+'_inter_dense_0', input_shape=(512,), use_bias=False, kernel_initializer=ln_init))
        block.add(BatchNormalization())
        block.add(Activation('relu'))
        block.add(Dense(num_units[1], name='l'+str(k)+'_inter_dense_1', use_bias=False, kernel_initializer=ln_init))
        block.add(BatchNormalization())
        block.add(Activation('relu'))
        return block

    def init_kth_out_block(self, k: int, out_units):
        n_units = 2 ** np.arange(0, 11)[::-1]
        n_units = n_units[(n_units > (4 * out_units)).sum() + 1]
        # modify here
        block = tf.keras.Sequential()
        block.add(Dense(n_units, name='l'+str(k)+'_out_dense_0', use_bias=False, kernel_initializer=ln_init,
                kernel_regularizer='l2'))
        block.add(BatchNormalization())
        block.add(Activation('relu'))
        block.add(Dense(n_units, name='l'+str(k)+'_out_dense_1', use_bias=False, kernel_initializer=ln_init,
                kernel_regularizer='l2'))
        block.add(BatchNormalization())
        block.add(Activation('relu'))
        block.add(Dense(n_units, name='l'+str(k)+'_out_dense_1', use_bias=False, kernel_initializer=ln_init,
                kernel_regularizer='l2'))
        block.add(BatchNormalization())
        block.add(Activation('relu'))
        block.add(Dense(out_units, name='l'+str(k), activation='sigmoid', use_bias=False))
        return block

    def init_base_block(self):
        block = tf.keras.Sequential()
        block.add(Conv1D(64, kernel_size=2, strides=1, padding='same', activation='relu', 
                          kernel_initializer=ln_init, use_bias=False, input_shape=(1462, 7)))
        block.add(Conv1D(64, kernel_size=2, strides=2, activation='relu', kernel_initializer=ln_init, use_bias=False))
        block.add(Conv1D(128, kernel_size=3, strides=2, activation='relu', kernel_initializer=ln_init, use_bias=False))
        block.add(Conv1D(128, kernel_size=3, strides=2, activation='relu', kernel_initializer=ln_init, use_bias=False))
        block.add(Conv1D(256, kernel_size=2, strides=2, activation='relu', kernel_initializer=ln_init, use_bias=False))
        block.add(Conv1D(256, kernel_size=4, strides=3, activation='relu', kernel_initializer=ln_init, use_bias=False))
        block.add(Conv1D(512, kernel_size=2, strides=2, activation='relu', kernel_initializer=ln_init, use_bias=False))
        block.add(Conv1D(512, kernel_size=3, strides=2, activation='relu', kernel_initializer=ln_init, use_bias=False,
                          kernel_regularizer='l2'))
        block.add(Conv1D(512, kernel_size=7, strides=1, use_bias=False, kernel_initializer=ln_init, kernel_regularizer='l2'))
        block.add(BatchNormalization())
        block.add(Activation('relu'))
        block.add(Flatten())
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
        base = self.base_blocks(input, training=training)
        inter_factors = [self.inter_blocks[i](base, training=training) for i in range(6)]
        concat_factors = [self.concat(inter_factors[0:i], axis=1) for i in range(1, 7)]
        y_s = [self.out_blocks[i](concat_factors[i], training=training) for i in range(6)]
        (l0_y, l1_y, l2_y, l3_y, l4_y, l5_y) = (self.postproc_layers[i](y_s[i], training=training) for i in range(6))
        return l0_y, l1_y, l2_y, l3_y, l4_y, l5_y


strategy = tf.distribute.MirroredStrategy()  
with strategy.scope(): 
#if True:
	model = Model()
	model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-5), #, clipnorm=5), # how about mommentum? beta1 and beta 2?
			  	  loss={layer: tf.keras.losses.CategoricalCrossentropy() if layer != 'l0' else tf.keras.losses.BinaryCrossentropy()
				  for layer in ['l' + str(i) for i in range(6)]},
			      #loss_weights={'l' + str(i): np.exp(i + 2) / np.exp(np.arange(2, 8)).sum() for i in range(6)},
			  	  loss_weights={'l' + str(i): i / 21 for i in range(6)},
			  	  metrics=[TruePositives(0.5, name='TP'), FalsePositives(0.5, name='FP'), TrueNegatives(0.5, name='TN'), 
				  FalseNegatives(0.5, name='FN'), 'acc', AUC(curve='ROC', name='auROC'), 
				  AUC(curve='PR', name='auPRC')])

model.summary()
history = model.fit(x=X[0:split_idx], y=[y[0:split_idx] for y in Y], epochs=1000, batch_size=512, validation_split=0.1, verbose=1) 
	# use sample_weight !!!!!!!!!!!!
model.evaluate(x=X[split_idx:], y=[y[split_idx:] for y in Y], verbose=1)
	#res = model.predict(X[split_idx:])
model.save('/data2/public/chonghui_backup/onn_model_8.2_tunning', save_format='tf')
pd.DataFrame(history.history).to_csv('/data2/public/chonghui_backup/training_history_tunning.csv')
#np.savez('/data2/public/chonghui_backup/prediction', list(res))
#np.savez('/data2/public/chonghui_backup/truth', t)


