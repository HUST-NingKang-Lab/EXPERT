#! /data2/public/chonghui_backup/anaconda3/bin/python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Concatenate, Conv2D, MaxPooling2D, Lambda, Add
import numpy as np
import os
from pprint import pprint
from tensorflow.keras import backend as K


# npz_file = '/data2/public/chonghui/MGNify_12.5w/subset_Soil_matrices_1462_features_coef_0.001.npz'
npz_file = '/data2/public/chonghui/MGNify_12.5w/matrices_1462_features_coef_0.001.npz'
gpu = False
split_idx = 4000
end_idx = 5000


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

matrices = matrices.reshape(matrices.shape[0], 1462, 7, 1)

X = matrices
Y = [l_root] + labels
print(X.shape)

layer_weights = [Y[0].shape[1], Y[1].shape[1], Y[2].shape[1], Y[3].shape[1], Y[4].shape[1], Y[5].shape[1]]


class Model(tf.keras.Model):

	def __init__(self, input_shape):

		# init Keras Model
		super(Model, self).__init__()
		self.base_c1_conv1 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), 
								 padding='valid', input_shape=input_shape, name='base_c1_conv1',
								 activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)) # 1462 * 5 * 64
		self.c1_bn1 = BatchNormalization()
		self.base_c1_conv2 = Conv2D(filters=64, kernel_size=(4, 1), padding='valid', strides=(3, 1), 
								 activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)) # 486 * 5 * 64
		self.c1_bn2 = BatchNormalization()
		self.base_c1_maxpool3 = MaxPooling2D(pool_size=(4, 1), strides=(2, 1), padding='valid') # 486 * 5 * 64
		self.base_c1_conv4 = Conv2D(filters=128, kernel_size=(4, 2), strides=(1, 1), padding='valid', 
								 activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.012)) # 483 * 4 * 128
		self.c1_bn4 = BatchNormalization()
		self.base_c1_conv5 = Conv2D(filters=128, kernel_size=(4, 2), strides=(1, 1), padding='valid', 
								 activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.012)) # 480 * 3 * 128
		self.c1_bn5 = BatchNormalization()
		self.base_c1_conv6 = Conv2D(filters=128, kernel_size=(4, 2), strides=(1, 1), padding='valid', 
								 activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.015)) # 477 * 2 * 128
		self.base_c1_maxpool7 = MaxPooling2D(pool_size=(3, 1), strides=(2, 1), padding='valid') # 238 * 2 * 128
		self.c1_bn7 = BatchNormalization()
		self.base_c1_conv8 = Conv2D(filters=256, kernel_size=(2, 2), strides=(2, 1), padding='valid', 
								 activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.015)) # 118 * 1 * 256
		self.c1_bn8 = BatchNormalization()
		self.base_c1_conv9 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
								 activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.015)) # 118 * 1 * 256
		self.c1_bn9 = BatchNormalization()
		self.base_c1_maxpool10 = MaxPooling2D(pool_size=(4, 1), strides=(3, 1), padding='valid')
		self.base_c1_flatten11 = Flatten()
		self.base_c1_dense12 = Dense(512, activation='relu', name='base_c1_dense12',
								 kernel_regularizer=tf.keras.regularizers.l2(0.015))
		self.base_c1_dense12_ext = Dense(256, activation='relu', name='base_c1_dense12_ext',
								 kernel_regularizer=tf.keras.regularizers.l2(0.015)) 

		self.concat = Concatenate(axis=1)
		#self.concat = Add()
		self.c1_bnc = BatchNormalization()
		
		self.l0_dense1 = Dense(256, activation='relu', name='l0d1', 
								 kernel_regularizer=tf.keras.regularizers.l2(0.05))
		self.l0_dense2 = Dense(128, activation='relu', name='l0d2')
		self.l0_linear = Dense(1, activation='sigmoid', name='l0out')

		self.l1_dense1 = Dense(256, activation='relu', name='l1d1',
								 kernel_regularizer=tf.keras.regularizers.l2(0.04))
		self.bn1 = BatchNormalization()
		self.l1_dense2 = Dense(128, activation='relu', name='l1d2', 
								 kernel_regularizer=tf.keras.regularizers.l2(0.02))
		self.bn2 = BatchNormalization()
		self.l1_dense3 = Dense(256, activation='relu', name='l1d3', 
								 kernel_regularizer=tf.keras.regularizers.l2(0.02))
		self.bn3 = BatchNormalization()
		self.l1_linear = Dense(4, activation='sigmoid', name='l1out')

		self.l2_dense1 = Dense(256, activation='relu', name='l2d1',
								 kernel_regularizer=tf.keras.regularizers.l2(0.03))
		self.l2_dense2 = Dense(128, activation='relu', name='l2d2',
								 kernel_regularizer=tf.keras.regularizers.l2(0.015))
		self.l2_dense3 = Dense(256, activation='relu', name='l2d3', 
								 kernel_regularizer=tf.keras.regularizers.l2(0.015))
		self.l2_linear = Dense(7, activation='sigmoid', name='l2out')

		self.l3_dense1 = Dense(256, activation='relu', name='l3d1',
								 kernel_regularizer=tf.keras.regularizers.l2(0.03))
		self.l3_dense2 = Dense(128, activation='relu', name='l3d2',
		                         kernel_regularizer=tf.keras.regularizers.l2(0.015))
		self.l3_dense3 = Dense(256, activation='relu', name='l3d3',
		                         kernel_regularizer=tf.keras.regularizers.l2(0.015))
		self.l3_linear = Dense(22, activation='sigmoid', name='l3out')

		self.l4_dense1 = Dense(256, activation='relu', name='l4d1',
		                         kernel_regularizer=tf.keras.regularizers.l2(0.03))
		self.l4_dense2 = Dense(128, activation='relu', name='l4d2',
		                         kernel_regularizer=tf.keras.regularizers.l2(0.015))
		self.l4_dense3 = Dense(256, activation='relu', name='l4d3',
		                         kernel_regularizer=tf.keras.regularizers.l2(0.015))
		self.l4_linear = Dense(56, activation='sigmoid', name='l4out')

		self.l5_dense1 = Dense(256, activation='relu', name='l4d1',
		                         kernel_regularizer=tf.keras.regularizers.l2(0.03))
		self.l5_dense2 = Dense(128, activation='relu', name='l4d2',
		                         kernel_regularizer=tf.keras.regularizers.l2(0.015))
		self.l5_dense3 = Dense(256, activation='relu', name='l4d3',
		                         kernel_regularizer=tf.keras.regularizers.l2(0.015))
		self.l5_linear = Dense(43, activation='sigmoid', name='l5out')
		
		def scale_output(x):
			total_contrib = tf.constant([[1]], dtype=tf.float32, shape=(1, 1))
			unknown_contrib = tf.subtract(total_contrib, tf.keras.backend.sum(x, keepdims=True, axis=1))
			contrib = tf.keras.backend.relu(tf.keras.backend.concatenate( (x, unknown_contrib), axis=1))
			scaled_contrib = tf.divide(contrib, tf.keras.backend.sum(contrib, keepdims=True, axis=1))
			return scaled_contrib
		self.cal_contribution = Lambda(scale_output)

	def call(self, inputs, training=False):
		x = inputs

		# Inception-like structure
		#x = self.base_c1_conv1(x)
		#x = self.c1_bn1(x, training=training)
		#x = self.base_c1_conv2(x)
		#x = self.c1_bn2(x, training=training)
		#x = self.base_c1_maxpool3(x)
		#x = self.base_c1_conv4(x)
		#x = self.base_c1_maxpool5(x)
		#x = self.c1_bn4(x, training=training)
		#x = self.base_c1_conv5(x)
		#x = self.c1_bn5(x, training=training)
		#x = self.base_c1_conv6(x)
		#x = self.base_c1_maxpool7(x)
		#x = self.c1_bn7(x, training=training)
		#x = self.base_c1_conv8(x)
		#x = self.c1_bn8(x, training=training)
		#x = self.base_c1_conv9(x)
		#x = self.c1_bn9(x, training=training)
		#x = self.base_c1_maxpool10(x)
		x = self.base_c1_dense12(x)
		x = self.c1_bn9(x, training=training)
		x = self.base_c1_dense12_ext(x)
		x = self.base_c1_flatten11(x)
		#base = self.concat([x, xc2])
		base = x
		base = self.c1_bnc(base, training=training)

		l0 = self.l0_dense1(base)
		l0 = self.bn1(l0, training=training)
		l0 = self.l0_dense2(l0)

		l1 = self.l1_dense1(base)
		l1 = self.bn1(l1, training=training)
		l1 = self.l1_dense2(l1)

		l2 = self.l2_dense1(base)
		l2 = self.bn1(l2, training=training)
		l2 = self.l2_dense2(l2)

		l3 = self.l3_dense1(base)
		l3 = self.bn1(l3, training=training)
		l3 = self.l3_dense2(l3)

		l4 = self.l4_dense1(base)
		l4 = self.bn1(l4, training=training)
		l4 = self.l4_dense2(l4)

		l5 = self.l5_dense1(base)
		l5 = self.bn1(l5, training=training)
		l5 = self.l5_dense2(l1)

		l1_c = self.concat([l0, l1])
		l2_c = self.concat([l1, l2])
		l3_c = self.concat([l2, l3])
		l4_c = self.concat([l3, l4])
		l5_c = self.concat([l4, l5])

		l1_c = self.bn2(l1_c, training=training)
		l2_c = self.bn2(l2_c, training=training)
		l3_c = self.bn2(l3_c, training=training)
		l4_c = self.bn2(l4_c, training=training)
		l5_c = self.bn2(l5_c, training=training)

		l0 = self.l0_linear(l0)
		l1 = self.l1_dense3(l1_c)
		l1 = self.bn3(l1, training=training)
		l1 = self.l1_linear(l1)
		l2 = self.l2_dense3(l2_c)
		l2 = self.bn3(l2, training=training)
		l2 = self.l2_linear(l2)
		l3 = self.l3_dense3(l3_c)
		l3 = self.bn3(l3, training=training)
		l3 = self.l3_linear(l3)
		l4 = self.l4_dense3(l4_c)
		l4 = self.bn3(l4, training=training)
		l4 = self.l4_linear(l4)
		l5 = self.l5_dense3(l5_c)
		l5 = self.bn3(l5, training=training)
		l5 = self.l5_linear(l5)

		l1 = self.cal_contribution(l1)
		l2 = self.cal_contribution(l2)
		l3 = self.cal_contribution(l3)
		l4 = self.cal_contribution(l4)
		l5 = self.cal_contribution(l5)
		
		#y = self.concat([l0, l1, l2, l3, l4, l5])
		return l0, l1, l2, l3, l4, l5


def r_s(y_true, y_pred):
	SS_res =  K.sum(K.square(y_true - y_pred)) 
	SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
	return ( 1 - SS_res/(SS_tot + K.epsilon()) )


model = Model(input_shape=X.shape[1:])
model.compile(optimizer=tf.keras.optimizers.Nadam(lr=1e-3),
			  loss={layer: 'categorical_crossentropy' for layer in ['output_' + str(i) for i in range(1, 7)]},
			  loss_weights={'output_' + str(i): 1/i for i in range(1, 7)},
			  metrics=['acc'])

model.fit(X[0:split_idx], [y[0:split_idx] for y in Y], epochs=10, batch_size=128, validation_split=0.1, verbose=1) 
# use sample_weight !!!!!!!!!!!!
model.evaluate(X[split_idx:], [y[split_idx:] for y in Y], verbose=1)
res = model.predict(X[split_idx:])
pprint('Prediction:------------------')
pprint(res)
pprint('Truth:-----------------------')
t = [y[split_idx:] for y in Y]
pprint(t)
model.summary()
model.save('/data2/public/chonghui_backup/onn_model_7.30', save_format='tf')
np.savez('/data2/public/chonghui_backup/prediction', list(res))
np.savez('/data2/public/chonghui_backup/truth', t)


