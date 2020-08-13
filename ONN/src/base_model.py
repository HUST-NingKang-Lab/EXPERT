import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Concatenate, Conv1D, \
	Lambda, Add, Activation
from tensorflow.keras.metrics import *
#import tensorflow_addons as tfa
import numpy as np
import os

X = None
Y = None
he_normal = tf.keras.initializers.HeNormal(seed=0)


class Model1462Row7Col(tf.keras.Model):

	def __init__(self):
		super(Model1462Row7Col, self).__init__()
		self.base_block1 = self.init_base_block1()
		self.base_block2 = self.init_base_block2()
		self.base_block3 = self.init_base_block3()
		self.base_block4 = self.init_base_block4()
		self.base_batchnorms = [self.init_bn_relu('base' + str(i)) for i in range(1, 5)]
		self.concat = tf.concat
		self.copy = tf.identity
		# self.add = Add()
		self.num_units = [1, 4, 7, 22, 56, 43]
		self.inter_blocks = [self.init_kth_inter_block(k, self.num_units[k]) for k in range(6)]
		self.out_batchnorms = [self.init_bn_relu(k) for k in range(6)]
		# self.dropouts = [Dropout(0.5) for k in range(6)]
		self.out_blocks = [self.init_kth_out_block(k, self.num_units[k]) for k in range(6)]
		self.postproc_layers = [self.init_kth_post_proc_layer(k) for k in range(6)]

	def init_base_block1(self):
		block = tf.keras.Sequential(name='base_block1')
		block.add(Conv1D(64, kernel_size=2, strides=1, padding='same',
						 kernel_initializer=he_normal, use_bias=True, input_shape=(1462, 7)))
		block.add(Conv1D(64, kernel_size=2, strides=2, kernel_initializer=he_normal, use_bias=True))
		return block

	def init_base_block2(self):
		block = tf.keras.Sequential(name='base_block2')
		block.add(Conv1D(128, kernel_size=3, strides=2, kernel_initializer=he_normal, use_bias=True))
		block.add(Conv1D(128, kernel_size=3, strides=2, kernel_initializer=he_normal, use_bias=True))
		return block

	def init_base_block3(self):
		block = tf.keras.Sequential(name='base_block3')
		block.add(Conv1D(256, kernel_size=2, strides=2, kernel_initializer=he_normal, use_bias=True))
		block.add(Conv1D(256, kernel_size=4, strides=3, kernel_initializer=he_normal, use_bias=True))
		block.add(Conv1D(256, kernel_size=2, strides=2, kernel_initializer=he_normal, use_bias=True))
		return block

	def init_base_block4(self):
		block = tf.keras.Sequential(name='base_block4')
		block.add(Conv1D(512, kernel_size=3, strides=2, kernel_initializer=he_normal, use_bias=True))
		block.add(Conv1D(512, kernel_size=7, strides=1, kernel_initializer=he_normal, use_bias=True))
		block.add(Conv1D(1024, kernel_size=1, strides=1, kernel_initializer=he_normal, use_bias=True))
		# block.add(BatchNormalization())
		# block.add(tfa.layers.GroupNormalization(32))
		# block.add(Activation('relu'))
		# block.add(Dropout(0.2))
		block.add(Flatten())
		return block

	def init_bn_relu(self, k):
		if type(k) == str:
			# if k.startswith('base'):
			block = tf.keras.Sequential(name='base_bn_relu_block_' + k[-1])
		else:
			block = tf.keras.Sequential(name=str(k) + 'th_bn_relu_block')
		block.add(BatchNormalization(momentum=0.9))
		block.add(Activation('relu'))
		# block.add(Dropout(0.5))
		return block

	def init_kth_inter_block(self, k: int, out_units):
		block = tf.keras.Sequential(name=str(k) + 'th_inter_block')
		block.add(Dense(256, name='l' + str(k) + '_inter_dense_0', input_shape=(1024,), use_bias=True,
						kernel_initializer=he_normal))
		block.add(Activation('relu'))
		# block.add(Dropout(0.5))
		block.add(Dense(128, name='l' + str(k) + '_inter_dense_1', use_bias=True, kernel_initializer=he_normal))
		# block.add(BatchNormalization())
		block.add(Activation('relu'))
		block.add(Dense(64, name='l' + str(k) + '_inter_dense_2', use_bias=True, kernel_initializer=he_normal))
		# block.add(BatchNormalization())
		# block.add(Activation('relu'))
		# block.add(Dense(num_units[1], name='l'+str(k)+'_inter_dense_3', use_bias=True, kernel_initializer=he_normal))
		return block

	def init_kth_out_block(self, k: int, out_units):
		filter_by_size = lambda x: x[(x > out_units * 4) & (x <= out_units * 16)][::-1]
		num_units = filter_by_size(2 ** np.arange(11))
		# modify here
		block = tf.keras.Sequential(name=str(k) + 'th_out_block')
		# block.add(Dropout(0.2))
		block.add(Dense(num_units[0], name='l' + str(k) + '_out_dense_0', use_bias=True, kernel_initializer=he_normal))
		block.add(BatchNormalization())
		block.add(Activation('relu'))
		# block.add(Dropout(0.3))
		# block.add(Dense(num_units[0], name='l'+str(k)+'_out_dense_1', use_bias=True, kernel_initializer=he_normal))
		# block.add(BatchNormalization())
		# block.add(Activation('relu'))
		# block.add(Dropout(0.3))
		block.add(Dense(num_units[1], name='l' + str(k) + '_out_dense_2', use_bias=True, kernel_initializer=he_normal))
		block.add(BatchNormalization())
		block.add(Activation('relu'))
		block.add(Dropout(0.5))
		block.add(Dense(out_units, name='l' + str(k), activation='sigmoid', use_bias=True))
		return block

	def init_kth_post_proc_layer(self, k):
		def scale_output(x):
			total_contrib = tf.constant([[1]], dtype=tf.float32, shape=(1, 1))
			unknown_contrib = tf.subtract(total_contrib, tf.keras.backend.sum(x, keepdims=True, axis=1))
			contrib = tf.keras.backend.relu(tf.keras.backend.concatenate((x, unknown_contrib), axis=1))
			scaled_contrib = tf.divide(contrib, tf.keras.backend.sum(contrib, keepdims=True, axis=1))
			return scaled_contrib

		return Lambda(scale_output, name='l' + str(k) + '_y')

	def call(self, input, training=False):
		base = self.base_block1(input, training=training)
		base = self.base_batchnorms[0](base, training=training)
		base = self.base_block2(base, training=training)
		base = self.base_batchnorms[1](base, training=training)
		base = self.base_block3(base, training=training)
		base = self.base_batchnorms[2](base, training=training)
		base = self.base_block4(base, training=training)
		base = self.base_batchnorms[3](base, training=training)

		# base_copy = self.copy(base)
		inter_factors = [self.inter_blocks[i](base, training=training) for i in range(6)]
		concat_factors = [self.concat(inter_factors[i - 1:i + 1], axis=1) if i > 0 else inter_factors[i] for i in
						  range(6)]
		# concat_factors = [self.add(inter_factors[0:i]) for i in range(1, 7)]
		concat_factors = [self.out_batchnorms[i](concat_factors[i], training=training) for i in range(6)]
		# concat_factors = [self.dropouts[i](concat_factors[i], training=training) for i in range(6)]
		y_s = [self.out_blocks[i](concat_factors[i], training=training) for i in range(6)]
		(l0_y, l1_y, l2_y, l3_y, l4_y, l5_y) = (self.postproc_layers[i](y_s[i], training=training) for i in range(6))

		return l0_y, l1_y, l2_y, l3_y, l4_y, l5_y


"""
6 by 1000 -(512*f3)-> 4 by 512 -(512*f3)-> 2 by 512 -(512*f2)-> 1 by 512 -flatten-> 512 -fc512-> base
base - 10 * inter_block(fc32)-> concatenate -> split -> output_block
"""
class Model6Row1000Col(tf.keras.Model):

	def __init__(self):
		super(Model6Row1000Col, self).__init__()
		self.base_block = self.init_base_block()
		self.inter_blocks = [self.init_inter_block(i) for i in range(20)]
		self.concat = tf.concat
		self.copy = tf.identity
		# self.add = Add()
		self.num_units = [1, 4, 7, 22, 56, 43]
		# self.dropouts = [Dropout(0.5) for k in range(6)]
		self.out_blocks = [self.init_kth_out_block(k, self.num_units[k]) for k in range(6)]
		self.postproc_layers = [self.init_kth_post_proc_layer(k) for k in range(6)]

	def init_base_block(self):
		block = tf.keras.Sequential(name='base_block')
		block.add(Conv1D(64, kernel_size=3, kernel_initializer=he_normal, input_shape=(1000, 6, 1)))
		block.add(Activation('relu'))
		block.add(Conv1D(512, kernel_size=3, kernel_initializer=he_normal))
		block.add(Activation('relu'))
		block.add(Conv1D(512, kernel_size=2, kernel_initializer=he_normal))
		block.add(Activation('relu'))
		block.add(Conv1D(512, kernel_size=1, kernel_initializer=he_normal))
		block.add(Activation('relu'))
		block.add(Flatten())
		block.add(Dense(512, kernel_initializer=he_normal))
		block.add(Activation('relu'))
		return block

	def init_inter_block(self, index: int):
		k = index
		block = tf.keras.Sequential(name=str(k) + 'th_inter_block')
		block.add(Dense(128, name='l' + str(k) + '_inter_fc0', input_shape=(512, ),
						kernel_initializer=he_normal))
		block.add(Activation('relu'))
		block.add(Dense(128, name='l' + str(k) + '_inter_fc1', kernel_initializer=he_normal))
		block.add(Activation('relu'))
		block.add(Dense(64, name='l' + str(k) + '_inter_fc2', kernel_initializer=he_normal))
		block.add(Activation('relu'))
		block.add(Dense(64, name='l' + str(k) + '_inter_fc3', kernel_initializer=he_normal))
		block.add(Activation('relu'))
		return block

	def init_bn_block(self, k):
		block = tf.keras.Sequential(name=str(k) + 'th_bn_block')
		block.add(BatchNormalization(momentum=0.99))
		block.add(Activation('relu'))
		return block

	def init_out_block(self, index: int, num_source):
		k = index
		filter_by_size = lambda x: x[(x > num_source * 4) & (x <= num_source * 16)][::-1]
		num_units = filter_by_size(2 ** np.arange(11))
		block = tf.keras.Sequential(name=str(k) + 'th_out_block')
		block.add(Dense(num_units[0], name='l' + str(k) + '_out_dense_0', kernel_initializer=he_normal))
		block.add(BatchNormalization())
		block.add(Activation('relu'))
		block.add(Dense(num_units[1], name='l' + str(k) + '_out_dense_1', kernel_initializer=he_normal))
		block.add(BatchNormalization())
		block.add(Activation('relu'))
		block.add(Dropout(0.7))
		block.add(Dense(num_source, name='l' + str(k), use_bias=True))
		return block

	def init_kth_post_proc_layer(self, k):
		def scale_output(x):
			total_contrib = tf.constant([[1]], dtype=tf.float32, shape=(1, 1))
			unknown_contrib = tf.subtract(total_contrib, tf.keras.backend.sum(x, keepdims=True, axis=1))
			contrib = tf.keras.backend.relu(tf.keras.backend.concatenate((x, unknown_contrib), axis=1))
			scaled_contrib = tf.divide(contrib, tf.keras.backend.sum(contrib, keepdims=True, axis=1))
			return scaled_contrib

		return Lambda(scale_output, name='l' + str(k) + '_y')

	def call(self, input, training=False):
		base = self.base_block(input, training=training)

		# base_copy = self.copy(base)
		inter_factors = [self.inter_blocks[i](base, training=training) for i in range(6)]
		concat_factors = [self.concat(inter_factors[i - 1:i + 1], axis=1)
						  if i > 0 else inter_factors[i]
						  for i in range(6)]
		# concat_factors = [self.add(inter_factors[0:i]) for i in range(1, 7)]
		concat_factors = [self.out_batchnorms[i](concat_factors[i], training=training) for i in range(6)]
		# concat_factors = [self.dropouts[i](concat_factors[i], training=training) for i in range(6)]
		y_s = [self.out_blocks[i](concat_factors[i], training=training) for i in range(6)]
		(l0_y, l1_y, l2_y, l3_y, l4_y, l5_y) = (self.postproc_layers[i](y_s[i], training=training)
												for i in range(6))

		return l0_y, l1_y, l2_y, l3_y, l4_y, l5_y
