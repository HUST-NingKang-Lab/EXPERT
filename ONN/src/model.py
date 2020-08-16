import os
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Concatenate, \
	Conv2D, Activation, Lambda
import tensorflow as tf
from collections import OrderedDict
import numpy as np


he_uniform = tf.keras.initializers.HeUniform(seed=1)

# transfer: load saved model, build new model from scratch, new model.base = saved model.base

class Model(tf.keras.Model):

	def __init__(self, layer_units=None, restore_from=None):
		"""
		:param phylogeny:
		:param ontology:
		:param extractor:
		"""
		# init Keras Model
		super(Model, self).__init__()
		# fine tune flag
		self.fine_tune = False
		if layer_units:
			self.n_layers = len(layer_units)
			self.base = self.init_base_block()
			self.spec_inters = [self.init_inter_block(index=layer, name='l{}_inter'.format(layer), 
													  n_units=n_units)
							   for layer, n_units in enumerate(layer_units)]
			self.spec_outputs = [self.init_output_block(index=layer, name='l{}_output'.format(layer),
														n_units=n_units)
								 for layer, n_units in enumerate(layer_units)]
		elif restore_from:
			self.__restore_from(restore_from)
			self.n_layers = len(self.spec_outputs)
		else:
			raise ValueError('Please given correct model path to restore, '
							 'or specify layer_units to build model from scratch.')
		self.concat = Concatenate(axis=1)
		self.spec_postprocs = [self.init_post_proc_layer(name='l{}_postproc'.format(layer))
							   for layer in range(self.n_layers)]

	def init_base_block(self):
		block = tf.keras.Sequential(name='base')
		block.add(Conv2D(64, kernel_size=(1, 3), use_bias=False, kernel_initializer=he_uniform, input_shape=(1000, 6, 1)))
		block.add(self._init_bn_layer())
		block.add(Activation('relu')) # (1000, 4, 64) -> 256000
		block.add(Conv2D(64, kernel_size=(1, 2), use_bias=False, kernel_initializer=he_uniform))
		block.add(self._init_bn_layer())
		block.add(Activation('relu')) # (1000, 3, 64) -> 192000
		block.add(Conv2D(128, kernel_size=(1, 2), use_bias=False, kernel_initializer=he_uniform))
		block.add(self._init_bn_layer())
		block.add(Activation('relu')) # (1000, 2, 64) -> 128000
		block.add(Conv2D(128, kernel_size=(1, 2), use_bias=False, kernel_initializer=he_uniform))
		block.add(self._init_bn_layer())
		block.add(Activation('relu')) # (1000, 1, 64) -> 64000
		block.add(Conv2D(128, kernel_size=(10, 1), strides=(5, 1), use_bias=False, kernel_initializer=he_uniform))
		block.add(self._init_bn_layer())
		block.add(Activation('relu')) # (198, 1, 128) -> 25344
		block.add(Conv2D(256, kernel_size=(6, 1), strides=(3, 1), use_bias=False, kernel_initializer=he_uniform))
		block.add(self._init_bn_layer())
		block.add(Activation('relu')) # (65, 1, 128) -> 8320
		block.add(Conv2D(256, kernel_size=(5, 1), strides=(5, 1), use_bias=False, kernel_initializer=he_uniform))
		block.add(self._init_bn_layer())
		block.add(Activation('relu')) # (13, 1, 256) -> 3328
		block.add(Conv2D(256, kernel_size=(5, 1), strides=(2, 1), use_bias=False, kernel_initializer=he_uniform))
		block.add(self._init_bn_layer())
		block.add(Activation('relu')) # (5, 1, 256) -> 1280
		block.add(Conv2D(512, kernel_size=(5, 1), use_bias=False, kernel_initializer=he_uniform))
		block.add(self._init_bn_layer())
		block.add(Activation('relu')) # (1, 1, 512) -> 512
		block.add(Flatten()) # (512, )
		block.add(Dense(512, use_bias=False, kernel_initializer=he_uniform))
		block.add(self._init_bn_layer())
		block.add(Activation('relu')) # (512, )
		return block

	def init_inter_block(self, index, name, n_units):
		k = index
		block = tf.keras.Sequential(name=name)
		block.add(Dense(self._get_n_units(n_units*8), name='l' + str(k) + '_inter_fc0', input_shape=(512, ), use_bias=False,
						kernel_initializer=he_uniform))
		block.add(self._init_bn_layer())
		block.add(Activation('relu'))
		block.add(Dense(self._get_n_units(n_units*4), name='l' + str(k) + '_inter_fc1', use_bias=False, kernel_initializer=he_uniform))
		block.add(self._init_bn_layer())
		block.add(Activation('relu'))
		block.add(Dense(self._get_n_units(n_units*4), name='l' + str(k) + '_inter_fc2', use_bias=False, kernel_initializer=he_uniform))
		block.add(self._init_bn_layer())
		block.add(Activation('relu'))
		block.add(Dense(self._get_n_units(n_units*4), name='l' + str(k) + '_inter_fc3', use_bias=False, kernel_initializer=he_uniform))
		block.add(self._init_bn_layer())
		block.add(Activation('relu'))
		return block

	def init_output_block(self, index, name, n_units):
		#input_shape = (128 * 2 if index > 0 else 128, )
		block = tf.keras.Sequential(name=name)
		block.add(Dense(self._get_n_units(n_units*8), name='l' + str(index) + '_out_fc0', use_bias=False,
						kernel_initializer=he_uniform))
		block.add(self._init_bn_layer())
		block.add(Activation('relu'))
		block.add(Dense(self._get_n_units(n_units*8), name='l' + str(index) + '_out_fc1', use_bias=False,
						kernel_initializer=he_uniform))
		block.add(self._init_bn_layer())
		block.add(Activation('relu'))
		block.add(Dense(self._get_n_units(n_units*8), name='l' + str(index) + '_out_fc2', use_bias=False,
						kernel_initializer=he_uniform))
		block.add(self._init_bn_layer())
		block.add(Activation('relu'))
		#block.add(Dropout(0.3))
		block.add(Dense(n_units, name='l' + str(index)))
		block.add(Activation('sigmoid'))
		return block

	def init_post_proc_layer(self, name):
		def scale_output(x):
			total_contrib = tf.constant([[1]], dtype=tf.float32, shape=(1, 1))
			unknown_contrib = tf.subtract(total_contrib, tf.keras.backend.sum(x, keepdims=True, axis=1))
			contrib = tf.keras.backend.relu(tf.keras.backend.concatenate((x, unknown_contrib), axis=1))
			scaled_contrib = tf.divide(contrib, tf.keras.backend.sum(contrib, keepdims=True, axis=1))
			return scaled_contrib
		return Lambda(scale_output, name=name)

	def _init_bn_layer(self):
		return BatchNormalization(momentum=0.9)

	def call(self, inputs, training=False):
		if self.fine_tune == False:
			base = self.base(inputs, training=training)
		else:
			base = self.base(inputs, training=False)

		inter_logits = [self.spec_inters[i](base, training=training)
						for i in range(self.n_layers)]

		concat_logits = [self.concat(inter_logits[i-1:i+1]) if i >= 1 else inter_logits[0]
						 for i in range(self.n_layers)]

		out_logits = [self.spec_outputs[i](concat_logits[i], training=training)
					  for i in range(self.n_layers)]

		#outputs = {'output_'+str(i): self.spec_postprocs[i](out_logits[i], training=training)
		#		   for i in range(self.n_layers)}
		outputs = [self.spec_postprocs[i](out_logits[i], training=training)
				   for i in range(self.n_layers)]
		if self.n_layers == 5:
			l1, l2, l3, l4, l5 = tuple(outputs)
			return l1, l2, l3, l4, l5
		elif self.n_layers == 4:
			l1, l2, l3, l4 = tuple(outputs)
			return l1, l2, l3, l4
		elif self.n_layers == 3:
			l1, l2, l3 = tuple(outputs)
			return l1, l2, l3
		elif self.n_layers == 2:
			l1, l2 = tuple(outputs)
			return l1, l2
		elif self.n_layers == 1:
			l1 = outputs[0]
			return l1
		else:
			return self.concat(outputs)


	def _get_n_units(self, num):
		# closest binary exponential number larger than num
		supported_range = 2**np.arange(1, 11)
		return supported_range[(supported_range < num).sum()]

	def save_base_model(self, path):
		self.base.save(path, save_format='tf')

	def save_blocks(self, path):
		inters_dir = self.__pthjoin(path, 'inters')
		outputs_dir = self.__pthjoin(path, 'outputs')
		for dir in [path, inters_dir, outputs_dir]:
			if not os.path.isdir(dir):
				os.mkdir(dir)
		self.base.save(self.__pthjoin(path, 'base'), save_format='tf')
		for layer in range(self.n_layers):
			self.spec_inters[layer].save(self.__pthjoin(inters_dir, str(layer)), save_format='tf')
			self.spec_outputs[layer].save(self.__pthjoin(outputs_dir, str(layer)), save_format='tf')

	def __restore_from(self, path):
		base_dir = self.__pthjoin(path, 'base')
		inters_dir = self.__pthjoin(path, 'inters')
		outputs_dir = self.__pthjoin(path, 'outputs')
		inter_dirs = [self.__pthjoin(inters_dir, i) for i in os.listdir(inters_dir)]
		output_dirs = [self.__pthjoin(outputs_dir, i) for i in os.listdir(outputs_dir)]
		self.base = tf.keras.models.load_model(base_dir)
		self.spec_inters = [tf.keras.models.load_model(dir) for dir in inter_dirs]
		self.spec_outputs = [tf.keras.models.load_model(dir) for dir in output_dirs]

	def __pthjoin(self, pth1, pth2):
		return os.path.join(pth1, pth2)
