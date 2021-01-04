import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Concatenate, \
	Conv2D, Activation, Lambda, Layer, Input, GaussianNoise, AlphaDropout
from collections import OrderedDict
import numpy as np
import pandas as pd
from tensorflow.keras import backend as K
from livingTree import SuperTree

init = tf.keras.initializers.HeUniform()
#init = tf.keras.initializers.LecunNormal()
sig_init = tf.keras.initializers.GlorotUniform()


class Model(object):

	def __init__(self, phylogeny, num_features, ontology=None, restore_from=None, dropout_rate=0, open_set=False):
		self.expand_dims = tf.expand_dims
		self.concat = Concatenate(axis=1)
		self.concat_a2 = Concatenate(axis=2)
		self.dropout = Dropout(dropout_rate)
		self.open_set = open_set
		if ontology:
			self.ontology = ontology
			self.labels, self.layer_units = parse_otlg(self.ontology)
			self.n_layers = len(self.layer_units)
			self.statistics = pd.DataFrame(index=range(phylogeny.shape[0] * phylogeny.shape[1]), columns=['mean', 'std'])
			self.base = self.init_base_block(num_features=num_features)
			self.spec_inters = [self.init_inter_block(index=layer, name='l{}_inter'.format(layer+2), n_units=n_units)
								for layer, n_units in enumerate(self.layer_units)]
			self.spec_integs = [self.init_integ_module(index=layer, name='l{}_integration'.format(layer + 2), n_units=n_units)
								for layer, n_units in enumerate(self.layer_units)]
			self.spec_outputs = [self.init_output_module(index=layer, name='l{}o'.format(layer + 2), n_units=n_units)
								 for layer, n_units in enumerate(self.layer_units)]
		elif restore_from:
			self.__restore_from(restore_from)
			self.n_layers = len(self.spec_outputs)
		else:
			raise ValueError('Please given correct model path to restore, '
							 'or specify layer_units to build model from scratch.')
		self.encoder = self.init_encoder_block(phylogeny)
		self.spec_postprocs = [self.init_post_proc_module(name='l{}'.format(layer + 2)) for layer in range(self.n_layers)]
		self.nn = self.build_graph(input_shape=(num_features * phylogeny.shape[1],))

	def save_blocks(self, path):
		inters_dir = self.__pthjoin(path, 'inters')
		integs_dir = self.__pthjoin(path, 'integs')
		outputs_dir = self.__pthjoin(path, 'outputs')
		for dir in [path, inters_dir, outputs_dir]:
			if not os.path.isdir(dir):
				os.mkdir(dir)
		#self.feature_mapper.save(self.__pthjoin(path, 'feature_mapper'))
		self.base.save(self.__pthjoin(path, 'base'), save_format='tf')
		self.ontology.to_pickle(self.__pthjoin(path, 'ontology.pkl'))
		self.statistics.to_csv(self.__pthjoin(path, 'statistics.csv'))
		for layer in range(self.n_layers):
			self.spec_inters[layer].save(self.__pthjoin(inters_dir, str(layer)), save_format='tf')
			self.spec_integs[layer].save(self.__pthjoin(integs_dir, str(layer)), save_format='tf')
			self.spec_outputs[layer].save(self.__pthjoin(outputs_dir, str(layer)), save_format='tf')

	def __restore_from(self, path):
		#mapper_dir = self.__pthjoin(path, 'feature_mapper')
		otlg_dir = self.__pthjoin(path, 'ontology.pkl')
		base_dir = self.__pthjoin(path, 'base')
		inters_dir = self.__pthjoin(path, 'inters')
		integs_dir = self.__pthjoin(path, 'integs')
		outputs_dir = self.__pthjoin(path, 'outputs')
		inter_dirs = [self.__pthjoin(inters_dir, i) for i in sorted(os.listdir(inters_dir), key=lambda x: int(x))]
		integ_dirs = [self.__pthjoin(integs_dir, i) for i in sorted(os.listdir(integs_dir), key=lambda x: int(x))]
		output_dirs = [self.__pthjoin(outputs_dir, i) for i in sorted(os.listdir(outputs_dir), key=lambda x: int(x))]
		self.ontology = load_otlg(otlg_dir)
		self.statistics = pd.read_csv(self.__pthjoin(path, 'statistics.csv'), index_col=0)
		self.labels, self.layer_units = parse_otlg(self.ontology)
		self.base = tf.keras.models.load_model(base_dir)
		self.spec_inters = [tf.keras.models.load_model(dir) for dir in inter_dirs]
		self.spec_integs = [tf.keras.models.load_model(dir) for dir in integ_dirs]
		self.spec_outputs = [tf.keras.models.load_model(dir) for dir in output_dirs]

	def init_mapper_block(self, num_features): # map input feature to ...
		block = tf.keras.Sequential(name='feature_mapper')
		block.add(Mapper(num_features=num_features, name='feature_mapper_layer'))
		return block

	def init_encoder_block(self, phylogeny):
		block = tf.keras.Sequential(name='feature_encoder')
		block.add(Encoder(phylogeny))
		return block

	def init_base_block(self, num_features):
		block = tf.keras.Sequential(name='base')
		block.add(Flatten()) # (1000, )
		block.add(Dense(2**10, kernel_initializer=init))
		block.add(Activation('relu')) # (1024, )
		block.add(Dense(2**9, kernel_initializer=init))
		block.add(Activation('relu')) # (512, )
		return block

	def init_inter_block(self, index, name, n_units):
		k = index
		block = tf.keras.Sequential(name=name)
		block.add(Dense(self._get_n_units(8*n_units), name='l' + str(k) + '_inter_fc0', kernel_initializer=init))
		block.add(Activation('relu'))
		block.add(Dense(self._get_n_units(4*n_units), name='l' + str(k) + '_inter_fc1', kernel_initializer=init))
		block.add(Activation('relu'))
		block.add(Dense(self._get_n_units(2*n_units), name='l' + str(k) + '_inter_fc2', kernel_initializer=init))
		block.add(Activation('relu'))
		return block

	def init_integ_module(self, index, name, n_units):
		block = tf.keras.Sequential(name=name)
		k = index
		block.add(Dense(self._get_n_units(3*n_units), name='l' + str(k) + '_integ_fc0', kernel_initializer=sig_init))
		block.add(Activation('tanh'))
		return block

	def init_output_module(self, index, name, n_units):
		k = index
		block = tf.keras.Sequential(name=name)
		block.add(Dense(n_units, name='l' + str(index+2) + 'o_fc', kernel_initializer=sig_init))
		#block.add(Activation('sigmoid'))
		return block

	def init_post_proc_module(self, name):
		def calculateUnknownContribution(x):
			#x = K.relu(x)
			total_contrib = tf.constant([[1]], dtype=tf.float32, shape=(1, 1))
			unknown_contrib = K.relu(tf.subtract(total_contrib, K.sum(x, keepdims=True, axis=1)))
			contrib = K.concatenate((x, unknown_contrib), axis=1)
			scaled_contrib = tf.divide(contrib, K.sum(contrib, keepdims=True, axis=1))
			return scaled_contrib

		block = tf.keras.Sequential(name=name)
		if self.open_set:
			block.add(Activation('sigmoid'))
			block.add(Lambda(calculateUnknownContribution, name=name+'_l'))
		else:
			block.add(Activation('softmax'))
		return block

	def build_graph(self, input_shape):
		inputs = Input(shape=input_shape)
		#features = self.encoder(inputs)
		features = self.dropout(inputs)
		base = self.base(features)
		inter_logits = [self.spec_inters[i](base) for i in range(self.n_layers)]
		integ_logits = []
		for layer in range(self.n_layers):
			if layer == 0:
				integ_logits.append(self.spec_integs[layer](inter_logits[layer]))
			else:
				#logits = self.concat([0.1 * integ_logits[layer-1], inter_logits[layer]])
				logits = self.concat([integ_logits[layer - 1], inter_logits[layer]])
				integ_logits.append(self.spec_integs[layer](logits))
		out_probas = [self.spec_outputs[i](integ_logits[i]) for i in range(self.n_layers)]
		nn = tf.keras.Model(inputs=inputs, outputs=out_probas)
		return nn

	def build_estimator(self):
		inputs = Input(shape=self.nn.input_shape[1:])
		logits = self.nn(inputs)
		if self.n_layers == 1:
			logits = [logits]
		contrib = [self.spec_postprocs[i](logits[i]) for i in range(self.n_layers)]
		self.estimator = tf.keras.Model(inputs=inputs, outputs=contrib)

	def update_statistics(self, mean, std):
		stats = self.statistics.copy()
		stats.loc[:, 'mean'] = mean.tolist()
		stats.loc[:, 'std'] = std.tolist()
		self.statistics = stats
		print(self.statistics)

	def standardize(self, X):
		return pd.DataFrame(X).sub(self.statistics['mean'], axis=1).div(self.statistics['std'] + 1e-8, axis=1)

	def _init_bn_layer(self):
		return BatchNormalization(momentum=0.9, scale=False)

	def _get_n_units(self, num):
		return int(num)

	def __pthjoin(self, pth1, pth2):
		return os.path.join(pth1, pth2)


class Mapper(Layer): # A PCA learner
	
	def __init__(self, num_features, name=None, **kwargs):
		super(Mapper, self).__init__(name=name)
		super(Mapper, self).__init__(kwargs)
		self.num_features = num_features
		self.w = self.add_weight(shape=(1024, num_features), name='w', initializer="random_normal", trainable=True)
		self.matmul = tf.matmul
	
	def call(self, inputs):
		outputs = self.matmul(self.w, inputs)
		return outputs

	def get_config(self):
		config = super(Mapper, self).get_config()
		config.update({"num_features": self.num_features})
		return config


class Encoder(Layer):

	def __init__(self, phylogeny, name=None, **kwargs):
		super(Encoder, self).__init__(name=name)
		super(Encoder, self).__init__(kwargs)
		self.ranks = phylogeny.columns.to_list()[:-1]
		self.W = {rank: self.get_W(phylogeny[rank]) for rank in self.ranks}
		self.dot = K.dot
		self.concatenate = K.concatenate
		self.expand_dims = K.expand_dims

	def get_W(self, taxons):
		cols = taxons.to_numpy().reshape(taxons.shape[0], 1)
		rows = taxons.to_numpy().reshape(1, taxons.shape[0])
		return tf.constant((rows == cols).astype(np.float32))

	def call(self, inputs):
		F_genus = inputs
		F_ranks = [self.expand_dims(self.dot(F_genus, self.W[rank]), axis=2) for rank in self.ranks] + \
				  [self.expand_dims(F_genus, axis=2)]
		outputs = self.concatenate(F_ranks, axis=2)
		return outputs

	def get_config(self):
		config = super(Encoder, self).get_config()
		config.update({"W": self.W, "ranks": self.ranks})
		return config


# re-define here to avoid circular importing

def load_otlg(path):
	otlg = SuperTree().from_pickle(path)
	return otlg

def parse_otlg(ontology):
	labels = OrderedDict([(layer, label) for layer, label in ontology.get_ids_by_level().items()
						  if layer > 0])
	layer_units = [len(label) for layer, label in labels.items()]
	return labels, layer_units
