from .utils import Parser
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Concatenate, Conv2D, MaxPooling2D
import tensorflow as tf
from collections import OrderedDict
import numpy as np

parser = Parser()


class Model(tf.keras.Model):

	def __init__(self, phylogeny, ontology, extractor, fine_tune=False):
		"""
		stacked_network=False: Independent feature extraction and source contribution calculation.
		stacked_network=True:  Stack feature extractor and predictor together.
		:param phylogeny:
		:param ontology:
		:param extractor:
		"""
		# init Keras Model
		super(Model, self).__init__()

		self.phylogeny = phylogeny
		self.ontology = ontology
		self.do_fine_tune = fine_tune
		if self.do_fine_tune:
			extractor.trainable = False

		self.input_shape_ = phylogeny.shape
		(self.output_labels, self.output_shapes) = parser.parse_ontology(ontology) # walk through ontology?
		self.layer_keys = list(self.output_shapes.keys())

		# generate neural network
		self.ftr_extrctor = extractor
		self.concat_layer = Concatenate(axis=1)
		self.otlg_walker = OrderedDict([])
		self._add_out_layers() # Ordered_dict

	def call(self, inputs, training=False):
		# store outputs
		outs = OrderedDict()
		ftrs = self.ftr_extrctor(inputs, training=training)
		for index, nnlayer in enumerate(self.otlg_walker.keys()):
			X = ftrs.copy()
			if index == 0:
				for key, layer in self.otlg_walker[nnlayer].items():
					if key.startswith('batchnorm'):
						X = layer(X, training=training)
					else:
						X = layer(X)
			else:
				for key, layer in self.otlg_walker[nnlayer].items():
					if key.startswith('batchnorm'):
						X = layer(X, training=training)
					elif key == 'dense1':
						'''output of dense1 -> Concatenation <- output of last layer of ONN
												      |-> dense'''
						X = layer(X)
						lastout = layer(outs[index - 1])
						X = self.concat_layer([lastout, X])
					else:
						X = layer(X)
			outs[index] = X
		concat_outs = tf.concat(list(outs.values()), axis=0)
		return concat_outs

	def _add_out_layers(self):
		"""
		For layer 0: features -> dense1 -> flatten -> dense2(out)
		For other layers: features -> Batchnorm -> dense1 -> concat <- dense1 <- output of last layer
															   |-> dense2 -> flatten -> dense3(out)
		nnlayer: each layer of output layers of the entire NN
		key: name for each layer in each nested out layers
		layer: each layer in each nested out layers
		:return:
		"""
		example = {1: (4, 1),
				   2: (10, 1)}   # Ordered dict
		for index, (nnlayer, shape) in enumerate(self.output_shapes.items()):
			self.otlg_walker[nnlayer] = OrderedDict()
			NL = shape[0]    # nlabel
			if index == 0:
				self.otlg_walker[nnlayer]['batchnorm'] = BatchNormalization()
				self.otlg_walker[nnlayer]['dense1'] = Dense(self._get_n_units(NL * 2), activation='relu')
				self.otlg_walker[nnlayer]['flatten'] = Flatten()
				self.otlg_walker[nnlayer]['dense2'] = Dense(NL, activation='softmax')
			else:
				self.otlg_walker[nnlayer]['batchnorm'] = BatchNormalization()
				self.otlg_walker[nnlayer]['dense1'] = Dense(self._get_n_units(NL * 4), activation='relu')
				'''output of dense1 -> Concatenation <- output of last layer of ONN
				             				|-> dense'''
				self.otlg_walker[nnlayer]['dense2'] = Dense(self._get_n_units(NL * 2), activation='relu')
				self.otlg_walker[nnlayer]['flatten'] = Flatten()
				self.otlg_walker[nnlayer]['dense3'] = Dense(NL, activation='softmax')

	def _get_n_units(self, num):
		# closest binary exponential number larger than num
		suupported_range = 2**np.arange(1, 11)
		return suupported_range[(suupported_range < num).sum()]

	def predict_source(self, X, cal_contribution=True, threshold=0, top_n=0):
		"""
		Predict source or source contribution of input X.
		:param X:
		:param cal_contribution:
		:param threshold:
		:param top_n:
		:return:
		"""
		if cal_contribution and (threshold != 0 or top_n != 0):
			assert RuntimeWarning('Cannot apply threshold and top_n when calculating source contribution.')
		else:
			pass

		if self.stacked_network and cal_contribution:
			source_contributions = self.predictor.predict(X)
		elif self.stacked_network and not cal_contribution:
			source_contributions = self.predictor.predict(X)
			return self._post_processing(source_contributions, threshold, top_n)
		elif cal_contribution and not self.stacked_network:
			features = self._extract_features(X)
			source_contributions = self.predictor.predict(features)
		else: # not cal contribution and not stacked network
			features = self._extract_features(X)
			source_contributions = self.predictor.predict(features)
			return self._post_processing(source_contributions, threshold, top_n)
		return source_contributions

	def _post_processing(self, source_contribution, threshold=0, top_n=0):
		# apply softmax
		source_contribution
		return source_contribution
