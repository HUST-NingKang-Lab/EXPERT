import os
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Concatenate, \
	Conv2D, Activation, Lambda, Layer
import tensorflow as tf
from collections import OrderedDict
import numpy as np
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K


he_uniform = tf.keras.initializers.HeUniform(seed=0)

# transfer: load saved model, build new model from scratch, new model.base = saved model.base

class Model(tf.keras.Model):

	def __init__(self, phylogeny, layer_units=None, num_features=1000, restore_from=None):
		"""
		:param phylogeny:
		:param ontology:
		:param extractor:
		"""
		# init Keras Model
		super(Model, self).__init__()
		# fine tune flag
		self.fine_tune = False
		print('Set .fine_tune to True to protect BatchNormalization layers in base block when doing Fine-tuning')
		if layer_units:
			self.n_layers = len(layer_units)
			
			'''self.feature_mapper = self.init_mapper_block(num_features=num_features)
			# Avoiding Non-trainble params bug in tensorflow 2.3.0
			self.feature_mapper.trainable = False
			self.feature_mapper.trainable = True'''
			self.encoder = self.init_encoder(phylogeny)
			self.base = self.init_base_block()
			self.spec_inters = [self.init_inter_block(index=layer, name='l{}_inter'.format(layer),
													  n_units=n_units)
							   for layer, n_units in enumerate(layer_units)]
			self.spec_integs = [self._init_integ_block(index=layer, name='l{}_integration'.format(layer),
														n_units=n_units)
								 for layer, n_units in enumerate(layer_units)]
			
			self.spec_outputs = [self.init_output_block(index=layer, name='l{}_output'.format(layer),
														n_units=n_units)
								 for layer, n_units in enumerate(layer_units)]
		elif restore_from:
			self.__restore_from(restore_from)  # finish here
			self.n_layers = len(self.spec_outputs)
		else:
			raise ValueError('Please given correct model path to restore, '
							 'or specify layer_units to build model from scratch.')
		self.expand_dims = tf.expand_dims
		self.concat = Concatenate(axis=1)
		self.spec_postprocs = [self.init_post_proc_layer(name='l{}_postproc'.format(layer))
							   for layer in range(self.n_layers)]

	def init_mapper_block(self, num_features): # map input feature to ...
		block = tf.keras.Sequential(name='feature_mapper')
		block.add(Mapper(num_features=num_features, name='feature_mapper_layer'))
		block.add(self._init_bn_layer())
		return block

	def init_encoder_block(self, phylogeny):
		block = tf.keras.Sequential(name='feature_encoder')
		block.add(Encoder(phylogeny))
		return block

	def init_base_block1(self):
		block = tf.keras.Sequential(name='base')
		block.add(Flatten()) # (1000, )
		block.add(Dense(4096,  kernel_initializer=he_uniform))
		block.add(self._init_bn_layer())
		block.add(Activation('relu')) # (512, )
		block.add(Dense(2048,  kernel_initializer=he_uniform))
		block.add(self._init_bn_layer())
		block.add(Activation('relu')) # (512, )
		block.add(Dense(2048,  kernel_initializer=he_uniform))
		block.add(self._init_bn_layer())
		block.add(Activation('relu')) # (512, )
		block.add(Dense(1024,  kernel_initializer=he_uniform))
		block.add(self._init_bn_layer())
		block.add(Activation('relu')) # (512, )
		block.add(Dense(1024,  kernel_initializer=he_uniform))
		block.add(self._init_bn_layer())
		block.add(Activation('relu')) # (512, )
		return block


	def init_base_block(self):
		block = tf.keras.Sequential(name='base')
		block.add(Conv2D(128, kernel_size=(1, 3),  kernel_initializer=he_uniform))
		block.add(self._init_bn_layer())
		block.add(Activation('relu')) # (1000, 4, 64) -> 256000
		block.add(Conv2D(128, kernel_size=(1, 2),  kernel_initializer=he_uniform))
		block.add(self._init_bn_layer())
		block.add(Activation('relu')) # (1000, 3, 64) -> 192000
		block.add(Conv2D(128, kernel_size=(1, 2),  kernel_initializer=he_uniform))
		block.add(self._init_bn_layer())
		block.add(Activation('relu')) # (1000, 2, 128) -> 256000
		block.add(Conv2D(256, kernel_size=(1, 2),  kernel_initializer=he_uniform))
		block.add(self._init_bn_layer())
		block.add(Activation('relu')) # (1000, 1, 128) -> 128000
		block.add(Conv2D(2, kernel_size=(1, 1),  kernel_initializer=he_uniform))
		block.add(self._init_bn_layer())
		block.add(Activation('relu')) # (1000, 1, 1) -> 1000
		block.add(Flatten()) # (1000, )
		block.add(Dense(2048,  kernel_initializer=he_uniform))
		block.add(self._init_bn_layer())
		block.add(Activation('relu')) # (512, )
		return block

	def init_inter_block(self, index, name, n_units):
		k = index
		block = tf.keras.Sequential(name=name)
		block.add(Dense(self._get_n_units(n_units*8), name='l' + str(k) + '_inter_fc0',  kernel_initializer=he_uniform))
		block.add(Activation('relu'))
		#block.add(Dropout(0.5))
		block.add(Dense(self._get_n_units(n_units*4), name='l' + str(k) + '_inter_fc1',  kernel_initializer=he_uniform))
		block.add(Activation('relu'))
		#block.add(Dropout(0.5))
		block.add(Dense(self._get_n_units(n_units*4), name='l' + str(k) + '_inter_fc2',  kernel_initializer=he_uniform))
		block.add(Activation('relu'))
		#block.add(Dropout(0.5))
		return block

	def _init_integ_block(self, index, name, n_units):
		block = tf.keras.Sequential(name=name)
		k = index
		block.add(Dense(self._get_n_units(n_units*8), name='l' + str(k) + '_integ_fc0', kernel_initializer=he_uniform))
		#block.add(self._init_bn_layer())
		block.add(Activation('relu'))
		return block

	def init_output_block(self, index, name, n_units):
		#input_shape = (128 * 2 if index > 0 else 128, )
		block = tf.keras.Sequential(name=name)
		k = index
		#block.add(Dropout(0.7))
		block.add(Dense(self._get_n_units(n_units*4), name='l' + str(k) + '_output_fc0', kernel_initializer=he_uniform))
		block.add(Activation('relu'))
		block.add(Dense(self._get_n_units(n_units*4), name='l' + str(k) + '_output_fc1', kernel_initializer=he_uniform))
		block.add(Activation('relu'))
		block.add(Dense(self._get_n_units(n_units*4), name='l' + str(k) + '_output_fc2', kernel_initializer=he_uniform))
		block.add(Activation('relu'))
		#block.add(Dropout(0.5))
		block.add(Dense(self._get_n_units(n_units*4), name='l' + str(k) + '_output_fc3', kernel_initializer=he_uniform))
		block.add(Activation('relu'))
		#block.add(Dropout(0.7))
		block.add(Dense(n_units, name='l' + str(index)))
		return block

	def _get_dropout_rate(self, index, last_n_units, next_n_units):
		prop = (last_n_units - next_n_units) / last_n_units
		scale_0206 = lambda prop:  0.2 + (prop - 0.5) / 0.25 * 0.4
		prop_scaled = scale_0206(prop)
		print('_get_dropout_rate: ', prop_scaled)
		return prop_scaled

	def _init_bn_layer(self):
		return BatchNormalization(momentum=0.9)

	def init_post_proc_layer(self, name):
		def scale_output(x):
			total_contrib = tf.constant([[1]], dtype=tf.float32, shape=(1, 1))
			unknown_contrib = tf.subtract(total_contrib, tf.keras.backend.sum(x, keepdims=True, axis=1))
			contrib = tf.keras.backend.relu(tf.keras.backend.concatenate((x, unknown_contrib), axis=1))
			scaled_contrib = tf.divide(contrib, tf.keras.backend.sum(contrib, keepdims=True, axis=1))
			return scaled_contrib
		return Lambda(scale_output, name=name)

	def call(self, inputs, training=False):
		inputs = self.encoder(inputs)
		base = self.base(inputs, training=training if self.fine_tune == False else False)
		inter_logits = [self.spec_inters[i](base, training=training) for i in range(self.n_layers)]
		
		integ_logits = []
		
		for layer in range(self.n_layers):
			if layer == 0:
				integ_logits.append(self.spec_integs[layer](inter_logits[layer], training=training))
			else:
				integ_logits.append(self.spec_integs[layer](self.concat([integ_logits[layer-1],
																		 inter_logits[layer]]),
													  training=training))
		
		out_logits = [self.spec_outputs[i](integ_logits[i],
										   training=training)
					  for i in range(self.n_layers)]

		outputs = [self.spec_postprocs[i](out_logits[i], training=training)
				   for i in range(self.n_layers)]
		#outputs = out_logits
		
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
		integs_dir = self.__pthjoin(path, 'integs')
		outputs_dir = self.__pthjoin(path, 'outputs')
		for dir in [path, inters_dir, outputs_dir]:
			if not os.path.isdir(dir):
				os.mkdir(dir)
		#self.feature_mapper.save(self.__pthjoin(path, 'feature_mapper'))
		self.base.save(self.__pthjoin(path, 'base'), save_format='tf')
		for layer in range(self.n_layers):
			self.spec_inters[layer].save(self.__pthjoin(inters_dir, str(layer)), save_format='tf')
			self.spec_integs[layer].save(self.__pthjoin(integs_dir, str(layer)), save_format='tf')
			self.spec_outputs[layer].save(self.__pthjoin(outputs_dir, str(layer)), save_format='tf')

	def __restore_from(self, path):
		#mapper_dir = self.__pthjoin(path, 'feature_mapper')
		base_dir = self.__pthjoin(path, 'base')
		inters_dir = self.__pthjoin(path, 'inters')
		integs_dir = self.__pthjoin(path, 'integs')
		outputs_dir = self.__pthjoin(path, 'outputs')
		inter_dirs = [self.__pthjoin(inters_dir, i) for i in os.listdir(inters_dir)]
		integ_dirs = [self.__pthjoin(integs_dir, i) for i in os.listdir(integs_dir)]
		output_dirs = [self.__pthjoin(outputs_dir, i) for i in os.listdir(outputs_dir)]
		#self.feature_mapper = tf.keras.models.load_model(mapper_dir, custom_objects={'Mapper': Mapper})
		self.base = tf.keras.models.load_model(base_dir)
		self.spec_inters = [tf.keras.models.load_model(dir) for dir in inter_dirs]
		self.spec_integs = [tf.keras.models.load_model(dir) for dir in integ_dirs]
		self.spec_outputs = [tf.keras.models.load_model(dir) for dir in output_dirs]

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

class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())
