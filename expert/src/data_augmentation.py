from sklearn.utils.class_weight import compute_sample_weight
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


class Mixer(object):
	# 给定 F，C，每次抽样决定用哪些样本
	def __init__(self, F, C_original):
		self.F = tf.constant(F.astype(np.float32), dtype=tf.float32) # num_samples * num_features
		self.n_f = F.shape[1]
		self.C_original = K.concatenate(tf.constant(C_original), axis=1)
		self.n_c = [C_tmp.shape[1] for C_tmp in C_original]
		self.n_layers = len(self.n_c)
		self.S = K.concatenate( [self.F, self.C_original], axis=1)
		self.n_samples = self.S.shape[0]
		self.overall_distribution = compute_sample_weight('balanced', self.C_original.numpy())
		self.div = tf.divide

	def generate_mixture(self, n_mixtures=1024*10**3, seed=0):
		np.random.seed(seed)
		C_pre = tf.constant(np.random.rand(n_mixtures, self.n_samples).astype(np.float32))
		C = self.div(C_pre, K.sum(C_pre, axis=1, keepdims=True))
		S_hat = K.dot(C, self.S)
		return S_hat[:, 0:self.n_f], [S_hat[:, 0:self.n_c[layer]] for layer in range(self.n_layers)]

