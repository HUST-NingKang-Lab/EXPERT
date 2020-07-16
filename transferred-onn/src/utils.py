from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Concatenate
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow_hub as hub
from tqdm import tqdm
import os
import argparse
import sqlite3
from functools import reduce
import re
from collections import OrderedDict
from livingTree import SuperTree


def load_extractor(self, model_path, input_shape):
	if model_path == None:
		model_path = 'transferred-onn/explore-NB/MNIST/feature-vector/'
	model = hub.KerasLayer(model_path, input_shape=input_shape)
	return model

def str_sum(iterable):
	return reduce(lambda x, y: x + y, iterable)


def get_argparser():
	return None


def get_model():
	return None


def get_extractor(model, begin_layer, end_layer):
	return None


def save_extractor(extractor, foldername):
	pass


def load_extractor(foldername):
	return None


class Parser(object):

	def __init__(self):
		pass

	def parse_ontology(self, ontology: SuperTree):
		"""
		get output shape and label from ontology
		:param ontology: ontology arranged in livingTree object
		:return:
		"""
		ids = ontology.get_ids_by_level()
		shapes = {layer: (len(id_), 1) for layer, id_ in ids.items()}
		return ids, shapes

	def parse_otus(self, otus):
		"""

		:param otus:
		:return:
		"""
		return None

	def parse_biom(self, biom):
		"""

		:param biom:
		:return:
		"""
		return None

	def parse_tsvs(self, tsvs):
		"""

		:param tsvs:
		:return:
		"""
		return None

	def parse_args(self):
		return None



