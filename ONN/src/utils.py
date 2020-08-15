import pandas as pd
import numpy as np
import argparse
from functools import reduce
from livingTree import SuperTree
import os
from tqdm import tqdm
from collections import OrderedDict


def read_matrices(path, split_idx, end_idx):
	include_ranks = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus']
	matrices = np.array([pd.read_hdf(path, key=rank).T for rank in include_ranks])
	matrices = matrices.swapaxes(0, 1).swapaxes(1, 2)
	matrices = np.expand_dims(matrices, axis=3)
	idx = np.arange(matrices.shape[0])
	np.random.seed(0)
	np.random.shuffle(idx)
	matrices = matrices[idx]
	return matrices[0:split_idx], matrices[split_idx:end_idx], idx

def generate_unk(df):
	df['Unknown'] = 1 - df.sum(axis=1)
	return df

def read_labels(path, shuffle_idx, split_idx, end_idx, dmax):
	# unk should be generated in map op, not here remember to fix
	labels = [generate_unk(pd.read_hdf(path, key='l'+str(layer))).iloc[shuffle_idx, :] for layer in range(dmax)]
	return [label[0:split_idx] for label in labels[1:]], \
		   [label[split_idx:end_idx] for label in labels[1:]] # except for layer 0 -> root 

def parse_otlg(path):
	otlg = SuperTree().from_pickle(path)
	labels = OrderedDict([(layer, label) for layer, label in otlg.get_ids_by_level().items()
						  if layer > 0])
	layer_units = [len(label) for layer, label in labels.items()]
	return labels, layer_units

def str_sum(iterable):
	return reduce(lambda x, y: x + y, iterable)

def samples_to_countmatrix(tsvs):
	keep_tax_abu = lambda x: x.loc[x['taxonomy'].str.contains('k__'),
								   x.columns[1:3]].groupby(by='taxonomy').sum()
	tsvs_keep = map(keep_tax_abu, tsvs)
	#matrix = reduce(lambda x, y: pd.merge(left=x, right=y, on='taxonomy', how='outer'), tsvs_keep)
	matrix = pd.concat(tsvs_keep, axis=1, join='outer') # add progress bar
	return matrix.fillna(0)

def merge_countmatrices(sub_matrices):
	clean = lambda x: x.loc[x.index.to_series().str.contains('k__'), :]
	matrix = reduce(lambda x, y: pd.merge(left=x, right=y, left_index=True, right_index=True, how='outer'),
					sub_matrices)
	return matrix.fillna(0)

def meta_from_dir(dir_):
	biomes = [os.path.join(dir_, biome) for biome in os.listdir(dir_)]
	samples = [os.path.join(biome, sample) for biome in biomes for sample in os.listdir(biome)]
	meta = pd.DataFrame(pd.Series(samples).apply(lambda x: x.split(os.sep)[-2:]).tolist(),
						columns=['Env', 'SampleID'])
	meta['SampleID'] = meta['SampleID'].str.extract(pat='([SED]RR.[0-9]{1,})')
	return meta

def map_to_ontology(mapper, otlg, unk):
	ids_by_level = {level: pd.Series(ids) for level, ids in otlg.get_ids_by_level().items()}
	str_cumsum = lambda x: [':'.join(x[0:i]) for i in range(1, len(x)+1)]
	layers = pd.DataFrame(mapper['Env'].str.split(':').apply(str_cumsum).tolist(),
						  index=mapper['SampleID'])
	#labels_by_level = {level: pd.DataFrame([], columns=ids) for level, ids in ids_by_level.items()}
	labels_by_level = {}
	for level, labels in tqdm(ids_by_level.items()):
		labels_arr = labels.to_numpy()
		labels_rowv = labels_arr.reshape(1, labels_arr.shape[0])
		env_colv = layers[level].to_numpy().reshape(layers.shape[0], 1)
		labels_df = pd.DataFrame((env_colv == labels_rowv).astype(int), columns=labels_rowv[0],
								 index=layers.index)
		#print(labels_df)
		if unk: # Fix here
			labels_df['Unknown'] =  1 - labels_df.sum(axis=1)
		labels_by_level[level] = labels_df
	return labels_by_level

def scale_abundance(matrix):
	return matrix / matrix.sum()

def get_CLI_parser():
	modes = ['map', 'construct','convert', 'select', 'train', 'transfer', 'search']
	parser = argparse.ArgumentParser(description=('The program is designed to help you '
												 'to transfer '
												 'Ontology-aware Neural Network model '
												 'to other source tracking tasks.\n'
												 'Feel free to contact us if you '
												 'have any question.\n'
												 'For more information, see Github. '
												 'Thank you for using Ontology-aware '
												 'neural network.'),
									 formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument('mode', type=str, default='search', choices=modes,
						help='The work mode for ONN program, choose from '
							 '[map, construct, convert, select, transfer, search].')
	parser.add_argument('-i', type=str, default=None,
						help='The input file, see input format for each work mode.')
	parser.add_argument('-o', type=str, default=None,
						help='The output file, see output format for each work mode.')
	parser.add_argument('-conf', type=str, default=None, help="The path to save temperature files.")
	parser.add_argument('-p', type=int, default=1, help='The number of processors to use.')
	parser.add_argument('-otlg', type=str, default=None, help='The path to microbiome ontology.')
	parser.add_argument('-labels', type=str, default=None, help='The path to npz file '
															   '(storing labels for the input data).')
	parser.add_argument('-dmax', type=int, help='The max depth of the ontology.')
	construct = parser.add_argument_group('construct',
										  'Constructing ontology using microbiome structure '
										  '".txt" file.\n'
										  'Input: microbiome structure ".txt" file. '
										  'Output: Constructed microbiome ontology.')
	map = parser.add_argument_group('map', '`-from-dir`: Getting mapper file from directory.\n'
										   'Input: The directory to generate mapper file, '
										   'Output: mapper file.\n'
										   '`-to-otlg`: Mapping source environments to '
										   'microbiome ontology.\n'
										   'Input: The mapper file, '
										   'Output: The ontologically arranged labels for all samples.')
	convert = parser.add_argument_group('convert', '1. With `-phylo` specified, The program will convert'
												   ' abundance data to hierarchical RRDM.\n'
												   'Input: the input data, Output: RRDM.\n'
												   '2. With `-gen-phylo`, THe program will generate '
												   'phylogeny using taxonomic entries involved in the data.\n'
												   'Warning: Only the countmatrix at Genus level will be converted '
												   'for you to select features.\n'
												   'Input: the input data, Output: RRDM at Genus level')
	select = parser.add_argument_group('select', 'Selecting `-top` n important features using feature '
												 'importance calculated by RandomForestRegressor.\n'
												 'Input: phylogeny generated by `... convert -gen-phylo`, '
												 'Output: selected phylogeny.')
	train = parser.add_argument_group('train', 'Training ONN model, '
												'the microbiome ontology and properly labeled '
												'data must be provided.\n'
											   'Input: samples, in pandas h5 format, output: ONN model')
	transfer = parser.add_argument_group('transfer', 'Transferring ONN model to fit in a new ontology, '
													 'The microbiome ontology and properly labeled '
													 'data must be provided.\n')
	search = parser.add_argument_group('search', 'Searching for source environments of your microbial '
												 'samples using ONN model.\n')
	construct.add_argument('-show', type=int, default=1, help='Whether to print the ontology to stdout.')
	map.add_argument('-from-dir', action='store_true', help='Getting mapper file from directory.')
	map.add_argument('-to-otlg', action='store_true',
					 help='Mapping source environments to microbiome ontology.')
	map.add_argument('-unk', action='store_true',
					 help='Whether to include Unknown source when generating labels.')
	convert.add_argument('-gen-phylo', action='store_true',
						 help="Generating phylogeny using taxonomic entries involved in the data.")
	convert.add_argument('-phylo', type=str, default=None,
						help="The phylogeny tree to use, in tsv format.")
	convert.add_argument('-db', type=str, default='/root/.etetoolkit/taxa.sqlite',
						help="The NCBI taxonomy database file to use, in sqlite format.")
	convert.add_argument('-in-cm', action='store_true',
						help="Whether to use the countmatrix as the input format.")
	select.add_argument('-top', type=int, default=1000,
						help='The number of top important features to select.')
	select.add_argument('-cm', type=str, default=None,
						help='The path to countmatrix, will use that to compute importance.')
	train.add_argument('-split-idx', type=int, default=None,
					   help='The index to split training and validation samples.')
	train.add_argument('-end-idx', type=int, default=None,
					   help='The index to split validation and testing samples.')
	train.add_argument('-log', type=str, default=None,
					   help='The path to store training history of ONN model.')
	transfer.add_argument('-base', type=str, default=None,
						  help='The path to base feature extractor model.')
	search.add_argument('-model', type=str, default=None,
						help='The path to ONN model to search against.')
	search.add_argument('-ofmt', type=str, default=None,
						help='The output format.')
	return parser

def load_ontology(path):
	otlg = SuperTree()
	otlg = otlg.from_pickle(path)
	return otlg

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

	def parse_args(self):
		return None



