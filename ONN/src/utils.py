import pandas as pd
import numpy as np
import argparse
from functools import reduce
from livingTree import SuperTree
import os
from tqdm import tqdm
from collections import OrderedDict
from pandas.io.json._normalize import nested_to_record
from ONN.src.model import Model


def zero_weight_unk(y, sample_weight):
	zero_weight_idx = y['Unknown'] == 1
	sample_weight[zero_weight_idx] = 0
	return sample_weight

def transfer_weights(base_model: Model, init_model: Model, new_mapper, reuse_levels):
	init_model.base = base_model.base
	init_model.base.trainable = False
	if not new_mapper:
		init_model.feature_mapper = base_model.feature_mapper
		init_model.feature_mapper.trainable = False
	for layer, reuse_level in enumerate(reuse_levels):
		use_level = int(reuse_level)
		if use_level == 0:
			pass
		elif use_level == 1:
			init_model.spec_inters[layer] = init_model.spec_inters[layer]
			init_model.spec_inters[layer].trainable = False
		elif use_level == 2:
			init_model.spec_inters[layer] = init_model.spec_inters[layer]
			init_model.spec_integs[layer] = init_model.spec_integs[layer]
			init_model.spec_inters[layer].trainable = False
			init_model.spec_integs[layer].trainable = False
		elif use_level == 3:
			init_model.spec_inters[layer] = init_model.spec_inters[layer]
			init_model.spec_integs[layer] = init_model.spec_integs[layer]
			init_model.spec_outputs[layer] = init_model.spec_outputs[layer]
			init_model.spec_inters[layer].trainable = False
			init_model.spec_integs[layer].trainable = False
			init_model.spec_outputs[layer].trainable = False
		else:
			raise ValueError('The maximum reuse_level is 3, check your config.')
	return init_model

def read_input_list(path):
	with open(path, 'r') as f:
		return f.read().splitlines()

def runid_from_taxassign(path):
	return pd.read_csv(path, sep='\t', nrows=1, index_col=0).columns.tolist()

def format_sample_info(sample_and_status):
	sample = sample_and_status[0]
	status = sample_and_status[1:3]
	metadata = pd.DataFrame(nested_to_record(sample)['attributes.sample-metadata'])
	metadata['key'] = metadata['key'] + metadata['unit'].apply(lambda x: '(unit: {})'.format(x) if x else '')
	metadata = dict(zip(metadata['key'], metadata['value']))
	metadata['Sample ID'] = sample['id']
	metadata['Sample Name'] = sample['attributes']['sample-name']
	metadata['Sample Description'] = sample['attributes']['sample-desc']
	metadata['URL status'] = ','.join(status)
	#print(metadata)
	return metadata

def read_matrices(path, split_idx, end_idx):
	include_ranks = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus']
	matrices = np.array([pd.read_hdf(path, key=rank).T for rank in include_ranks])
	matrices = matrices.swapaxes(0, 1).swapaxes(1, 2)
	#matrices = np.expand_dims(matrices, axis=3)
	idx = np.arange(matrices.shape[0])
	np.random.seed(0)
	np.random.shuffle(idx)
	matrices = matrices[idx]
	return matrices[0:split_idx], matrices[split_idx:end_idx], idx

def generate_unk(df):
	df['Unknown'] = 1 - df.sum(axis=1)
	return df

def read_genus_abu(path, split_idx, end_idx):
	genus_abu = pd.read_hdf(path, key='genus').T
	idx = np.arange(genus_abu.shape[0])
	np.random.seed(0)
	np.random.shuffle(idx)
	genus_abu = genus_abu.iloc[idx, :]
	return genus_abu.iloc[0:split_idx, :], genus_abu.iloc[split_idx:end_idx, :], idx

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
	# unknown abundance data -> convert -> select (filter) -> extract
	# training data -> convert -> select -> mix -> extract
	modes = ['init', 'download', 'map', 'construct', 'convert', 'select', 'train', 'transfer', 'search']
	# noinspection PyTypeChecker
	parser = argparse.ArgumentParser(
		description=('The program is designed to help you to transfer Ontology-aware Neural Network model '
					 'to other source tracking tasks.\n'
					 'Feel free to contact us if you have any question.\n'
					 'For more information, see Github. Thank you for using Ontology-aware neural network.'),
		formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument('mode', type=str, default='search', choices=modes,
						help='The work mode for ONN program.')
	parser.add_argument('-i', type=str, default=None,
						help='The input file, see input format for each work mode.')
	parser.add_argument('-o', type=str, default=None,
						help='The output file, see output format for each work mode.')
	parser.add_argument('-cfg', type=str, default=None,
						help='The config.ini file.')
	parser.add_argument('-tmp', type=str, default=None,
						help="The path to save temperature files.")
	parser.add_argument('-p', type=int, default=1,
						help='The number of processors to use.')
	parser.add_argument('-otlg', type=str, default=None,
						help='The path to microbiome ontology.')
	parser.add_argument('-labels', type=str, default=None,
						help='The path to npz file (storing labels for the input data).')
	parser.add_argument('-phylo', type=str, default=None,
						help="The phylogeny tree to use, in tsv format.")
	parser.add_argument('-dmax', type=int,
						help='The max depth of the ontology.')

	# ------------------------------------------------------------------------------------------------------------------
	construct = parser.add_argument_group(
		title='construct', description='Constructing ontology using microbiome structure ".txt" file.\n' 
					'Input: microbiome structure ".txt" file. Output: Constructed microbiome ontology.')
	construct.add_argument('-show', action='store_true', help='Printing the ontology to stdout.')

	# ------------------------------------------------------------------------------------------------------------------
	map = parser.add_argument_group(
		title='map', description='`-from-dir`: Getting mapper file from directory.\n'
								 'Input: The directory to generate mapper file, Output: mapper file.\n'
								 '`-to-otlg`: Mapping source environments to microbiome ontology.\n'
								 'Input: The mapper file, Output: The ontologically arranged labels.')
	map.add_argument('-from-dir', action='store_true', help='Getting mapper file from directory.')
	map.add_argument('-to-otlg', action='store_true',
					 help='Mapping source environments to microbiome ontology.')
	map.add_argument('-unk', action='store_true',
					 help='Whether to include Unknown source when generating labels.')

	# ------------------------------------------------------------------------------------------------------------------
	convert = parser.add_argument_group(
		title='convert', description='Converting input abundance data to countmatrix at Genus level and '
									 'generating phylogeny using taxonomic entries involved in the data.\n'
									 'Preparing for feature selection\n'
									 'Input: the input data, Output: RRDM at Genus level')
	convert.add_argument('-db', type=str, default='/root/.etetoolkit/taxa.sqlite',
						help="The NCBI taxonomy database file to use, in sqlite format.")
	convert.add_argument('-in-cm', action='store_true',
						help="Whether to use the countmatrix as the input format.")

	# ------------------------------------------------------------------------------------------------------------------
	select = parser.add_argument_group(
		title='select', description='Selecting features above the threshold. Variance and importance are '
									'calculated using Pandas and RandomForestRegressor, respectively.\n'
									'Input: countmatrix generated by `ONN convert`, '
									'Output: selected features and phylogeny (tmp).')
	select.add_argument('-filter-only', action='store_true',
						help='Filter features using a selected phylogeny.')
	select.add_argument('-use-rf', action='store_true',
						help="Whether to use the randomForest when performing selection.")
	select.add_argument('-C', type=float, default=1e-3,
						help='The coefficient C in `Threshold = C * mean(stat)`.')

	# ------------------------------------------------------------------------------------------------------------------
	train = parser.add_argument_group(
		title='train', description='Training ONN model, the microbiome ontology and properly labeled data '
								   'must be provided.\n'
								   'Input: samples, in pandas h5 format, output: ONN model')
	train.add_argument('-split-idx', type=int, default=None,
					   help='The index to split training and validation samples.')
	train.add_argument('-end-idx', type=int, default=None,
					   help='The index to split validation and testing samples.')
	train.add_argument('-log', type=str, default=None,
					   help='The path to store training history of ONN model.')

	# ------------------------------------------------------------------------------------------------------------------
	transfer = parser.add_argument_group(
		title='transfer', description='Transferring ONN model to fit in a new ontology, The microbiome ontology '
									  'and properly labeled data must be provided.\n')
	transfer.add_argument('-base', type=str, default=None,
						  help='The path to base feature extractor model.')

	# ------------------------------------------------------------------------------------------------------------------
	search = parser.add_argument_group(
		title='search', description='Searching for source environments of your microbial samples using ONN model.\n')
	search.add_argument('-model', type=str, default=None,
						help='The path to ONN model to search against.')
	search.add_argument('-ofmt', type=str, default=None,
						help='The output format.')
	return parser


def extract_RRDM(F, ):
	'''

	:param F:
	:return:
		else:
		phylo = pd.read_csv(args.phylo, index_col=0)
		tm = Transformer(conf_path=args.tmp, phylogeny=phylo, db_file=args.db)
		matrix_by_rank = tm._extract_layers(matrix, included_ranks=included_ranks)
		print('Saving results...')
		for rank, matrix in matrix_by_rank.items():
			matrix.to_hdf(args.o, key=rank, mode='a')
	'''
	pass

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



