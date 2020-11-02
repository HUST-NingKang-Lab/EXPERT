import pandas as pd
import numpy as np
from functools import reduce
from livingTree import SuperTree
import os
from tqdm import tqdm
from collections import OrderedDict
from pandas.io.json._normalize import nested_to_record


def get_dmax(path):
	with pd.HDFStore(path) as hdf:
		dmax = len(hdf.keys())
		#hdf.close()
		return dmax


def zero_weight_unk(y, sample_weight):
	zero_weight_idx = (y['Unknown'] == 1).to_numpy()
	sample_weight[zero_weight_idx] = 0
	sample_weight = sample_weight / (1e-8 + 1 - zero_weight_idx.sum() / zero_weight_idx.shape[0])
	return sample_weight


def transfer_weights(base_model, init_model, reuse_levels):
	init_model.statistics = base_model.statistics
	init_model.base = base_model.base
	init_model.base.trainable = False
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


def generate_unk(df):
	df['Unknown'] = 1 - df.sum(axis=1)
	return df


def read_genus_abu(path):
	genus_abu = pd.read_hdf(path, key='genus').T
	idx = np.arange(genus_abu.shape[0])
	np.random.seed(0)
	np.random.shuffle(idx)
	genus_abu = genus_abu.iloc[idx, :]
	return genus_abu, idx


def read_labels(path, shuffle_idx, dmax):
	# unk should be generated in map op, not here remember to fix
	labels = [generate_unk(pd.read_hdf(path, key='l'+str(layer))).iloc[shuffle_idx, :] for layer in range(dmax)]
	return labels[1:]


def load_otlg(path):
	otlg = SuperTree().from_pickle(path)
	return otlg


def parse_otlg(ontology):
	labels = OrderedDict([(layer, label) for layer, label in ontology.get_ids_by_level().items()
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



