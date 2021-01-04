import argparse
import os
from configparser import ConfigParser
import pkg_resources

def set_seed(SEED):
	os.environ['PYTHONHASHSEED'] = str(SEED)
	from numpy.random import seed
	seed(SEED)
	from tensorflow.random import set_seed
	set_seed(SEED)

def get_CFG_reader():
	cfg = ConfigParser()
	assert pkg_resources.resource_exists('expert', 'resources/config.ini')
	cfg.read(pkg_resources.resource_filename('expert', 'resources/config.ini'))
	return cfg


def find_pkg_resource(path):
	if pkg_resources.resource_exists('expert', path):
		return pkg_resources.resource_filename('expert', path)
	else:
		raise FileNotFoundError('Resource {} not found, please check'.format(path))


def get_CLI_parser():
	modes = ['init', 'download', 'map', 'construct', 'convert', 'select', 'train', 'transfer', 'evaluate', 'search']
	# noinspection PyTypeChecker
	parser = argparse.ArgumentParser(
		description=('The program is designed to help you to transfer Ontology-aware Neural Network model '
					 'to other source tracking tasks.\n'
					 'Feel free to contact us if you have any question.\n'
					 'For more information, see Github. Thank you for using Ontology-aware neural network.'),
		formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument('mode', type=str, default='search', choices=modes,
						help='The work mode for expert program.')
	parser.add_argument('-i', '--input', type=str, default=None,
						help='The input file, see input format for each work mode.')
	parser.add_argument('-o', '--output', type=str, default=None,
						help='The output file, see output format for each work mode.')
	parser.add_argument('-c', '--config', type=str, default=None,
						help='The config.ini file.')
	parser.add_argument('-p', '--processors', type=int, default=1,
						help='The number of processors to use.')
	parser.add_argument('-t', '--otlg', type=str, default=None,
						help='The path to microbiome ontology.')
	parser.add_argument('-l', '--labels', type=str, default=None,
						help='The path to h5 file (storing labels for the input data).')
	parser.add_argument('-m', '--model', type=str, default=pkg_resources.resource_filename('expert', 'resources/general_model'),
						help='The path to expert model')
	parser.add_argument('-g', '--gpu', type=int, default=-1,
						help='-1: CPU only, 0: GPU0, 1: GPU1, ...')
	parser.add_argument('-s', '--val-split', type=float, default=0.1,
					   help='The fraction of validation samples.')
	parser.add_argument('-H', '--log', type=str, default=None,
					   help='The path to store training history of expert model.')
	parser.add_argument('--dropout-rate', type=float, default=0,
						  help='Set the dropout rate to prevent over-fitting.')
	parser.add_argument('--batch-size', type=int, default=32,
						help='Set the batch size for searching.')
	parser.add_argument('--seed', type=int, default=1,
						  help='Set the seed for randomized processes.')

	# ------------------------------------------------------------------------------------------------------------------
	construct = parser.add_argument_group(
		title='construct', description='Construct ontology using microbiome structure ".txt" file.\n' 
					'Input: microbiome structure ".txt" file. Output: Constructed microbiome ontology.')
	construct.add_argument('-silence', action='store_true', help='Work in silence mode (don\'t display ontology).')

	# ------------------------------------------------------------------------------------------------------------------
	map = parser.add_argument_group(
		title='map', description='`-from-dir`: Get mapper file from directory.\n'
								 'Input: The directory to generate mapper file, Output: mapper file.\n'
								 '`-to-otlg`: Map source environments to microbiome ontology.\n'
								 'Input: The mapper file, Output: The ontologically arranged labels.')
	map.add_argument('--from-dir', action='store_true',
					 help='Getting mapper file from directory.')
	map.add_argument('--to-otlg', action='store_true',
					 help='Map source environments to microbiome ontology.')

	# ------------------------------------------------------------------------------------------------------------------
	convert = parser.add_argument_group(
		title='convert', description='Convert input abundance data to countmatrix at Genus level and '
									 'generate phylogeny using taxonomic entries involved in the data.\n'
									 'Preparing for feature selection\n'
									 'Input: the input data, Output: RRDM at Genus level')
	convert.add_argument('--in-cm', action='store_true',
						help="Whether to use the countmatrix as the input format.")
	convert.add_argument('--db_file', type=str, default=os.path.join(os.path.expanduser('~'), '.etetoolkit/taxa.sqlite'),
						 help='Local NCBI taxonomy database file, must be in sqlite format.')

	# ------------------------------------------------------------------------------------------------------------------
	select = parser.add_argument_group(
		title='select', description='Select features above the threshold. Variance and importance are '
									'calculated using Pandas and RandomForestRegressor, respectively.\n'
									'Input: countmatrix generated by `expert convert`, '
									'Output: selected features and phylogeny (tmp).')
	select.add_argument('--use-var', action='store_true',
						help='Filter features using a selected phylogeny.')
	select.add_argument('--use-rf', action='store_true',
						help="Whether to use the randomForest when performing selection.")
	select.add_argument('-C', '--coef', type=float, default=1e-3,
						help='The coefficient C in `Threshold = C * mean(stat)`.')

	# ------------------------------------------------------------------------------------------------------------------
	train = parser.add_argument_group(
		title='train', description='Train expert model, the microbiome ontology and properly labeled data '
								   'must be provided.\n'
								   'Input: samples, in pandas h5 format, output: expert model')

	# ------------------------------------------------------------------------------------------------------------------
	transfer = parser.add_argument_group(
		title='transfer', description='Transfer expert model to fit in a new ontology, The microbiome ontology '
									  'and properly labeled data must be provided.\n'
									  'use `-model` option to indicate a customized base model.\n'
									  'Input: samples, in pandas h5 format, output: expert model')
	transfer.add_argument('--update-statistics', action='store_true',
						  help='Normalize input samples using mean and std calculated from itself.')
	transfer.add_argument('--finetune', action='store_true',
						  help='Finetune the model during fast adaptation.')

	# ------------------------------------------------------------------------------------------------------------------
	evaluate = parser.add_argument_group(
		title='evaluate', description='Evaluate the expert model, properly labeled data must be provided.\n'
									  'use `-model` option to indicate a customized model.\n'
									  'Input: search results, output: evaluation report')
	evaluate.add_argument('-T', type=int, default=100,
						  help='The number of thresholds for evaluation.')
	evaluate.add_argument('-S', type=int, default=0,
						  help='The threshold when averaging metrics of each biome source with in each ontology layer')

	# ------------------------------------------------------------------------------------------------------------------
	search = parser.add_argument_group(
		title='search', description='Search for source environments of your microbial samples using expert model.\n')
	search.add_argument('--ofmt', type=str, default=None,
						help='The output format.')
	search.add_argument('--measure-unknown', action='store_true',
						help='Measure the contribution from unknown source.')
	return parser