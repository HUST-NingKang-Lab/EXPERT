import argparse
import os


def get_CLI_parser():
	# generate a hidden folder for expert, to store files
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
	parser.add_argument('-i', type=str, default=None,
						help='The input file, see input format for each work mode.')
	parser.add_argument('-o', type=str, default=None,
						help='The output file, see output format for each work mode.')
	parser.add_argument('-cfg', type=str, default=os.path.join(os.path.expanduser('~'), '.expert.ini'),
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
	parser.add_argument('-gpu', type=int, default=-1,
						help='-1: CPU only, 0: GPU0, 1: GPU1, ...')

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
									'Input: countmatrix generated by `expert convert`, '
									'Output: selected features and phylogeny (tmp).')
	select.add_argument('-filter-only', action='store_true',
						help='Filter features using a selected phylogeny.')
	select.add_argument('-use-rf', action='store_true',
						help="Whether to use the randomForest when performing selection.")
	select.add_argument('-C', type=float, default=1e-3,
						help='The coefficient C in `Threshold = C * mean(stat)`.')

	# ------------------------------------------------------------------------------------------------------------------
	train = parser.add_argument_group(
		title='train', description='Training expert model, the microbiome ontology and properly labeled data '
								   'must be provided.\n'
								   'Input: samples, in pandas h5 format, output: expert model')
	train.add_argument('-split-idx', type=int, default=None,
					   help='The index to split training and validation samples.')
	train.add_argument('-end-idx', type=int, default=None,
					   help='The index to split validation and testing samples.')
	train.add_argument('-log', type=str, default=None,
					   help='The path to store training history of expert model.')

	# ------------------------------------------------------------------------------------------------------------------
	transfer = parser.add_argument_group(
		title='transfer', description='Transferring expert model to fit in a new ontology, The microbiome ontology '
									  'and properly labeled data must be provided.\n')
	transfer.add_argument('-base', type=str, default=None,
						  help='The path to base feature extractor model.')

	# ------------------------------------------------------------------------------------------------------------------
	search = parser.add_argument_group(
		title='search', description='Searching for source environments of your microbial samples using expert model.\n')
	search.add_argument('-model', type=str, default=None,
						help='The path to expert model to search against.')
	search.add_argument('-ofmt', type=str, default=None,
						help='The output format.')
	return parser