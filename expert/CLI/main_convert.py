from expert.src.utils import samples_to_countmatrix, merge_countmatrices, scale_abundance
from expert.src.preprocessing import Transformer
from expert.CLI.CLI_utils import find_pkg_resource
import pandas as pd
from tqdm import tqdm
import numpy as np


def convert(cfg, args):
	print('running...')
	print('Reading and concatenating data, this could be slow if you have huge amount of data')
	db = args.db_file
	print('db file:', db)
	with open(args.input, 'r') as f:
		input_files = f.read().splitlines()
	if args.in_cm:
		sub_matrices = map(lambda x: pd.read_csv(x, sep='\t', index_col=0), tqdm(input_files))
		matrix = merge_countmatrices(sub_matrices)
	else:
		samples = map(lambda x: pd.read_csv(x, sep='\t', header=1), tqdm(input_files))
		matrix = samples_to_countmatrix(samples)

	included_ranks = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus']
	print(matrix.describe(percentiles=[]))
	matrix = matrix.astype(np.float32)
	tm = Transformer(phylogeny=pd.read_csv(find_pkg_resource('resources/phylogeny.csv'), index_col=0),  db_file=db)
	matrix_genus = tm._extract_layers(matrix, included_ranks=included_ranks)
	print('Normalizing results...')
	matrix_genus = matrix_genus.div(matrix_genus.sum(axis=0) + 1e-8)
	print(matrix_genus.describe(percentiles=[]))
	print('Total NaNs:', matrix_genus.isna().sum().sum())
	print('Saving results...')
	matrix_genus.to_hdf(args.output, key='genus', mode='a')
