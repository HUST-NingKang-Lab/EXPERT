from expert.src.utils import samples_to_countmatrix, merge_countmatrices, scale_abundance
from expert.src.preprocessing import Transformer
from expert.CLI.CLI_utils import find_pkg_resource
import pandas as pd
from tqdm import tqdm
import numpy as np
import os


def convert(cfg, args):
	print('running...')
	print('Reading and concatenating data, this could be slow if you have huge amount of data')
	db = os.path.join(os.path.expanduser('~'), cfg.get('DEFAULT', 'db_file').lstrip('~/'))
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
	matrix = matrix.astype(np.uint64)
	find = find_pkg_resource
	tm = Transformer(tmp_path=find(cfg.get('DEFAULT', 'tmp')), phylogeny=pd.read_csv(find(cfg.get('DEFAULT', 'phylo')), index_col=0),
					 db_file=db)
	matrix_genus = tm._extract_layers(matrix, included_ranks=included_ranks)
	print('Saving results...')
	matrix_genus.to_hdf(args.output, key='genus', mode='a')
