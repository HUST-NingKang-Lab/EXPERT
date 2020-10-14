from expert.src.utils import samples_to_countmatrix, merge_countmatrices, scale_abundance
from expert.src.preprocessing import Transformer
from expert.CLI.CLI_utils import find_pkg_resource
import pandas as pd
from tqdm import tqdm
import numpy as np
import os


def convert(cfg, args):
	if not os.path.isdir(args.tmp):
		os.mkdir(args.tmp)
	print('running...')
	print('Reading and concatenating data, this could be slow if you have huge amount of data')
	with open(args.i, 'r') as f:
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
	tm = Transformer(tmp_path=find(cfg.get('DEFAULT', 'tmp')), phylogeny=find(cfg.get('DEFAULT', 'phylogeny')),
					 db_file=find(cfg.get('DEFAULT', 'db_file')))
	matrix_genus = tm._extract_layers(matrix, included_ranks=included_ranks)
	print('Saving results...')
	matrix_genus.to_hdf(args.o, key='genus', mode='a')
	print('Phylogeny is saved under `conf` you have specified.')
	tm.phylogeny.reset_index(drop=True).to_csv(tm.get_conf_savepath('phylogeny_by_transformer.csv'))