from ONN.src.utils import samples_to_countmatrix, merge_countmatrices, scale_abundance, load_ontology
from ONN.src.preprocessing import Transformer
import pandas as pd
from tqdm import tqdm
import numpy as np
import os


def convert(args):
	if not os.path.isdir(args.conf):
		os.mkdir(args.conf)
	print('running...')
	print('Reading and concatenating data, this could be slow if you have huge amount of data')
	with open(args.i, 'r') as f:
		input_files = f.read().splitlines()
	if args.in_cm:
		sub_matrices = map(lambda x: pd.read_csv(x, sep='\t'), tqdm(input_files))
		matrix = merge_countmatrices(sub_matrices)
	else:
		samples = map(lambda x: pd.read_csv(x, sep='\t', header=1), tqdm(input_files))
		matrix = samples_to_countmatrix(samples)

	included_ranks = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus']
	matrix = scale_abundance(matrix) # Calculate relative abundance

	if args.gen_phylo:
		print('Optional argument `phylo` is unspecified, '
			  'the program will generate countmatrix at genus level. '
			  'Phylogeny will be automatically constructed '
			  'using all taxonomic entries data involved. '
			  'You can select features use that.')
		tm = Transformer(conf_path=args.conf, phylogeny=args.phylo, db_file=args.db)
		matrix_genus = tm._extract_layers(matrix, included_ranks=included_ranks)
		print('Saving results...')
		matrix_genus.to_hdf(args.o, key='genus', mode='a')
		print('Phylogeny is saved under `conf` you have specified.')
		tm.phylogeny.reset_index(drop=True).to_csv(tm.get_conf_savepath('phylogeny_by_transformer.csv'))
	else:
		phylo = pd.read_csv(args.phylo, index_col=0)
		tm = Transformer(conf_path=args.conf, phylogeny=phylo, db_file=args.db)
		matrix_by_rank = tm._extract_layers(matrix, included_ranks=included_ranks)
		print('Saving results...')
		for rank, matrix in matrix_by_rank.items():
			matrix.to_hdf(args.o, key=rank, mode='a')
