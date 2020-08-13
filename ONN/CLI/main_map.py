from ONN.src.utils import load_ontology, meta_from_dir, map_to_ontology
import pandas as pd
from tqdm import tqdm


def map(args):
	if args.from_dir:
		meta = meta_from_dir(args.i)
		meta.to_csv(args.o)
	elif args.to_otlg:
		mapper = pd.read_csv(args.i)
		otlg = load_ontology(args.otlg)
		labels_by_level = map_to_ontology(mapper, otlg, args.unk)
		print('The ontology contains {} layers.'.format(len(labels_by_level)))
		print('Saving labels for each ontology layer.')
		for level, labels in tqdm(labels_by_level.items()):
			# consider save lmax for using
			print(labels.sum())
			labels.to_csv(args.o+'.csv')
			labels.to_hdf(args.o, key='l'+str(level), mode='a')
	else:
		raise ValueError('Please given one of the [from_dir, to_otlg]')
