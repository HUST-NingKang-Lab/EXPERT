from expert.src.utils import load_otlg, meta_from_dir, map_to_ontology
import pandas as pd
from tqdm import tqdm


def map(cfg, args):
	if args.from_dir:
		print('Generating table for metadata..')
		meta = meta_from_dir(args.input)
		print('Done')
		meta.to_csv(args.output)
	elif args.to_otlg:
		mapper = pd.read_csv(args.input)
		otlg = load_otlg(args.otlg)
		print('Mapping sources to microbiome ontology...')
		labels_by_level = map_to_ontology(mapper, otlg, True)
		print('The ontology contains {} layers.'.format(len(labels_by_level)))
		print('Saving labels for each ontology layer...')
		for level, labels in tqdm(labels_by_level.items()):
			# consider save lmax for using
			print(labels.sum())
			#labels.to_csv(args.o+'.csv')
			labels.to_hdf(args.output, key='l'+str(level), mode='a')
	else:
		raise ValueError('Please given one of the [from_dir, to_otlg]')
