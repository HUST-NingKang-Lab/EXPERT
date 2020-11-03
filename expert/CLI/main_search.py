from expert.src.model import Model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf, pandas as pd, numpy as np
from expert.CLI.CLI_utils import find_pkg_resource


def search(cfg, args):
	# Basic configurations for GPU and CPU
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
	if args.gpu >= 0:
		gpus = tf.config.list_physical_devices('GPU')
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)

	# Read data
	X = pd.read_hdf(args.input, key='genus').T
	sampleIDs = X.index
	phylogeny = pd.read_csv(find_pkg_resource('resources/phylogeny.csv'), index_col=0)

	# Build EXPERT model
	model = Model(phylogeny=phylogeny, num_features=phylogeny.shape[0], restore_from=args.model)
	X = model.encoder(X.to_numpy()).numpy().reshape(X.shape[0], X.shape[1] * phylogeny.shape[1])
	X = model.standardize(X)
	model.build_estimator()

	# Calculate source contribution
	contrib_arrs = model.estimator.predict(X)
	labels = model.labels
	contrib_layers = {'layer-'+str(i+2): pd.DataFrame(contrib_arrs[i], index=sampleIDs, columns=labels[i+1] + ['Unknown'])
					  for i, key in enumerate(labels.keys())}
	for layer, contrib in contrib_layers.items():
		if not os.path.isdir(args.output):
			os.mkdir(args.output)
		contrib.to_csv(os.path.join(args.output, layer+'.csv'))
