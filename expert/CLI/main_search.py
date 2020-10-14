from expert.src.model import Model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf, pandas as pd, numpy as np
from expert.CLI.CLI_utils import find_pkg_resource


def search(cfg, args):
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
	if args.gpu >= 0:
		gpus = tf.config.list_physical_devices('GPU')
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)

	X = pd.read_hdf(args.i, key='genus').T
	sampleIDs = X.index
	phylogeny = pd.read_csv(find_pkg_resource(cfg.get('DEFALUT', 'phylo')), index_col=0)
	model = Model(phylogeny=phylogeny, num_features=phylogeny.shape[0], restore_from=args.model)
	# need to fix here
	X = model.encoder(X.to_numpy()).numpy().reshape(X.shape[0], X.shape[1] * phylogeny.shape[1])
	X_mean = np.load(os.path.join(find_pkg_resource(cfg.get('DEFALUT', 'tmp')), 'mean_f.for.X_train.npy'))
	X_std = np.load(os.path.join(find_pkg_resource(cfg.get('DEFALUT', 'tmp')), 'std_f.for.X_train.npy'))
	X = (X - X_mean) / X_std
	model.build_estimator()
	contrib_arrs = model.estimator.predict(X)
	labels = model.labels
	contrib_layers = {'layer-'+str(i+1): pd.DataFrame(contrib_arrs[i], index=sampleIDs, columns=labels[i+1] + ['Unknown'])
					  for i, key in enumerate(labels.keys())}
	for layer, contrib in contrib_layers.items():
		if not os.path.isdir(args.o):
			os.mkdir(args.o)
		contrib.to_csv(os.path.join(args.o, layer+'.csv'))
