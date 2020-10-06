from expert.src.model import Model
import tensorflow as tf, pandas as pd, os


def search(args):
	if args.gpu >= 0:
		os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
		os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
		gpus = tf.config.list_physical_devices('GPU')
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
	else:
		os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
		os.environ["CUDA_VISIBLE_DEVICES"] = ''
	X = pd.read_hdf(args.i, key='genus').T
	phylogeny = pd.read_csv(args.phylo, index_col=0)
	model = Model(phylogeny=phylogeny, num_features=phylogeny.shape[0], restore_from=args.model)
	model.build_estimator()
	contrib_arrs = model.estimator.predict(X)
	labels = model.labels
	contrib_layers = {'layer-'+str(i+1): pd.DataFrame(contrib_arrs[i], index=X.index, columns=labels[i])
					  for i in labels.keys()}
	for layer, contrib in contrib_layers.items():
		contrib.to_csv(os.path.join(args.o, layer+'.csv'))