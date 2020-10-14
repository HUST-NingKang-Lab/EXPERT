from expert.src.model import Model
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping
from expert.src.utils import read_genus_abu, read_labels, load_otlg, zero_weight_unk, parse_otlg, get_dmax
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Nadam
from tensorflow.keras.metrics import AUC, BinaryAccuracy
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
import tensorflow as tf, numpy as np, pandas as pd, os
import tensorflow.keras.backend as K
from expert.CLI.CLI_utils import find_pkg_resource


def train(cfg, args):
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
	if args.gpu > -1:
		gpus = tf.config.list_physical_devices('GPU')
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)

	validation_split = args.val_split
	X, idx = read_genus_abu(args.i)
	Y = read_labels(args.labels, shuffle_idx=idx, dmax=get_dmax(args.labels))
	print('Total correct samples:', sum(X.index == Y.index))

	pretrain_ep = cfg.getint('train', 'pretrain_ep')
	pretrain_lr = cfg.getfloat('train', 'pretrain_lr')
	pretrain_stop_patience = cfg.getint('train', 'pretrain_stop_patience')

	lr = cfg.getfloat('train', 'lr')
	epochs = cfg.getint('train', 'epochs')
	reduce_patience = cfg.getint('train', 'reduce_patience')
	stop_patience = cfg.getint('train', 'stop_patience')
	batch_size = cfg.getint('train', 'batch_size')

	pretrain_logger = CSVLogger(filename=args.log)
	pretrain_stopper = EarlyStopping(monitor='val_loss', patience=pretrain_stop_patience, verbose=5,
									 restore_best_weights=True)

	logger = CSVLogger(filename=args.log)
	lrreducer = ReduceLROnPlateau(monitor='val_loss', patience=reduce_patience, min_lr=1e-5, verbose=5, factor=0.1)
	stopper = EarlyStopping(monitor='val_loss', patience=stop_patience, verbose=5, restore_best_weights=True)
	phylogeny = pd.read_csv(find_pkg_resource(cfg.get('DEFALUT', 'phylo')), index_col=0)
	pretrain_opt = Adam(lr=pretrain_lr)
	optimizer = Adam(lr=lr)

	# calculate sample weight for each layer, assign 0 weight for sample with 0 labels
	'''sample_weight = [zero_weight_unk(y=y, sample_weight=compute_sample_weight(class_weight='balanced', 
																			  y=y.to_numpy().argmax(axis=1)))
					 for i, y in enumerate(Y_train)]
	'''
	ontology = load_otlg(args.otlg)
	_, layer_units = parse_otlg(ontology)
	sample_weight = [zero_weight_unk(y=y, sample_weight=np.ones(y.shape[0])) for i, y in enumerate(Y)]

	model = Model(phylogeny=phylogeny, num_features=X.shape[1], ontology=ontology)
	X = model.encoder(X.to_numpy()).numpy().reshape(X.shape[0], X.shape[1] * phylogeny.shape[1])
	Xf_stats = {'mean': X.mean(), 'std': X.std() + 1e-8}
	np.save(os.path.join(find_pkg_resource(cfg.get('DEFALUT', 'tmp')), 'mean_f.for.X_train.npy'), Xf_stats['mean'])
	np.save(os.path.join(find_pkg_resource(cfg.get('DEFALUT', 'tmp')), 'std_f.for.X_train.npy'), Xf_stats['std'])
	X = (X - Xf_stats['mean']) / Xf_stats['std']
	Y = [y.iloc[:, :-1] for y in Y]

	print('Pre-training using Adam with lr={}...'.format(pretrain_lr))
	model.nn.compile(optimizer=pretrain_opt,
				  loss=BinaryCrossentropy(),
				  loss_weights=(np.array(layer_units) / sum(layer_units)).tolist(),
				  weighted_metrics=[BinaryAccuracy(name='acc')]) 
	model.nn.fit(X, Y, validation_split=validation_split,
			  batch_size=batch_size, epochs=pretrain_ep,
			  sample_weight=sample_weight,
			  callbacks=[pretrain_logger, lrreducer, pretrain_stopper][0:1])

	model.nn.summary()
	
	print('Training using Adam with lr={}...'.format(lr))
	model.nn.compile(optimizer=optimizer,
				  loss=BinaryCrossentropy(),
				  loss_weights=(np.array(layer_units) / sum(layer_units)).tolist(),
				  weighted_metrics=[BinaryAccuracy(name='acc'), 
				  					AUC(num_thresholds=100, name='auROC', multi_label=False), 
									AUC(num_thresholds=100, name='auPRC', curve='PR', multi_label=False)])
	model.nn.fit(X, Y, validation_split=validation_split,
			  batch_size=batch_size, initial_epoch=pretrain_ep, epochs=epochs + pretrain_ep,
			  sample_weight=sample_weight,
			  callbacks=[logger, lrreducer, pretrain_stopper][0:3])
	
	model.save_blocks(args.o)

	# exported to evaluate module
	'''test_sample_weight = [zero_weight_unk(y=y, sample_weight=np.ones(y.shape[0]))
					 for i, y in enumerate(Y_test)]
	X_test = model.encoder(X_test.to_numpy()).numpy().reshape(X_test.shape[0], X_test.shape[1] * phylogeny.shape[1])
	X_test = (X_test - Xf_stats['mean']) / Xf_stats['std']
	Y_test = [y.iloc[:, :-1] for y in Y_test]
	model.nn.evaluate(X_test, Y_test, sample_weight=test_sample_weight)'''
