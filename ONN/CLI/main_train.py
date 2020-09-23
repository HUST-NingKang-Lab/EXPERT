from ONN.src.model import Model
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping
from ONN.src.utils import read_genus_abu, read_matrices, read_labels, load_otlg, zero_weight_unk, parse_otlg
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Nadam
from tensorflow.keras.metrics import AUC, BinaryAccuracy
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from tensorflow.distribute import MirroredStrategy
from configparser import ConfigParser
import tensorflow as tf, numpy as np, pandas as pd, os
import tensorflow.keras.backend as K


def train(args):
	cfg = ConfigParser()
	cfg.read(args.cfg)
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = ''
	X_train, X_test, shuffle_idx = read_genus_abu(args.i, split_idx=args.split_idx, end_idx=args.end_idx)
	Y_train, Y_test = read_labels(args.labels, shuffle_idx=shuffle_idx, split_idx=args.split_idx,
								  end_idx=args.end_idx, dmax=args.dmax)
	print('Total correct samples:', sum(X_train.index == Y_train[0].index))

	pretrain_ep = cfg.getint('train', 'pretrain_ep')
	pretrain_lr = cfg.getfloat('train', 'pretrain_lr')
	pretrain_stop_patience = cfg.getint('train', 'pretrain_stop_patience')

	lr = cfg.getfloat('train', 'lr')
	epochs = cfg.getint('train', 'epochs')
	reduce_patience = cfg.getint('train', 'reduce_patience')
	stop_patience = cfg.getint('train', 'stop_patience')
	label_smoothing = cfg.getfloat('train', 'label_smoothing')
	batch_size = cfg.getint('train', 'batch_size')
	use_sgd = cfg.getboolean('train', 'use_sgd')

	pretrain_logger = CSVLogger(filename=args.log)
	pretrain_stopper = EarlyStopping(monitor='val_loss', patience=pretrain_stop_patience, verbose=5,
									 restore_best_weights=True)

	logger = CSVLogger(filename=args.log)
	lrreducer = ReduceLROnPlateau(monitor='val_loss', patience=reduce_patience, min_lr=1e-5, verbose=5, factor=0.1)
	stopper = EarlyStopping(monitor='val_loss', patience=stop_patience, verbose=5, restore_best_weights=True)
	#lrreducer = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=pretrain_lr, decay_steps=100, decay_rate=0.5)
	phylogeny = pd.read_csv(args.phylo, index_col=0)
	#pretrain_opt = Adam(lr=pretrain_lr, clipvalue=1)
	pretrain_opt = Adam(lr=pretrain_lr)
	if use_sgd:
		optimizer = SGD(lr=lr, momentum=0.9, nesterov=True)
	else:
		optimizer = Adam(lr=lr)

	# calculate sample weight for each layer, assign 0 weight for sample with 0 labels
	'''sample_weight = [zero_weight_unk(y=y, sample_weight=compute_sample_weight(class_weight='balanced', 
																			  y=y.to_numpy().argmax(axis=1)))
					 for i, y in enumerate(Y_train)]
	'''
	ontology = load_otlg(args.otlg)
	_, layer_units = parse_otlg(ontology)
	sample_weight = [zero_weight_unk(y=y, sample_weight=np.ones(y.shape[0])) for i, y in enumerate(Y_train)]
	#Xf_stats = {'mean': X_train.mean(), 'std': X_train.std() + 1e-8}
	#X_train = (X_train - Xf_stats['mean']) / Xf_stats['std']
	Y_train = [y.iloc[:, :-1] for y in Y_train]
	model = Model(phylogeny=phylogeny, num_features=X_train.shape[1], ontology=ontology)
	X_train = model.encoder(X_train.to_numpy()).numpy().reshape(X_train.shape[0], X_train.shape[1] * phylogeny.shape[1])
	Xf_stats = {'mean': X_train.mean(), 'std': X_train.std() + 1e-8}
	np.save(os.path.join(args.tmp, 'mean_f.for.X_train.json'), Xf_stats['mean'])
	np.save(os.path.join(args.tmp, 'var_f.for.X_train.json'), Xf_stats['std'])
	X_train = (X_train - Xf_stats['mean']) / Xf_stats['std']
	
	os.environ["CUDA_VISIBLE_DEVICES"] = cfg.get('train', 'gpu')
	gpus = tf.config.list_physical_devices('GPU')
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)

	print('Pre-training using Adam with lr={}...'.format(pretrain_lr))
	model.nn.compile(optimizer=pretrain_opt,
				  loss=BinaryCrossentropy(),
				  loss_weights=(np.array(layer_units) / sum(layer_units)).tolist(),
				  weighted_metrics=[BinaryAccuracy(name='acc')]) 
	model.nn.fit(X_train, Y_train, validation_split=0.1, #validation_data=(X_test, Y_test),
			  batch_size=batch_size, epochs=pretrain_ep,
			  sample_weight=sample_weight,
			  callbacks=[pretrain_logger, lrreducer, pretrain_stopper][0:1])

	model.nn.summary()
	
	print('Training using Adam with lr={}...'.format(lr))
	model.nn.compile(optimizer=optimizer,
				  loss=BinaryCrossentropy(),
				  loss_weights=(np.array(layer_units) / sum(layer_units)).tolist(),
				  weighted_metrics=[BinaryAccuracy(name='acc'), 
				  					AUC(num_thresholds=100, name='auROC'), 
									AUC(num_thresholds=100, name='auPRC', curve='PR')])
	model.nn.fit(X_train, Y_train, validation_split=0.1, #validation_data=(X_test, Y_test),
			  batch_size=batch_size, initial_epoch=pretrain_ep, epochs=epochs + pretrain_ep,
			  sample_weight=sample_weight,
			  callbacks=[logger, lrreducer, pretrain_stopper][0:3])
	
	model.save_blocks(args.o)
	test_sample_weight = [zero_weight_unk(y=y, sample_weight=np.ones(y.shape[0]))
					 for i, y in enumerate(Y_test)]
	X_test = model.encoder(X_test.to_numpy()).numpy().reshape(X_test.shape[0], X_test.shape[1] * phylogeny.shape[1])
	X_test = (X_test - Xf_stats['mean']) / Xf_stats['std']
	Y_test = [y.iloc[:, :-1] for y in Y_test]
	model.nn.evaluate(X_test, Y_test, sample_weight=test_sample_weight)
