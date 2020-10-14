from expert.src.model import Model
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping
from expert.src.utils import read_genus_abu, read_labels, parse_otlg, transfer_weights, zero_weight_unk, load_otlg
from expert.CLI.CLI_utils import find_pkg_resource
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import AUC, BinaryAccuracy
from sklearn.utils.class_weight import compute_sample_weight
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from expert.src.utils import get_dmax


def transfer(cfg, args):
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = cfg.get('train', 'gpu')
	if args.gpu > -1:
		gpus = tf.config.list_physical_devices('GPU')
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)

	X, idx = read_genus_abu(args.i)
	Y = read_labels(args.labels, shuffle_idx=idx, dmax=get_dmax(args.labels))
	validation_split = args.val_split
	phylogeny = pd.read_csv(args.phylo, index_col=0)
	do_finetune = cfg.getboolean('transfer', 'do_finetune')
	new_mapper = cfg.getboolean('transfer', 'new_mapper')
	reuse_levels = cfg.get('transfer', 'reuse_levels')
	finetune_eps = cfg.getint('transfer', 'finetune_epochs')
	finetune_lr = cfg.getfloat('transfer', 'finetune_lr')
	epochs = cfg.getint('transfer', 'epochs')
	lr = cfg.getfloat('transfer', 'lr')
	min_lr = cfg.getfloat('transfer', 'min_lr')
	reduce_patience = cfg.getint('transfer', 'reduce_patience')
	stop_patience = cfg.getint('transfer', 'stop_patience')
	label_smoothing = cfg.getfloat('transfer', 'label_smoothing')
	batch_size = cfg.getint('transfer', 'batch_size')

	logger = CSVLogger(filename=args.log)
	ft_logger = CSVLogger(filename=args.log, append=True)
	lrreducer = ReduceLROnPlateau(patience=reduce_patience, verbose=5, factor=0.1, min_lr=min_lr)
	stopper = EarlyStopping(patience=stop_patience, verbose=5, restore_best_weights=True)

	ontology = load_otlg(args.otlg)
	_, layer_units = parse_otlg(ontology)
	'''sample_weight = [compute_sample_weight(class_weight='balanced', y=y.to_numpy().argmax(axis=1))
					 for i, y in enumerate(Y_train)]'''
	sample_weight = [zero_weight_unk(y=y, sample_weight=np.ones(y.shape[0])) for i, y in enumerate(Y)]
	loss_weights = [units/sum(layer_units) for units in layer_units]
	
	optimizer = Adam(lr=lr)
	f_optimizer = Adam(lr=finetune_lr)

	base_model = Model(phylogeny=phylogeny, num_features=X.shape[1], restore_from=args.model)
	init_model = Model(phylogeny=phylogeny, num_features=X.shape[1], ontology=ontology)

	print('Total correct samples: {}?{}'.format(sum(X.index == Y[0].index), Y.shape[0]))
	X_train = init_model.encoder(X.to_numpy()).numpy().reshape(X.shape[0], X.shape[1] * phylogeny.shape[1])
	Xf_stats = {}
	Xf_stats['mean'] = np.load(os.path.join(find_pkg_resource(cfg.get('DEFALUT', 'tmp')), 'mean_f.for.X_train.npy'))
	Xf_stats['std'] = np.load(os.path.join(find_pkg_resource(cfg.get('DEFALUT', 'tmp')), 'var_f.for.X_train.npy'))
	X_train = (X_train - Xf_stats['mean']) / Xf_stats['std']
	Y_train = [y.drop(columns=['Unknown']) for y in Y]

	# All transferred blocks and layers will be set to be non-trainable automatically.
	model = transfer_weights(base_model, init_model, new_mapper, reuse_levels)
	model.nn = model.build_graph(input_shape=(X_train.shape[1], ))
	print('Training using optimizer with lr={}...'.format(lr))
	model.nn.compile(optimizer=optimizer,
				  loss=BinaryCrossentropy(label_smoothing=label_smoothing),
				  loss_weights=loss_weights, 
				  weighted_metrics=[BinaryAccuracy(name='acc'),
									AUC(num_thresholds=100, name='auROC', multi_label=False),
									AUC(num_thresholds=100, name='auPRC', curve='PR', multi_label=False)])
	model.nn.fit(X_train, Y_train, validation_split=validation_split, batch_size=batch_size, epochs=epochs,
			  sample_weight=sample_weight,
			  callbacks=[logger, lrreducer, stopper])
	model.nn.summary()

	if do_finetune:
		finetune_eps += stopper.stopped_epoch
		print('Fine-tuning using optimizer with lr={}...'.format(finetune_lr))
		model.base.trainable = True
		for layer in range(model.n_layers):
			model.spec_inters[layer].trainable = True
			model.spec_integs[layer].trainable = True
			model.spec_outputs[layer].trainable = True
		model.nn = model.build_graph(input_shape=(X_train.shape[1], ))
		model.nn.compile(optimizer=f_optimizer,
						 loss=BinaryCrossentropy(label_smoothing=label_smoothing),
						 loss_weights=loss_weights,
						 weighted_metrics=[BinaryAccuracy(name='acc'),
										   AUC(num_thresholds=100, name='auROC', multi_label=False),
										   AUC(num_thresholds=100, name='auPRC', curve='PR', multi_label=False)])
		model.nn.fit(X_train, Y_train, validation_split=validation_split,
				  batch_size=batch_size,
				  epochs=finetune_eps,
				  initial_epoch=stopper.stopped_epoch, sample_weight=sample_weight,
				  callbacks=[ft_logger, stopper])
		
		model.save_blocks(args.o)

		'''sample_weight_test = [zero_weight_unk(y=y, sample_weight=np.ones(y.shape[0])) for i, y in enumerate(Y_test)]	
		X_test = init_model.encoder(X_test.to_numpy()).numpy().reshape(X_test.shape[0], X_test.shape[1] * phylogeny.shape[1])
		X_test = (X_test - Xf_stats['mean']) / Xf_stats['std']
		Y_test = [y.drop(columns=['Unknown']) for y in Y_test]
		model.nn.evaluate(X_test, Y_test, sample_weight=sample_weight_test)'''
