from expert.src.model import Model
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping
from expert.src.utils import read_genus_abu, read_labels, parse_otlg, transfer_weights, zero_weight_unk, load_otlg
from expert.CLI.CLI_utils import find_pkg_resource
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import AUC, BinaryAccuracy
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from expert.src.utils import get_dmax


def transfer(cfg, args):
	# Basic configurations for GPU and CPU
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
	if args.gpu > -1:
		gpus = tf.config.list_physical_devices('GPU')
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)

	# Read data
	X, idx = read_genus_abu(args.input)
	Y = read_labels(args.labels, shuffle_idx=idx, dmax=get_dmax(args.labels))
	print('Reordering labels and samples...')
	IDs = sorted(list(set(X.index.to_list()).intersection(Y[0].index.to_list())))
	X = X.loc[IDs, :]
	Y = [y.loc[IDs, :] for y in Y]
	print('Total matched samples:', sum(X.index == Y[0].index))

	# Basic configurations
	phylogeny = pd.read_csv(find_pkg_resource('resources/phylogeny.csv'), index_col=0)
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
	lrreducer = ReduceLROnPlateau(patience=reduce_patience, verbose=5, factor=0.1, min_lr=min_lr)
	stopper = EarlyStopping(patience=stop_patience, verbose=5, restore_best_weights=True)
	callbacks = [lrreducer, stopper]
	ft_callbacks = [stopper]
	if args.log:
		logger = CSVLogger(filename=args.log)
		ft_logger = CSVLogger(filename=args.log, append=True)
		callbacks.append(logger)
		ft_callbacks.append(ft_logger)
	optimizer = Adam(lr=lr)
	f_optimizer = Adam(lr=finetune_lr)
	dropout_rate = args.dropout_rate

	# Build EXPERT model
	ontology = load_otlg(args.otlg)
	_, layer_units = parse_otlg(ontology)
	base_model = Model(phylogeny=phylogeny, num_features=X.shape[1], restore_from=args.model)
	init_model = Model(phylogeny=phylogeny, num_features=X.shape[1],
					   ontology=ontology, dropout_rate=dropout_rate)

	# All transferred blocks and layers will be set to be non-trainable automatically.
	model = transfer_weights(base_model, init_model, reuse_levels)
	model.nn = model.build_graph(input_shape=(X.shape[1] * phylogeny.shape[1],))
	print('Total correct samples: {}?{}'.format(sum(X.index == Y[0].index), Y[0].shape[0]))

	# Feature encoding and standardization
	X = model.encoder.predict(X.to_numpy(), batch_size=128).reshape(X.shape[0], X.shape[1] * phylogeny.shape[1])
	if args.update_statistics:
		model.update_statistics(mean=X.mean(axis=0), std=X.std(axis=0))
	X = model.standardize(X)

	# Sample weight "zero" to mask unknown samples' contribution to loss
	sample_weight = [zero_weight_unk(y=y, sample_weight=np.ones(y.shape[0])) for i, y in enumerate(Y)]
	Y = [y.drop(columns=['Unknown']) for y in Y]

	# Train EXPERT model
	loss_weights = [units/sum(layer_units) for units in layer_units]
	print('Training using optimizer with lr={}...'.format(lr))
	model.nn.compile(optimizer=optimizer, loss=BinaryCrossentropy(from_logits=True, label_smoothing=label_smoothing),
					 loss_weights=loss_weights,
					 weighted_metrics=[BinaryAccuracy(name='acc')])
	model.nn.fit(X, Y, validation_split=args.val_split, batch_size=batch_size, epochs=epochs,
				 sample_weight=sample_weight, callbacks=callbacks)
	model.nn.summary()

	if args.finetune:
		finetune_eps += stopper.stopped_epoch
		print('Fine-tuning using optimizer with lr={}...'.format(finetune_lr))
		model.base.trainable = True
		for layer in range(model.n_layers):
			model.spec_inters[layer].trainable = True
			model.spec_integs[layer].trainable = True
			model.spec_outputs[layer].trainable = True
		model.nn = model.build_graph(input_shape=(X.shape[1], ))
		model.nn.compile(optimizer=f_optimizer,
						 loss=BinaryCrossentropy(from_logits=True, label_smoothing=label_smoothing),
						 loss_weights=loss_weights,
						 weighted_metrics=[BinaryAccuracy(name='acc')])
		model.nn.fit(X, Y, validation_split=args.val_split, batch_size=batch_size, epochs=finetune_eps,
					 initial_epoch=stopper.stopped_epoch, sample_weight=sample_weight, callbacks=ft_callbacks)

	# Save EXPERT model
	model.save_blocks(args.output)
