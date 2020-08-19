from ONN.src.model import Model
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping
from ONN.src.utils import read_matrices, read_labels, parse_otlg, transfer_weights
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np
from tensorflow.distribute import MirroredStrategy
import tensorflow as tf
import os
from configparser import ConfigParser


# supporting user define layers to transfer !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# label smoothing
# stop after 30 epochs
def transfer(args):
	cfg = ConfigParser()
	cfg.read(args.cfg)
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
	gpus = tf.config.list_physical_devices('GPU')
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)

	X_train, X_test, shuffle_idx = read_matrices(args.i, split_idx=args.split_idx, end_idx=args.end_idx)
	Y_train, Y_test = read_labels(args.labels, shuffle_idx=shuffle_idx, split_idx=args.split_idx,
								  end_idx=args.end_idx, dmax=args.dmax)

	do_finetune = cfg.getboolean('transfer', 'do_finetune')
	new_mapper = cfg.getboolean('transfer', 'new_mapper')
	reuse_levels = cfg.get('transfer', 'reuse_levels')
	finetune_eps = cfg.getint('transfer', 'finetune_epochs')
	finetune_lr = cfg.getint('transfer', 'finetune_lr')
	warmup_eps = cfg.getint('transfer', 'warmup_epochs')
	epochs = cfg.getint('transfer', 'epochs')
	warmup_lr = cfg.getfloat('transfer', 'warmup_lr')
	lr = cfg.getfloat('transfer', 'lr')
	min_lr = cfg.getfloat('transfer', 'min_lr')
	reduce_patience = cfg.getint('transfer', 'reduce_patience')
	stop_patience = cfg.getint('transfer', 'stop_patience')
	label_smoothing = cfg.getfloat('transfer', 'label_smoothing')
	batch_size = cfg.getint('transfer', 'batch_size')

	warmup_logger = CSVLogger(filename=args.log)
	logger = CSVLogger(filename=args.log, append=True)
	lrreducer = ReduceLROnPlateau(patience=reduce_patience, verbose=5, factor=0.1, min_lr=min_lr)
	stopper = EarlyStopping(patience=stop_patience, verbose=5, restore_best_weights=True)

	_, layer_units = parse_otlg(args.otlg)  # sources and layer units
	sample_weight = [compute_sample_weight(class_weight='balanced', y=y.to_numpy().argmax(axis=1))
					 for i, y in enumerate(Y_train)]

	strategy = MirroredStrategy()
	with strategy.scope():
		base_model = Model(restore_from=args.model)
		init_model = Model(layer_units=layer_units, num_features=X_train.shape[1])
		# All transferred blocks and layers will be set to be non-trainable automatically.
		model = transfer_weights(base_model, init_model, new_mapper, reuse_levels)
		print('Warming up training using optimizer with lr={}...'.format(warmup_lr))
		model.compile(optimizer=Adam(lr=warmup_lr),
					  loss=CategoricalCrossentropy(from_logits=True, label_smoothing=label_smoothing),
					  loss_weights=layer_units,
					  metrics='acc')
	model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=batch_size, epochs=warmup_eps,
			  sample_weight=sample_weight,
			  callbacks=[warmup_logger])
	model.summary()

	with strategy.scope():
		print('Training using optimizer with lr={}...'.format(lr))
		model.compile(optimizer=Adam(lr=lr),
					  loss=CategoricalCrossentropy(from_logits=True, label_smoothing=label_smoothing),
					  loss_weights=layer_units,
					  metrics='acc')
	model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=batch_size, epochs=epochs,
			  initial_epoch=warmup_eps, sample_weight=sample_weight,
			  callbacks=[logger, lrreducer, stopper])

	if do_finetune:
		finetune_eps += stopper.stopped_epoch
		with strategy.scope():
			print('Fine-tuning using optimizer with lr={}...'.format(lr))
			model.fine_tune = True
			model.feature_mapper.trainable = True
			model.base.trainable = True
			for layer in range(model.n_layers):
				model.spec_inters[layer].trainable = True
				model.spec_integs[layer].trainable = True
				model.spec_outputs[layer].trainable = True
			model.compile(optimizer=Adam(lr=finetune_lr),
						  loss=CategoricalCrossentropy(from_logits=True, label_smoothing=label_smoothing),
						  loss_weights=layer_units,
						  metrics='acc')
		model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=batch_size, epochs=finetune_eps,
				  initial_epoch=stopper.stopped_epoch, sample_weight=sample_weight,
				  callbacks=[logger, lrreducer, stopper])
		model.save_blocks(args.o)
