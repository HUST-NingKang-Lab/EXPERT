from ONN.src.model import Model
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping
from ONN.src.utils import read_matrices, read_labels, parse_otlg
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np
from tensorflow.distribute import MirroredStrategy
import tensorflow as tf
import os
from configparser import ConfigParser


def train(args):
	cfg = ConfigParser()
	cfg.read(args.cfg)
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
	gpus = tf.config.list_physical_devices('GPU')
	for gpu in gpus:
	   tf.config.experimental.set_memory_growth(gpu, True)

	X_train, X_test, shuffle_idx = read_matrices(args.i, split_idx=args.split_idx, end_idx=args.end_idx)
	Y_train, Y_test = read_labels(args.labels, shuffle_idx=shuffle_idx, split_idx=args.split_idx,
								  end_idx=args.end_idx, dmax=args.dmax)

	pretrain_ep = cfg.getint('train', 'pretrain_ep')
	pretrain_lr = cfg.getfloat('train', 'pretrain_lr')
	pretrain_stop_patience = cfg.getint('train', 'pretrain_stop_patience')

	lr = cfg.getfloat('train', 'lr')
	min_lr = cfg.getfloat('train', 'min_lr')
	epochs = cfg.getint('train', 'epochs')
	reduce_patience = cfg.getint('train', 'reduce_patience')
	stop_patience = cfg.getint('train', 'stop_patience')
	label_smoothing = cfg.getfloat('train', 'label_smoothing')
	batch_size = cfg.getint('train', 'batch_size')
	use_sgd = cfg.getboolean('train', 'use_sgd')

	warmup_logger = CSVLogger(filename=args.log)
	logger = CSVLogger(filename=args.log, append=True)
	pretrain_stopper = EarlyStopping(monitor='val_loss', patience=pretrain_stop_patience, verbose=5, 
									 restore_best_weights=True)
	lrreducer = ReduceLROnPlateau(patience=reduce_patience, verbose=5, factor=0.2, min_lr=min_lr)
	stopper = EarlyStopping(monitor='val_loss', patience=stop_patience, verbose=5, restore_best_weights=True)
	pretrain_opt = Adam(lr=pretrain_lr)
	if use_sgd:
		optimizer = SGD(lr=lr, momentum=0.9, nesterov=True)
	else:
		optimizer = Adam(lr=lr)

	_, layer_units = parse_otlg(args.otlg)			   # sources and layer units
	sample_weight = [compute_sample_weight(class_weight='balanced', y=y.to_numpy().argmax(axis=1)) for i, y in enumerate(Y_train)]
	loss_weights = [units/sum(layer_units) for units in layer_units]
	print('Loss weights: ', loss_weights)

	strategy = MirroredStrategy()
	with strategy.scope():
		model = Model(layer_units=layer_units, num_features=X_train.shape[1])
		print('Pre-training using Adam with lr={}...'.format(pretrain_lr))
		model.compile(optimizer=pretrain_opt,
					  loss=CategoricalCrossentropy(from_logits=True, label_smoothing=label_smoothing),
					  loss_weights=loss_weights, 
					  metrics=['acc'])
	model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=batch_size, epochs=pretrain_ep,
			  sample_weight=sample_weight, 
			  callbacks=[warmup_logger, pretrain_stopper])

	with open('/data2/public/chonghui_backup/model_summary.txt', 'w') as f:
		model.summary(print_fn=lambda x: f.write(x + '\n'))
	model.summary()
	
	model.save_blocks(args.o+'pre_training')
	with strategy.scope():
		epochs += pretrain_stopper.stopped_epoch
		print('Training using optimizer with lr={}...'.format(lr))
		model.compile(optimizer=optimizer,
					  loss=CategoricalCrossentropy(from_logits=True, label_smoothing=label_smoothing),
					  loss_weights=loss_weights,
					  metrics=['acc'])
	model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=batch_size, epochs=epochs,
			  initial_epoch=pretrain_stopper.stopped_epoch, sample_weight=sample_weight,
			  callbacks=[logger, lrreducer, stopper])
	model.save_blocks(args.o)
