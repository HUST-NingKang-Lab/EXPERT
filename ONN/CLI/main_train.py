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

	warmup_ep = cfg.getint('train', 'warmup_epochs')
	epochs = cfg.getint('train', 'epochs')
	warmup_lr = cfg.getfloat('train', 'warmup_lr')
	lr = cfg.getfloat('train', 'lr')
	min_lr = cfg.getfloat('train', 'min_lr')
	reduce_patience = cfg.getint('train', 'reduce_patience')
	stop_patience = cfg.getint('train', 'stop_patience')
	label_smoothing = cfg.getfloat('train', 'label_smoothing')
	batch_size = cfg.getint('train', 'batch_size')
	use_sgd = cfg.getboolean('train', 'use_sgd')

	warmup_logger = CSVLogger(filename=args.log)
	logger = CSVLogger(filename=args.log, append=True)
	lrreducer = ReduceLROnPlateau(patience=reduce_patience, verbose=5, factor=0.1, min_lr=min_lr)
	stopper = EarlyStopping(patience=stop_patience, verbose=5, restore_best_weights=True)
	if use_sgd:
		w_optimizer = SGD(lr=warmup_lr, momentum=0.9, nesterov=True)
		optimizer = SGD(lr=lr, momentum=0.9, nesterov=True)
	else:
		w_optimizer = Adam(lr=warmup_lr)
		optimizer = Adam(lr=lr)

	_, layer_units = parse_otlg(args.otlg)			   # sources and layer units
	sample_weight = [compute_sample_weight(class_weight='balanced', y=y.to_numpy().argmax(axis=1)) for i, y in enumerate(Y_train)]

	strategy = MirroredStrategy()
	with strategy.scope():
		model = Model(layer_units=layer_units, num_features=X_train.shape[1])
		print('Warming up training using optimizer with lr={}...'.format(warmup_lr))
		model.compile(optimizer=w_optimizer,
					  loss=CategoricalCrossentropy(from_logits=False, label_smoothing=label_smoothing),
					  #loss_weights=(np.array(layer_units) / sum(layer_units)).tolist(), 
					  metrics='acc')
	model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=batch_size, epochs=warmup_ep,
			  sample_weight=sample_weight, 
			  callbacks=[warmup_logger])

	with open('/data2/public/chonghui_backup/model_summary.txt', 'w') as f:
		model.summary(print_fn=lambda x: f.write(x + '\n'))
	model.summary()
	
	with strategy.scope():
		print('Training using optimizer with lr={}...'.format(lr))
		model.compile(optimizer=optimizer,
					  loss=CategoricalCrossentropy(from_logits=True, label_smoothing=label_smoothing),
					  #loss_weights=(np.array(layer_units) / sum(layer_units)).tolist(), 
					  metrics='acc')
	model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=batch_size, epochs=epochs,
			  initial_epoch=warmup_ep, sample_weight=sample_weight, 
			  callbacks=[logger, lrreducer, stopper])
	model.save_blocks(args.o)
	model = Model(restore_from=args.o)
