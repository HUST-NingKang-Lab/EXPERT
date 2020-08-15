from ONN.src.model import Model
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping
from ONN.src.utils import read_matrices, read_labels, parse_otlg
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np
from tensorflow.distribute import MirroredStrategy


def train(args):
	X_train, X_test, shuffle_idx = read_matrices(args.i, split_idx=args.split_idx, end_idx=args.end_idx)
	Y_train, Y_test = read_labels(args.labels, shuffle_idx=shuffle_idx, split_idx=args.split_idx,
								  end_idx=args.end_idx, dmax=args.dmax)
	
	warmup_ep = 2
	epochs = 500
	warmup_lr = 1e-4
	lr = 1e-3

	warmup_logger = CSVLogger(filename=args.log)
	logger = CSVLogger(filename=args.log, append=True)
	lrreducer = ReduceLROnPlateau(patience=5, verbose=5, min_lr=1e-10)
	stopper = EarlyStopping(patience=20, verbose=5, restore_best_weights=True)

	_, layer_units = parse_otlg(args.otlg)			   # sources and layer units
	sample_weight = [compute_sample_weight(class_weight='balanced', y=y.to_numpy().argmax(axis=1)) for i, y in enumerate(Y_train)]

	strategy = MirroredStrategy()
	with strategy.scope():
		model = Model(layer_units=layer_units)
		print('Warming up training using optimizer with lr={}...'.format(warmup_lr))
		model.compile(optimizer=Adam(lr=warmup_lr), loss=CategoricalCrossentropy(from_logits=True),
					  loss_weights=layer_units, 
					  metrics='acc')
	model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=512, epochs=warmup_ep,
			  sample_weight=sample_weight, 
			  callbacks=[warmup_logger])

	model.summary()
	
	with strategy.scope():
		print('Training using optimizer with lr={}...'.format(lr))
		model.compile(optimizer=Adam(lr=lr), loss=CategoricalCrossentropy(from_logits=True),
					  loss_weights=layer_units, 
					  metrics='acc')
	model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=512, epochs=epochs,
			  initial_epoch=warmup_ep, sample_weight=sample_weight, 
			  callbacks=[logger, lrreducer, stopper])
	model.save_blocks(args.o)
