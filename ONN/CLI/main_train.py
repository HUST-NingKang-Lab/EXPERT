from ONN.src.model import Model, CyclicLR
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping
from ONN.src.utils import read_genus_abu, read_matrices, read_labels, parse_otlg
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.utils.class_weight import compute_sample_weight
from tensorflow.distribute import MirroredStrategy
from configparser import ConfigParser
import tensorflow as tf, numpy as np, pandas as pd, os
import tensorflow.keras.backend as K


def train(args):
	cfg = ConfigParser()
	cfg.read(args.cfg)
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = cfg.get('train', 'gpu')
	gpus = tf.config.list_physical_devices('GPU')
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)

	X_train, X_test, shuffle_idx = read_genus_abu(args.i, split_idx=args.split_idx, end_idx=args.end_idx)
	Y_train, Y_test = read_labels(args.labels, shuffle_idx=shuffle_idx, split_idx=args.split_idx,
								  end_idx=args.end_idx, dmax=args.dmax)

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
	lrreducer = ReduceLROnPlateau(monitor='val_loss', patience=reduce_patience, verbose=5, factor=0.1)
	stopper = EarlyStopping(monitor='val_loss', patience=stop_patience, verbose=5, restore_best_weights=True)
	clr = CyclicLR(base_lr=pretrain_lr, max_lr=0.01, step_size=100., mode='exp_range', gamma=0.99994)	

	phylogeny = pd.read_csv(args.phylo, index_col=0)
	pretrain_opt = Adam(lr=pretrain_lr, clipvalue=50)
	if use_sgd:
		optimizer = SGD(lr=lr, momentum=0.9, nesterov=True)
	else:
		optimizer = Adam(lr=lr, clipvalue=50)

	def r2(y_true, y_pred):
		a = K.square(y_pred - y_true)
		b = K.sum(a)
		c = K.mean(y_true)
		d = K.square(y_true - c)
		e = K.sum(d)
		f = 1 - b / e
		return f

	_, layer_units = parse_otlg(args.otlg)			   # sources and layer units
	sample_weight = [compute_sample_weight(class_weight='balanced', y=y.to_numpy().argmax(axis=1)) for i, y in enumerate(Y_train)]
	
	units = np.array(layer_units)
	loss_weights = (units - units.min()) / (units.max()-units.min()) * 99 + 1
	loss_weights = loss_weights.tolist()
	print('Loss weights: ', loss_weights)

	model = Model(phylogeny=phylogeny, layer_units=layer_units, num_features=X_train.shape[1])
	print('Pre-training using Adam with lr={}...'.format(pretrain_lr))
	model.compile(optimizer=pretrain_opt,
				  loss='mse',
				  loss_weights=[layer_units, loss_weights][0],
				  metrics=[r2])
	model.fit(X_train, Y_train, validation_data=(X_test, Y_test), 
			  batch_size=batch_size, epochs=pretrain_ep,
			  #sample_weight=sample_weight, 
			  callbacks=[pretrain_logger, pretrain_stopper, clr][0:1])

	with open('/data2/public/chonghui_backup/model_summary.txt', 'w') as f:
		model.summary(print_fn=lambda x: f.write(x + '\n'))
	model.summary()
	model.compile(optimizer=pretrain_opt,
				  loss='mse',
				  loss_weights=[1,1,1,1,1], 
				  metrics=[r2])
	model.fit(X_train, Y_train, validation_data=(X_test, Y_test), 
			  batch_size=batch_size, initial_epoch=pretrain_ep, epochs=epochs,
			  #sample_weight=sample_weight, 
			  callbacks=[logger, lrreducer, pretrain_stopper, clr][0:1])

	print(model.evaluate(X_test, Y_test))
	model.save_blocks(args.o)
