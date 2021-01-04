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
	# Basic configurations for GPU and CPU
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
	if args.gpu > -1:
		gpus = tf.config.list_physical_devices('GPU')
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)

	# Reading data
	X, idx = read_genus_abu(args.input)
	Y = read_labels(args.labels, shuffle_idx=idx, dmax=get_dmax(args.labels))
	print('Reordering labels and samples...')
	IDs = sorted(list(set(X.index.to_list()).intersection(Y[0].index.to_list())))
	X = X.loc[IDs, :]
	Y = [y.loc[IDs, :] for y in Y]
	print('Total matched samples:', sum(X.index == Y[0].index))

	# Reading basic configurations from config file
	pretrain_ep = cfg.getint('train', 'pretrain_ep')
	pretrain_lr = cfg.getfloat('train', 'pretrain_lr')

	lr = cfg.getfloat('train', 'lr')
	epochs = cfg.getint('train', 'epochs')
	reduce_patience = cfg.getint('train', 'reduce_patience')
	stop_patience = cfg.getint('train', 'stop_patience')
	batch_size = cfg.getint('train', 'batch_size')
	lrreducer = ReduceLROnPlateau(monitor='val_loss', patience=reduce_patience, min_lr=1e-5, verbose=5, factor=0.1)
	stopper = EarlyStopping(monitor='val_loss', patience=stop_patience, verbose=5, restore_best_weights=True)
	pretrain_callbacks = []
	callbacks = [lrreducer, stopper]
	if args.log:
		pretrain_logger = CSVLogger(filename=args.log)
		logger = CSVLogger(filename=args.log)
		pretrain_callbacks.append(pretrain_logger)
		callbacks.append(logger)
	phylogeny = pd.read_csv(find_pkg_resource('resources/phylogeny.csv'), index_col=0)
	pretrain_opt = Adam(lr=pretrain_lr)
	optimizer = Adam(lr=lr)
	dropout_rate = args.dropout_rate

	# Calculate sample weight for each layer, assign 0 weight for sample with 0 labels
	#sample_weight = [zero_weight_unk(y=y, sample_weight=compute_sample_weight(class_weight='balanced',
	#																		  y=y.to_numpy().argmax(axis=1)))
	#				 for i, y in enumerate(Y_train)]

	# Build the model
	ontology = load_otlg(args.otlg)
	_, layer_units = parse_otlg(ontology)
	model = Model(phylogeny=phylogeny, num_features=X.shape[1],
				  ontology=ontology, dropout_rate=dropout_rate)

	# Feature encoding and standardization
	X = model.encoder.predict(X, batch_size=batch_size)
	X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
	print('N. NaN in input features:', np.isnan(X).sum())
	model.update_statistics(mean=X.mean(axis=0), std=X.std(axis=0))
	X = model.standardize(X)

	#------------------------------- SELECTIVE LEARNING-----------------------------------------------
	# Sample weight "zero" to mask unknown samples' contribution to loss
	sample_weight = [zero_weight_unk(y=y, sample_weight=np.ones(y.shape[0])) for i, y in enumerate(Y)]
	Y = [y.drop(columns=['Unknown']) for y in Y]

	# Train EXPERT model
	print('Pre-training using Adam with lr={}...'.format(pretrain_lr))
	model.nn.compile(optimizer=pretrain_opt,
				  loss=BinaryCrossentropy(from_logits=True),
				  loss_weights=(np.array(layer_units) / sum(layer_units)).tolist(),
				  weighted_metrics=[BinaryAccuracy(name='acc')]) 
	model.nn.fit(X, Y, validation_split=args.val_split,
			  batch_size=batch_size, epochs=pretrain_ep,
			  sample_weight=sample_weight,
			  callbacks=pretrain_callbacks)

	model.nn.summary()
	
	print('Training using Adam with lr={}...'.format(lr))
	model.nn.compile(optimizer=optimizer,
				  loss=BinaryCrossentropy(from_logits=True),
				  loss_weights=(np.array(layer_units) / sum(layer_units)).tolist(),
				  weighted_metrics=[BinaryAccuracy(name='acc')])
	model.nn.fit(X, Y, validation_split=args.val_split,
			  batch_size=batch_size, initial_epoch=pretrain_ep, epochs=epochs + pretrain_ep,
			  sample_weight=sample_weight,
			  callbacks=callbacks)

	# Save the EXPERT model
	model.save_blocks(args.output)
