#! /data2/public/chonghui_backup/anaconda3/bin/python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Concatenate, Conv2D, MaxPooling2D, Lambda, Add
import numpy as np
import os
from pprint import pprint
from tensorflow.keras import backend as K


# npz_file = '/data2/public/chonghui/MGNify_12.5w/subset_Soil_matrices_1462_features_coef_0.001.npz'
npz_file = '/data2/public/chonghui/MGNify_12.5w/matrices_1462_features_coef_0.001.npz'
gpu = True
split_idx = 4990
end_idx = 5000


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)

if gpu:
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)


def get_data_label():
	data = np.load(npz_file)
	matrices = data['matrices']
	l0, l1, l2, l3, l4 = (data['label_0'], data['label_1'], data['label_2'],
							  data['label_3'], data['label_4'])
	
	l0 = np.concatenate( (l0, (1-l0.sum(axis=1)).reshape(l0.shape[0], 1)), axis=1) # Unknown
	l1 = np.concatenate( (l1, (1-l1.sum(axis=1)).reshape(l1.shape[0], 1)), axis=1) # Unknown
	l2 = np.concatenate( (l2, (1-l2.sum(axis=1)).reshape(l2.shape[0], 1)), axis=1) # Unknown
	l3 = np.concatenate( (l3, (1-l3.sum(axis=1)).reshape(l3.shape[0], 1)), axis=1) # Unknown
	l4 = np.concatenate( (l4, (1-l4.sum(axis=1)).reshape(l4.shape[0], 1)), axis=1) # Unknown
		
	idx = np.arange(matrices.shape[0])
	np.random.seed(1)
	np.random.shuffle(idx)
	nidx = idx[0:end_idx]
	labels = [l0[nidx], l1[nidx], l2[nidx], l3[nidx], l4[nidx]]
	return matrices[nidx], labels

matrices, labels = get_data_label()

l_root = np.ones((labels[0].shape[0], 1))

#matrices = matrices.reshape(matrices.shape[0], 1462, 7, 1)

X = matrices
Y = [l_root] + labels
print(X.shape)
for i in Y:
	print(i.shape)
layer_weights = [Y[0].shape[1], Y[1].shape[1], Y[2].shape[1], Y[3].shape[1], Y[4].shape[1], Y[5].shape[1]]

def r_s(y_true, y_pred):
	SS_res =  K.sum(K.square(y_true - y_pred)) 
	SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
	return ( 1 - SS_res/(SS_tot + K.epsilon()) )

x = tf.keras.Input(shape=(1462, 7))
#print(x.shape)

#fully connected layers using for extract underlying logic
base = Dense(512, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
base = BatchNormalization()(base)
base = Dense(512, activation=tf.nn.relu)(base)
base = BatchNormalization()(base)
#base0_3 = Dense(base0_2, 1024, activation=tf.nn.relu)
base = Dense(512, activation=tf.nn.relu)(base)
base = BatchNormalization()(base)
base = Dense(256, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(base)
base = BatchNormalization()(base)

#Ontology layer0 for label0 classifying
#l0 = Dense(256, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(base)
#l0 = BatchNormalization()(l0)
l0 = Dense(64, activation=tf.nn.relu)(base)
l0 = BatchNormalization()(l0)
#l0 = Dense(64, activation=tf.nn.relu)(l0)
#l0 = BatchNormalization()(l0)
l0_f = Flatten()(l0)
l0_y = Dense(1, activation='sigmoid', name='l0')(l0_f)

#Ontology layer1 for label1 classifying
#l1 = Dense(256, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(base)
#l1 = BatchNormalization()(l1)
l1 = Dense(128, activation=tf.nn.relu)(base)
l1 = BatchNormalization()(l1)
#l1 = Dense(64, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.012))(l1)
#l1 = BatchNormalization()(l1)
l1_c = l1#tf.concat([l0, l1], axis=1)
l1_c = Dense(32, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(l1_c)
l1_c = BatchNormalization()(l1_c)
l1_c = Flatten()(l1_c)
l1_y = Dense(4, activation='sigmoid', name='l1_y')(l1_c)

#Ontology layer2 for label2 classifying
#l2 = Dense(256, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(base)
#l2 = BatchNormalization()(l2)
l2 = Dense(128, activation=tf.nn.relu)(base)
l2 = BatchNormalization()(l2)
#l2 = Dense(64, activation=tf.nn.relu)(l2)
#l2 = BatchNormalization()(l2)
l2_c = tf.concat([l2, l1], axis=1)
l2_c = Dense(64, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(l2_c)
l2_c = BatchNormalization()(l2_c)
l2_c = Flatten()(l2_c)
l2_y = Dense(7, activation='sigmoid', name='l2_y')(l2_c)

#Ontology layer3 for label3 classifying
#l3 = Dense(256, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(base)
#l3 = BatchNormalization()(l3)
l3 = Dense(128, activation=tf.nn.relu)(base)
l3 = BatchNormalization()(l3)
l3 = Dense(128, activation=tf.nn.relu)(l3)
l3 = BatchNormalization()(l3)
l3_c = tf.concat([l3, l2], axis=1)
l3_c = Dense(128, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(l3_c)
l3_c = BatchNormalization()(l3_c)
l3_c = Flatten()(l3_c)
l3_y = Dense(22, activation='sigmoid', name='l3_y')(l3_c)

#Ontology layer4 for label4 classifying
#l4 = Dense(256, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(base)
#l4 = BatchNormalization()(l4)
l4 = Dense(128, activation=tf.nn.relu)(base)
l4 = BatchNormalization()(l4)
l4 = Dense(128, activation=tf.nn.relu)(l4)
l4 = BatchNormalization()(l4)
l4_c = tf.concat([l4, l3], axis=1)
l4_c = Dense(256, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(l4_c)
l4 = BatchNormalization()(l4)
l4_c = Dense(128, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(l4_c)
l4 = BatchNormalization()(l4)
l4_c = Flatten()(l4_c)
l4_y = Dense(56, activation='sigmoid', name='l4_y')(l4_c)

#Ontology layer5 for label5 classifying
#l5 = Dense(256, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(base)
#l5 = BatchNormalization()(l5)
l5 = Dense(128, activation=tf.nn.relu)(base)
l5 = BatchNormalization()(l5)
l5 = Dense(128, activation=tf.nn.relu)(l5)
l5 = BatchNormalization()(l5)
l5_c = tf.concat([l5, l4], axis=1)
l5_c = Dense(256, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(l5_c)
l5_c = BatchNormalization()(l5_c)
l5_c = Dense(128, activation=tf.nn.relu)(l5_c)
l5_c = BatchNormalization()(l5_c)
l5_c = Flatten()(l5_c)
l5_y = Dense(43, activation='sigmoid', name='l5_y')(l5_c)

def scale_output(x):
	total_contrib = tf.constant([[1]], dtype=tf.float32, shape=(1, 1))
	unknown_contrib = tf.subtract(total_contrib, tf.keras.backend.sum(x, keepdims=True, axis=1))
	contrib = tf.keras.backend.relu(tf.keras.backend.concatenate( (x, unknown_contrib), axis=1))
	scaled_contrib = tf.divide(contrib, tf.keras.backend.sum(contrib, keepdims=True, axis=1))
	return scaled_contrib


l1_y = Lambda(scale_output, name='l1')(l1_y)
l2_y = Lambda(scale_output, name='l2')(l2_y)
l3_y = Lambda(scale_output, name='l3')(l3_y)
l4_y = Lambda(scale_output, name='l4')(l4_y)
l5_y = Lambda(scale_output, name='l5')(l5_y)

model = tf.keras.Model(inputs=x, outputs=[l0_y, l1_y, l2_y, l3_y, l4_y, l5_y])
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4, clipvalue=5),
			  loss={layer: tf.keras.losses.CategoricalCrossentropy(from_logits=False) 
					for layer in ['l' + str(i) for i in range(6)]},
			  loss_weights={'l' + str(i): np.log(i + 2) / np.log(np.arange(2, 8)).sum() for i in range(6)},
			  metrics=['acc'])

history = model.fit(x=X[0:split_idx], y=[y[0:split_idx] for y in Y], epochs=5, batch_size=128, validation_split=0.1, verbose=1) 
pprint(history)
# use sample_weight !!!!!!!!!!!!
model.evaluate(x=X[split_idx:], y=[y[split_idx:] for y in Y], verbose=1)
#res = model.predict(X[split_idx:])
#pprint('Prediction:------------------')
#pprint(res)
#pprint('Truth:-----------------------')
#t = [y[split_idx:] for y in Y]
#pprint(t)
model.summary()
model.save('/data2/public/chonghui_backup/onn_model_7.30', save_format='tf')
#np.savez('/data2/public/chonghui_backup/prediction', list(res))
#np.savez('/data2/public/chonghui_backup/truth', t)


