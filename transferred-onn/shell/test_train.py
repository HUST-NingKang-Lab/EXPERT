from src.utils import *


npz_file = ''


def get_data_label():
	data = np.load(npz_file)
	matrices = data['matrices']
	l0, l1, l2, l3, l4 = (data['label_0'], data['label_1'], data['label_2'],
							  data['label_3'], data['label_4'])
	labels = np.concatenate( (l0, l1, l2, l3, l4), axis=1)
	return matrices, labels

matrices, labels = get_data_label()

nlabels = np.concatenate( (np.ones((11528, 1)), labels), axis=1)

matrices = matrices.reshape(11528, 1462, 7, 1)

class Model(tf.keras.Model):

	def __init__(self, input_shape):

		# init Keras Model
		super(Model, self).__init__()
		self.base_c1_conv1 = Conv2D(filters=128, kernel_size=(1, 4), use_bias=True, name='base_c1_conv1',
								 padding='valid', input_shape=input_shape,
								 activation='relu') # 1462 * 4 * 128
		self.base_c1_conv2 = Conv2D(filters=256, kernel_size=(4, 1), padding='valid', name='base_c1_conv2',
								 use_bias=True,
								 activation='relu') # 1459 * 4 * 256
		self.base_c1_maxpool3 = MaxPooling2D(pool_size=(4, 1), strides=(3, 1), padding='valid') # 486 * 4 * 256
		self.base_c1_conv4 = Conv2D(filters=64, kernel_size=(1,1), padding='same', name='base_c1_con4',
									activation='relu') # 486 * 4 * 64
		self.base_c1_maxpool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid') # 243 * 2 * 64
		self.base_c1_flatten6 = Flatten()
		self.base_c1_dense7 = Dense(128, activation='relu', name='base_c1_dense7') # 31104 * 128
		self.base_c1_dense8 = Dense(128, activation='relu', name='base_c1_dense8') # 128 * 128

		self.base_c2_dense1 = Dense(512, activation='relu') # 1462 * 7 * 512
		self.base_c2_dense2 = Dense(256, activation='relu') # 1462 * 7 * 256
		self.base_c2_dense3 = Dense(128, activation='relu') # 1462 * 7 * 128
		self.base_c2_flatten4 = Flatten()
		self.base_c2_dense5 = Dense(128, activation='relu') # 2333352 * 128
		self.base_c2_dense6 = Dense(128, activation='relu') # 128 * 128
		self.concat = Concatenate(axis=1)

		self.l0_dense1 = Dense(128, activation='relu')
		self.l0_dense2 = Dense(64, activation='relu')
		self.l0_softmax = Dense(1, activation='softmax')

		self.l1_dense1 = Dense(128, activation='relu')
		self.l1_dense2 = Dense(64, activation='relu')
		self.l1_dense3 = Dense(128, activation='relu')
		self.l1_softmax = Dense(4, activation='softmax')

		self.l2_dense1 = Dense(128, activation='relu')
		self.l2_dense2 = Dense(64, activation='relu')
		self.l2_dense3 = Dense(128, activation='relu')
		self.l2_softmax = Dense(7, activation='softmax')

		self.l3_dense1 = Dense(128, activation='relu')
		self.l3_dense2 = Dense(64, activation='relu')
		self.l3_dense3 = Dense(128, activation='relu')
		self.l3_softmax = Dense(22, activation='softmax')

		self.l4_dense1 = Dense(128, activation='relu')
		self.l4_dense2 = Dense(64, activation='relu')
		self.l4_dense3 = Dense(128, activation='relu')
		self.l4_softmax = Dense(56, activation='softmax')

		self.l5_dense1 = Dense(128, activation='relu')
		self.l5_dense2 = Dense(64, activation='relu')
		self.l5_dense3 = Dense(128, activation='relu')
		self.l5_softmax = Dense(43, activation='softmax')

	def call(self, inputs, training=False):
		x = inputs

		# Inception-like structure
		xc1 = x
		xc1 = self.base_c1_conv1(xc1)
		xc1 = self.base_c1_conv2(xc1)
		xc1 = self.base_c1_maxpool3(xc1)
		xc1 = self.base_c1_conv4(xc1)
		xc1 = self.base_c1_maxpool5(xc1)
		xc1 = self.base_c1_flatten6(xc1)
		xc1 = self.base_c1_dense7(xc1)
		xc1 = self.base_c1_dense8(xc1)

		xc2 = x
		xc2 = self.base_c2_dense1(xc2)
		xc2 = self.base_c2_dense2(xc2)
		xc2 = self.base_c2_dense3(xc2)
		xc2 = self.base_c2_flatten4(xc2)
		xc2 = self.base_c2_dense5(xc2)
		xc2 = self.base_c2_dense6(xc2)

		#base = self.concat([xc1, xc2])
		base = xc1
		l0 = self.l0_dense1(base)
		l0 = self.l0_dense2(l0)

		l1 = self.l1_dense1(base)
		l1 = self.l1_dense2(l1)

		l2 = self.l2_dense1(base)
		l2 = self.l2_dense2(l2)

		l3 = self.l3_dense1(base)
		l3 = self.l3_dense2(l3)

		l4 = self.l4_dense1(base)
		l4 = self.l4_dense2(l4)

		l5 = self.l5_dense1(base)
		l5 = self.l5_dense2(l1)

		l1 = self.concat([l0, l1])
		l2 = self.concat([l1, l2])
		l3 = self.concat([l2, l3])
		l4 = self.concat([l3, l4])
		l5 = self.concat([l4, l5])

		l0 = self.l0_softmax(l0)
		l1 = self.l1_dense3(l1)
		l1 = self.l1_softmax(l1)
		l2 = self.l2_dense3(l2)
		l2 = self.l2_softmax(l2)
		l3 = self.l3_dense3(l3)
		l3 = self.l3_softmax(l3)
		l4 = self.l4_dense3(l4)
		l4 = self.l4_softmax(l4)
		l5 = self.l5_dense3(l5)
		l5 = self.l5_softmax(l5)

		y = self.concat([l0, l1, l2, l3, l4, l5])
		return y