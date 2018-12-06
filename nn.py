import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.models import Model, load_model
from keras.layers import *
from keras.optimizers import SGD
from keras import regularizers

from params import *

class NN(object):
	'''
		The architecture of the model is based off of the model used in this link:
		https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning/blob/master/model.py
	'''
	def __init__(self, input_shape, path_to_nn=None, load=False):
		self.input_shape = input_shape
		if load:
			self.load_model(path_to_nn)			
		else:
			self.model = self.create_model()
			

	def create_model(self):
		'''
			This method should create the deep neural net
			It should stack several residual/convulutional layers
			on top of each and then split off into the policy_head and the
			value_head
		'''
		inputs = Input(shape=self.input_shape, name='main_input')

		## Base of the model before split
		base = self.model_base(inputs)

		## Split net into value_head and the policy_head
		policy_head = self.policy_head(base)
		value_head = self.value_head(base)

		model = Model(inputs=inputs, outputs=[policy_head, value_head])
		model.compile(
					optimizer=SGD(lr=LEARNING_RATE, momentum=MOMENTUM), 
					loss={'value_head': 'mean_squared_logarithmic_error', 'policy_head': 'categorical_crossentropy'},
					loss_weights={'value_head': 0.5, 'policy_head': 0.5}
					)

		return model


	def fit(self, data, labels, epochs, batch_size):
		self.model.fit(data, labels, epochs=epochs, batch_size=batch_size)


	def evaluate(self, data, y, batch_size):
		return self.model.evaluate(data, y, batch_size=batch_size)


	def predict(self, data):
		return self.model.predict(data)


	def model_base(self, inputs):
		x = self.conv_layer(inputs, BASE_LAYERS[0]['filters'], BASE_LAYERS[0]['kernel_size'])

		for i in range(1,len(BASE_LAYERS) - 1):
			x = self.res_layer(x, BASE_LAYERS[i]['filters'], BASE_LAYERS[i]['kernel_size'])

		return x


	def policy_head(self, x):
		'''
			This head will create a probability vector of all of the
			moves
		'''
		p = self.conv_layer(x, 2, 1)
		p = Flatten()(p)
		p = Dense(
					42, 
					use_bias=False, 
					activation='linear', 
					kernel_regularizer=regularizers.l2(REG_CONST),
					name='policy_head'
					)(p)

		return p


	def value_head(self, x):
		'''
			This head will have 1 output node of how favorable the board is for the user
			between -1 and 1
		'''
		v = self.conv_layer(x, 1, 1)
		v = Flatten()(v)
		v = Dense(
					20, 
					use_bias=False, 
					activation='linear', 
					kernel_regularizer=regularizers.l2(REG_CONST)
					)(v)
		v = LeakyReLU()(v)
		v = Dense(
					1, 
					use_bias=False, 
					activation='tanh', 
					kernel_regularizer=regularizers.l2(REG_CONST),
					name='value_head'
					)(v)

		return v


	def res_layer(self, x, filters, kernel_size):
		model = self.conv_layer(x, filters, kernel_size)

		model = Conv2D(
			filters=filters, # integer, "dimensionality of the output space"
			kernel_size=kernel_size, # integer or tuple/list of 2 ints,height and width of window
			padding='same',
			data_format='channels_last',
			use_bias='False',
			activation='linear',
			kernel_regularizer=regularizers.l2(REG_CONST),
			)(model)

		model = BatchNormalization(axis=1)(model)
		model = add([model, x])
		model = LeakyReLU()(model)

		return model


	def conv_layer(self, x, filters, kernel_size):
		x = Conv2D(
			filters=filters, # integer, "dimensionality of the output space"
			kernel_size=kernel_size, # integer or tuple/list of 2 ints,height and width of window
			padding='same',
			data_format='channels_last',
			use_bias='False',
			activation='linear',
			kernel_regularizer=regularizers.l2(REG_CONST),
			)(x)

		x = BatchNormalization(axis=1)(x)
		x = LeakyReLU()(x)

		return x


	def load_model(self, path):
		self.model = load_model(path)


	def save_model(self, path):
		self.model.save(path)	
