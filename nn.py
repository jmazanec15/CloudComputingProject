from keras.models import Model, load_model
from keras.layers import *
from keras.optimizers import SGD
from keras import regularizers

from params import params

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
					optimizer=SGD(lr=params['learning_rate'], momentum=params['momentum']), 
					loss={'value_head': 'mean_squared_logarithmic_error', 'policy_head': 'categorical_crossentropy'},
					loss_weights={'value_head': 0.5, 'policy_head': 0.5}
					)

		return model


	def fit(self, data, labels, epochs, batch_size):
		data = np.array([d[:-1].reshape(self.input_shape) for d in data])
		self.model.fit(data, labels, epochs=epochs, batch_size=batch_size)

	def evaluate(self, x, y, batch_size):
		data = np.array([d[:-1].reshape(self.input_shape) for d in x])
		return self.model.evaluate(data, y, batch_size=batch_size)

	def predict(self, data):
		shape = np.insert(self.input_shape, 0, 1)
		return self.model.predict(data[:-1].reshape(shape))

	def model_base(self, inputs):
		x = self.conv_layer(inputs, params['base_layers'][0]['filters'], params['base_layers'][0]['kernel_size'])

		for i in range(1,len(params['base_layers']) - 1):
			x = self.res_layer(x, params['base_layers'][i]['filters'], 
								params['base_layers'][i]['kernel_size'])

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
					kernel_regularizer=regularizers.l2(params['reg_const']),
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
					kernel_regularizer=regularizers.l2(params['reg_const'])
					)(v)
		v = LeakyReLU()(v)
		v = Dense(
					1, 
					use_bias=False, 
					activation='tanh', 
					kernel_regularizer=regularizers.l2(params['reg_const']),
					name='value_head'
					)(v)

		return v


	def res_layer(self, x, filters, kernel_size):
		model = self.conv_layer(x, filters, kernel_size)

		model = Conv2D(
			filters=filters, # integer, "dimensionality of the output space"
			kernel_size=kernel_size, # integer or tuple/list of 2 ints,height and width of window
			padding='same',
			data_format='channels_first',
			use_bias='False',
			activation='linear',
			kernel_regularizer=regularizers.l2(params['reg_const']),
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
			data_format='channels_first',
			use_bias='False',
			activation='linear',
			kernel_regularizer=regularizers.l2(params['reg_const']),
			)(x)

		x = BatchNormalization(axis=1)(x)
		x = LeakyReLU()(x)

		return x

	def load_model(self, path):
		self.model = load_model(path)

	def save_model(self, path):
		self.model.save(path)


		
