from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.initializers import Identity, Constant, RandomNormal, RandomUniform
from keras.regularizers import l2

def basic_seq_model(input_shape, nb_actions, use_bias=True):
	model = Sequential()
	# model.add(Flatten(input_shape= input_shape))
	model.add(Flatten())
	model.add(Dense(128, use_bias=use_bias, kernel_initializer=RandomNormal(), bias_initializer=RandomUniform(), bias_regularizer=l2()))
	model.add(Activation('relu'))
	model.add(Dense(256, use_bias=use_bias, kernel_initializer=RandomNormal(), bias_initializer=RandomUniform(), bias_regularizer=l2()))
	model.add(Activation('relu'))
	model.add(Dense(256, use_bias=use_bias, kernel_initializer=RandomNormal(), bias_initializer=RandomUniform(), bias_regularizer=l2()))
	model.add(Activation('relu'))
	model.add(Dense(128, use_bias=use_bias, kernel_initializer=RandomNormal(), bias_initializer=RandomUniform(), bias_regularizer=l2()))
	model.add(Activation('relu'))
	model.add(Dense(nb_actions, use_bias=use_bias, kernel_initializer=RandomNormal(), bias_initializer=RandomUniform(), bias_regularizer=l2()))
	model.add(Activation('linear'))

	return model

def heavy_seq_model(input_shape, nb_actions,use_bias=True):
	model = Sequential()
	# model.add(Flatten(input_shape= input_shape))
	model.add(Flatten())
	model.add(Dense(256))
	model.add(Activation('tanh'))
	model.add(Dense(256))
	model.add(Activation('tanh'))
	model.add(Dense(256))
	model.add(Activation('tanh'))
	model.add(Dense(128))
	model.add(Activation('tanh'))
	model.add(Dense(128))
	model.add(Activation('tanh'))
	model.add(Dense(128))
	model.add(Activation('tanh'))
	model.add(Dense(nb_actions))
	model.add(Activation('linear'))

	return model

def very_heavy_seq_model(input_shape, nb_actions, use_bias=True):
	model = Sequential()
	# model.add(Flatten(input_shape= input_shape))
	model.add(Flatten())
	model.add(Dense(256, use_bias=use_bias))
	model.add(Activation('tanh'))
	model.add(Dense(256, use_bias=use_bias))
	model.add(Activation('tanh'))
	model.add(Dense(256, use_bias=use_bias))
	model.add(Activation('tanh'))
	model.add(Dense(128, use_bias=use_bias))
	model.add(Activation('tanh'))
	model.add(Dense(128, use_bias=use_bias))
	model.add(Activation('tanh'))
	model.add(Dense(128, use_bias=use_bias))
	model.add(Activation('tanh'))
	model.add(Dense(64, use_bias=use_bias))
	model.add(Activation('tanh'))
	model.add(Dense(64, use_bias=use_bias))
	model.add(Activation('tanh'))
	model.add(Dense(64, use_bias=use_bias))
	model.add(Activation('tanh'))
	model.add(Dense(nb_actions, use_bias=use_bias))
	model.add(Activation('linear'))

	return model

def dummy_model(input_shape, nb_actions, use_bias=True):
	model = Sequential()
	model.add(Flatten())
	# model.add(Flatten(input_shape= input_shape))
	model.add(Dense(nb_actions, kernel_initializer=Identity(gain=-1), use_bias=use_bias,  bias_initializer=Constant(1)))
	return model
