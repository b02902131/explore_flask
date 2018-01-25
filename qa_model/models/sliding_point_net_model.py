from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dot, Flatten, Add, Concatenate, Multiply
from keras.layers.core import *
from keras.layers import GRU, Bidirectional, TimeDistributed, Permute
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback

def onlyStartModel(args):
	context_vector = Input(shape=(args.c_max_len,1))
	gru_context = Bidirectional(GRU(args.hidden_size, return_sequences=True))(context_vector)
	pointer_start = TimeDistributed(Dense(1))(gru_context)
	pointer_start = Flatten()(pointer_start)

	# pointer_end = TimeDistributed(Dense(1))(gru_context)
	# pointer_end = Flatten()(pointer_end)

	output_start = Activation(K.softmax)(pointer_start)
	# output_end = Activation(K.softmax)(pointer_end)
	outputs = [output_start]

	inputs = context_vector

	model = Model(inputs, outputs)

	model.compile(optimizer=args.optimizer,
		loss=args.loss,
		metrics=['acc'])

	return model

def onlyStartModelMultiLayer(args):
	print("model!!!args.context_vector_level+args.punctuation_level)",args.context_vector_level+args.punctuation_level)
	context_vector 	 = Input(shape=(args.c_max_len, args.context_vector_level+args.punctuation_level))
	gru_context = Bidirectional(GRU(args.hidden_size, return_sequences=True))(context_vector)
	pointer_start = TimeDistributed(Dense(1))(gru_context)
	pointer_start = Flatten()(pointer_start)

	# pointer_end = TimeDistributed(Dense(1))(gru_context)
	# pointer_end = Flatten()(pointer_end)

	output_start = Activation(K.softmax)(pointer_start)
	# output_end = Activation(K.softmax)(pointer_end)
	outputs = [output_start]

	inputs = context_vector

	model = Model(inputs, outputs)

	model.compile(optimizer=args.optimizer,
		loss=args.loss,
		metrics=['acc'])

	return model

def onlyEndModelMultiLayer(args):
	context_vector 	 = Input(shape=(args.c_max_len, args.context_vector_level+args.punctuation_level))
	answer_start = Input(shape=(args.c_max_len,1))
	concat_start_context = Concatenate(2)([answer_start, context_vector])
	
	gru_start_context = Bidirectional(GRU(args.hidden_size, return_sequences=True))(concat_start_context)
	pointer_end = TimeDistributed(Dense(1, activation='relu'))(gru_start_context)
	pointer_end = Flatten()(pointer_end)

	# output_start = Activation(K.softmax)(pointer_start)
	output_end = Activation(K.softmax)(pointer_end)
	outputs = [output_end]

	inputs = [context_vector, answer_start]

	model = Model(inputs, outputs)

	model.compile(optimizer=args.optimizer,
		loss=args.loss,
		metrics=['acc'])

	return model


def onlyEndModel(args):
	context_vector = Input(shape=(args.c_max_len,1))
	answer_start = Input(shape=(args.c_max_len,1))
	concat_start_context = Concatenate(2)([answer_start, context_vector])
	
	gru_start_context = Bidirectional(GRU(args.hidden_size, return_sequences=True))(concat_start_context)
	pointer_end = TimeDistributed(Dense(1, activation='relu'))(gru_start_context)
	pointer_end = Flatten()(pointer_end)

	# output_start = Activation(K.softmax)(pointer_start)
	output_end = Activation(K.softmax)(pointer_end)
	outputs = [output_end]

	inputs = [context_vector, answer_start]

	model = Model(inputs, outputs)

	model.compile(optimizer=args.optimizer,
		loss=args.loss,
		metrics=['acc'])

	return model


def simpleRnn(args):

	context_vector = Input(shape=(c_max_len,1))
	gru_context = Bidirectional(GRU(args.hidden_size, return_sequences=True))(context_vector)
	pointer_start = TimeDistributed(Dense(1))(gru_context)
	pointer_start = Flatten()(pointer_start)

	pointer_end = TimeDistributed(Dense(1))(gru_context)
	pointer_end = Flatten()(pointer_end)

	output_start = Activation(K.softmax)(pointer_start)
	output_end = Activation(K.softmax)(pointer_end)
	outputs = [output_start, output_end]

	inputs = context_vector

	model = Model(inputs, outputs)

	model.compile(optimizer=args.optimizer,
		loss=args.loss,
		metrics=['acc'])

	return model

def pointRnn(args):
	context_vector = Input(shape=(c_max_len,1))
	gru_context = Bidirectional(GRU(args.hidden_size, return_sequences=True))(context_vector)
	pointer_start = TimeDistributed(Dense(1, activation='relu'))(gru_context)

	# output_start = Flatten()(pointer_start)
	output_start = Permute((2,1))(pointer_start)
	output_start = Activation(K.softmax)(output_start)
	output_start = Permute((2,1))(output_start)

	print("output_start.shape",output_start.shape)

	concat_start_context = Concatenate(2)([output_start, context_vector])
	gru_start_context = Bidirectional(GRU(args.hidden_size, return_sequences=True))(concat_start_context)
	pointer_end = TimeDistributed(Dense(1, activation='relu'))(gru_start_context)
	pointer_end = Flatten()(pointer_end)
	output_end = Activation(K.softmax)(pointer_end)

	inputs = context_vector

	output_start = Flatten()(output_start)
	outputs = [output_start, output_end]

	model = Model(inputs, outputs)

	model.compile(optimizer=args.optimizer,
		loss=args.loss,
		metrics=['acc'])

	return model





