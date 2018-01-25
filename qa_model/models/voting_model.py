import keras

from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dot, Flatten, Add, Concatenate, concatenate, Multiply, multiply, Reshape
from keras.layers.core import *
from keras.layers import LSTM, GRU, Bidirectional, TimeDistributed, Permute
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback


def voting_model(args):
	
	nb_voter = 3
	nb_end = 2

	'''Voting Inputs'''
	voting_input = Input(shape=(args.c_max_len, nb_voter*nb_end))

	rnn_voting = Bidirectional(GRU(args.hidden_size, return_sequences=True))(voting_input)
	print("rnn_voting.shape",rnn_voting.shape)

	output_start = TimeDistributed(Dense(args.hidden_size, activation='relu'))(rnn_voting)
	output_start = TimeDistributed(Dense(1))(output_start)
	output_start = Flatten()(output_start)
	output_start = Activation(K.softmax)(output_start)

	'''Inputs answer_start '''
	answer_start_input = Input(shape=(args.c_max_len, ))
	
	if args.isTeacherForcing == True:
		answer_start = answer_start_input
	else:
		answer_start = output_start

	# answer_start = Reshape((args.c_max_len, 1), input_shape=(args.c_max_len, ))(answer_start)
	# Answer end prediction depends on the start prediction
	def s_answer_feature(x):
	    maxind = K.argmax(
	        x,
	        axis=1,
	    )
	    return maxind

	x = Lambda(lambda x: K.tf.cast(s_answer_feature(x), dtype=K.tf.int32))(answer_start)
	start_position = Lambda(lambda x: K.tf.one_hot(K.argmax(x), args.c_max_len))(answer_start)
	start_position = Reshape((args.c_max_len, 1), input_shape=(args.c_max_len, ))(start_position)
	start_position = Bidirectional(GRU(args.hidden_size, return_sequences=True))(start_position)

	# Answer end prediction
	output_end = Lambda(lambda arg: concatenate([arg[i] for i in range(len(arg))]))([
		start_position,
		rnn_voting])

	output_end = TimeDistributed(Dense(args.hidden_size, activation='relu'))(output_end)
	output_end = TimeDistributed(Dense(1))(output_end)
	output_end = Flatten()(output_end)
	output_end = Activation(K.softmax)(output_end)

	# define model in/out and compile
	inputs = [voting_input, answer_start_input]
	outputs = [output_start, output_end]
	model = Model(inputs, outputs)
	model.compile(optimizer=args.optimizer,
		loss=args.loss,
		loss_weights=[0.9, 0.1],
		metrics=['acc'])

	return model
