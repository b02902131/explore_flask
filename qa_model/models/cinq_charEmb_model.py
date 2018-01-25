import keras

from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dot, Flatten, Add, Concatenate, concatenate, Multiply, multiply
from keras.layers.core import *
from keras.layers import GRU, Bidirectional, TimeDistributed, Permute
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback

def cinq_charEmb_start_model(args):

	latent_dim = args.latent_dim

	'''Inputs P and Q'''
	P_Input = Input(shape=(args.c_max_len, ), name='P')
	Q_Input = Input(shape=(args.q_max_len, ), name='Q')
	char_embedding = Embedding(args.char_size, args.char_emb_dim)

	P = char_embedding(P_Input)
	Q = char_embedding(Q_Input)

	encoder = Bidirectional(GRU(units=args.char_emb_dim,
		return_sequences=True))

	passage_encoding = P
	passage_encoding = encoder(passage_encoding)
	passage_encoding = TimeDistributed(
		Dense(args.char_emb_dim,
			use_bias=False,
			trainable=True,
			weights=np.concatenate((np.eye(args.char_emb_dim), np.eye(args.char_emb_dim)), 
				axis=1)))(passage_encoding)

	question_encoding = Q
	question_encoding = encoder(question_encoding)
	question_encoding = TimeDistributed(
		Dense(args.char_emb_dim,
			use_bias=False,
			trainable=True,
			weights=np.concatenate((np.eye(args.char_emb_dim), np.eye(args.char_emb_dim)), 
			axis=1)))(question_encoding)

	'''Attention over question'''
	# compute the importance of each step
	question_attention_vector = TimeDistributed(Dense(1))(question_encoding)
	question_attention_vector = Lambda(lambda q: keras.activations.softmax(q, axis=1))(question_attention_vector)

	# apply the attention
	question_attention_vector = Lambda(lambda q: q[0] * q[1])([question_encoding, question_attention_vector])
	question_attention_vector = Lambda(lambda q: K.sum(q, axis=1))(question_attention_vector)
	question_attention_vector = RepeatVector(args.c_max_len)(question_attention_vector)

	# Answer start prediction
	answer_start_charEmb = Lambda(lambda arg: concatenate([arg[0], arg[1], arg[2]]))([
		passage_encoding,
		question_attention_vector,
		multiply([passage_encoding, question_attention_vector])])

	answer_start_charEmb = TimeDistributed(Dense(args.char_emb_dim, activation='relu'))(answer_start_charEmb)
	answer_start_charEmb = TimeDistributed(Dense(latent_dim))(answer_start_charEmb)

	'''Input context vector'''
	context_vector 	 = Input(shape=(args.c_max_len, args.context_vector_level+args.punctuation_level))

	gru_context = Bidirectional(GRU(args.hidden_size, return_sequences=True))(context_vector)
	answer_start_cinq = TimeDistributed(Dense(latent_dim))(gru_context)
	

	answer_start = Multiply()([answer_start_charEmb, answer_start_cinq])
	answer_start = Lambda(lambda a: K.sum(a, axis=2))(answer_start)
	# answer_start = Flatten()(answer_start)

	output_start = Activation(K.softmax)(answer_start)

	# define model in/out and compile
	outputs = [output_start]
	inputs = [context_vector, P_Input, Q_Input]
	model = Model(inputs, outputs)
	model.compile(optimizer=args.optimizer,
		loss=args.loss,
		metrics=['acc'])

	return model

def cinq_charEmb_end_model(args):
	latent_dim = args.latent_dim

	'''Inputs P and Q'''
	P_Input = Input(shape=(args.c_max_len, ), name='P')
	Q_Input = Input(shape=(args.q_max_len, ), name='Q')
	char_embedding = Embedding(args.char_size, args.char_emb_dim)

	'''Inputs context vector'''
	context_vector 	 = Input(shape=(args.c_max_len, args.context_vector_level+args.punctuation_level))
	answer_start = Input(shape=(args.c_max_len,1))

	'''char embedding interact'''
	P = char_embedding(P_Input)
	Q = char_embedding(Q_Input)

	encoder = Bidirectional(GRU(units=args.char_emb_dim,
		return_sequences=True))

	passage_encoding = P
	passage_encoding = encoder(passage_encoding)
	passage_encoding = TimeDistributed(
		Dense(args.char_emb_dim,
			use_bias=False,
			trainable=True,
			weights=np.concatenate((np.eye(args.char_emb_dim), np.eye(args.char_emb_dim)), 
				axis=1)))(passage_encoding)

	question_encoding = Q
	question_encoding = encoder(question_encoding)
	question_encoding = TimeDistributed(
		Dense(args.char_emb_dim,
			use_bias=False,
			trainable=True,
			weights=np.concatenate((np.eye(args.char_emb_dim), np.eye(args.char_emb_dim)), 
			axis=1)))(question_encoding)

	'''Attention over question'''
	# compute the importance of each step
	question_attention_vector = TimeDistributed(Dense(1))(question_encoding)
	question_attention_vector = Lambda(lambda q: keras.activations.softmax(q, axis=1))(question_attention_vector)

	# apply the attention
	question_attention_vector = Lambda(lambda q: q[0] * q[1])([question_encoding, question_attention_vector])
	question_attention_vector = Lambda(lambda q: K.sum(q, axis=1))(question_attention_vector)
	question_attention_vector = RepeatVector(args.c_max_len)(question_attention_vector)

	# Answer start prediction
	answer_end_charEmb = Lambda(lambda arg: concatenate([arg[0], arg[1], arg[2], arg[3], arg[4]]))([
		passage_encoding,
		question_attention_vector,
		answer_start, 
		multiply([passage_encoding, question_attention_vector]),
		multiply([passage_encoding, answer_start])])

	answer_end_charEmb = TimeDistributed(Dense(args.char_emb_dim, activation='relu'))(answer_end_charEmb)
	answer_end_charEmb = TimeDistributed(Dense(latent_dim))(answer_end_charEmb)

	# cinq method
	concat_start_context = Concatenate(2)([answer_start, context_vector])
	
	gru_start_context = Bidirectional(GRU(args.hidden_size, return_sequences=True))(concat_start_context)
	answer_end_cinq = TimeDistributed(Dense(latent_dim, activation='relu'))(gru_start_context)

	# merge two method
	answer_end = Multiply()([answer_end_charEmb, answer_end_cinq])
	answer_end = Lambda(lambda a: K.sum(a, axis=2))(answer_end)
	# answer_start = Flatten()(answer_start)

	output_end = Activation(K.softmax)(answer_end)
	
	# define model in/out and compile
	outputs = [output_end]
	inputs = [context_vector, P_Input, Q_Input, answer_start]
	model = Model(inputs, outputs)
	model.compile(optimizer=args.optimizer,
		loss=args.loss,
		metrics=['acc'])
	return model