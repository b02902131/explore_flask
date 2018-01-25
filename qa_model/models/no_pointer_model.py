import keras

from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dot, Flatten, Add, Concatenate, concatenate, Multiply, multiply, Reshape
from keras.layers.core import *
from keras.layers import LSTM, GRU, Bidirectional, TimeDistributed, Permute
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback

def get_question_attention_vector(c_max_len, passage_encoding, question_encoding):
	'''Attention over question'''
	# compute the importance of each step
	question_attention_vector = TimeDistributed(Dense(1))(question_encoding)
	question_attention_vector = Lambda(lambda q: keras.activations.softmax(q, axis=1))(question_attention_vector)

	# apply the attention
	question_attention_vector = Lambda(lambda q: q[0] * q[1])([question_encoding, question_attention_vector])
	question_attention_vector = Lambda(lambda q: K.sum(q, axis=1))(question_attention_vector)

	# add a dense here
	question_attention_vector = Dense(int(question_attention_vector.shape[1]), activation='sigmoid')(question_attention_vector)

	question_attention_vector = RepeatVector(c_max_len)(question_attention_vector)

	return question_attention_vector

def encoder2(emb_dim, encode):
	encode = TimeDistributed(
		Dense(emb_dim,
			use_bias=False,
			trainable=True,
			weights=np.concatenate((np.eye(emb_dim), np.eye(emb_dim)), 
				axis=1)))(encode)
	return encode

def sentence_encode(args, P_Input_word, Q_Input_word, P_Input_char, 
	Q_Input_char, embedding_matrix):
	word_embedding = Embedding(args.vocab_size, args.embedding_dim, 
		weights=[embedding_matrix], trainable=args.word_trainable)

	P_word = word_embedding(P_Input_word)
	Q_word = word_embedding(Q_Input_word)

	char_embedding = Embedding(args.char_size, args.char_emb_dim)

	P_char = char_embedding(P_Input_char)
	Q_char = char_embedding(Q_Input_char)

	if args.RNN_Type == 'gru':
		char_encoder = Bidirectional(GRU(units=args.char_emb_dim,
			return_sequences=True))
		word_encoder = Bidirectional(GRU(units=args.embedding_dim,
			return_sequences=True))
	elif args.RNN_Type == 'lstm':
		char_encoder = Bidirectional(LSTM(units=args.char_emb_dim,
			return_sequences=True))
		word_encoder = Bidirectional(LSTM(units=args.embedding_dim,
			return_sequences=True))

	
	P_char_encode = char_encoder(P_char)
	P_word_encode = word_encoder(P_word)
	Q_char_encode = char_encoder(Q_char)
	Q_word_encode = word_encoder(Q_word)

	passage_encoding  = Concatenate(2)([P_char_encode, P_word_encode])
	question_encoding = Concatenate(2)([Q_char_encode, Q_word_encode])

	P_char_encode = encoder2(args.char_emb_dim, P_char_encode)
	P_word_encode = encoder2(args.embedding_dim, P_word_encode)
	Q_char_encode = encoder2(args.char_emb_dim, Q_char_encode)
	Q_word_encode = encoder2(args.embedding_dim, Q_word_encode)

	passage_encoding  = encoder2(args.char_emb_dim + args.embedding_dim, passage_encoding)
	question_encoding = encoder2(args.char_emb_dim + args.embedding_dim, question_encoding)

	return P_char_encode, P_word_encode, passage_encoding, \
	Q_char_encode, Q_word_encode, question_encoding


def get_attentions(args, P_char_encode, P_word_encode, passage_encoding, 
	Q_char_encode, Q_word_encode, question_encoding):
	
	q_char_attention_vector = get_question_attention_vector(args.c_max_len, P_char_encode, Q_char_encode)
	q_word_attention_vector = get_question_attention_vector(args.c_max_len, P_word_encode, Q_word_encode)
	question_attention_vector = get_question_attention_vector(args.c_max_len, passage_encoding, question_encoding)
	
	return q_char_attention_vector, q_word_attention_vector, question_attention_vector

def no_pointer_model(args, embedding_matrix, isTeacherForcing=True):
	
	'''Inputs P and Q'''
	P_Input_word = Input(shape=(args.c_max_len, ), name='Pword')
	Q_Input_word = Input(shape=(args.q_max_len, ), name='Qword')

	P_Input_char = Input(shape=(args.c_max_len, ), name='Pchar')
	Q_Input_char = Input(shape=(args.q_max_len, ), name='Qchar')

	'''Input context vector'''
	cinq_vector = Input(shape=(args.c_max_len, args.context_vector_level * 2))
	punctuation_vector = Input(shape=(args.c_max_len,  args.punctuation_level))

	if args.RNN_Type == 'gru':
		rnn_cinq_vector = Bidirectional(GRU(64, return_sequences=True))(cinq_vector)
		rnn_punctuation = Bidirectional(GRU(64, return_sequences=True))(punctuation_vector)
	elif args.RNN_Type == 'lstm':

		rnn_cinq_vector = Bidirectional(LSTM(64, return_sequences=True))(cinq_vector)
		rnn_punctuation = Bidirectional(LSTM(64, return_sequences=True))(punctuation_vector)

	P_char_encode, P_word_encode, passage_encoding, \
	Q_char_encode, Q_word_encode, question_encoding \
	= sentence_encode(args, P_Input_word, Q_Input_word, P_Input_char, Q_Input_char, embedding_matrix)

	q_char_attention_vector, q_word_attention_vector, question_attention_vector \
	= get_attentions(args, P_char_encode, P_word_encode, passage_encoding, 
	Q_char_encode, Q_word_encode, question_encoding)

	# Answer start prediction
	output_start  = Lambda(lambda arg: concatenate([arg[i] for i in range(len(arg))]))([
		P_char_encode,
		P_word_encode,
		passage_encoding,
		q_char_attention_vector,
		q_word_attention_vector,
		question_attention_vector,
		rnn_cinq_vector,
		rnn_punctuation,
		multiply([P_char_encode, q_char_attention_vector]),
		multiply([P_word_encode, q_word_attention_vector]),
		multiply([passage_encoding, question_attention_vector])])

	output_start = TimeDistributed(Dense(args.hidden_size, activation='relu'))(output_start)
	output_probs = TimeDistributed(Dense(1, activation='sigmoid'))(output_start)

	output_probs = Flatten()(output_probs)

	output_probs = Activation(K.softmax)(output_probs)

	# define model in/out and compile
	inputs = [cinq_vector, punctuation_vector, 
	P_Input_word, Q_Input_word, P_Input_char, Q_Input_char]
	outputs = [output_probs]
	model = Model(inputs, outputs)
	model.compile(optimizer=args.optimizer,
		loss=args.loss,
		metrics=['acc'])

	return model

def cinq_wordEmb_model(args, embedding_matrix, isTeacherForcing=True):
	
	

	'''Inputs P and Q'''
	P_Input_word = Input(shape=(args.c_max_len, ), name='Pword')
	Q_Input_word = Input(shape=(args.q_max_len, ), name='Qword')

	P_Input_char = Input(shape=(args.c_max_len, ), name='Pchar')
	Q_Input_char = Input(shape=(args.q_max_len, ), name='Qchar')

	'''Input context vector'''
	cinq_vector = Input(shape=(args.c_max_len, args.context_vector_level))
	gru_cinq = Bidirectional(GRU(64, return_sequences=True))(cinq_vector)

	punctuation_vector = Input(shape=(args.c_max_len,  args.punctuation_level))

	P_char_encode, P_word_encode, passage_encoding, \
	Q_char_encode, Q_word_encode, question_encoding \
	= sentence_encode(args, P_Input_word, Q_Input_word, P_Input_char, Q_Input_char, embedding_matrix)

	q_char_attention_vector, q_word_attention_vector, question_attention_vector \
	= get_attentions(args, P_char_encode, P_word_encode, passage_encoding, 
	Q_char_encode, Q_word_encode, question_encoding)

	# Answer start prediction
	output_start  = Lambda(lambda arg: concatenate([arg[i] for i in range(len(arg))]))([
		passage_encoding,
		q_char_attention_vector,
		q_word_attention_vector,
		question_attention_vector,
		gru_cinq,
		punctuation_vector,
		multiply([P_char_encode, q_char_attention_vector]),
		multiply([P_word_encode, q_word_attention_vector]),
		multiply([passage_encoding, question_attention_vector])])

	output_start = TimeDistributed(Dense(args.hidden_size, activation='relu'))(output_start)
	output_start = TimeDistributed(Dense(1))(output_start)

	output_start = Flatten()(output_start)

	output_start = Activation(K.softmax)(output_start)

	'''Inputs answer_start '''
	answer_start_input = Input(shape=(args.c_max_len, ))
	
	if isTeacherForcing == True:
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
	start_feature = Lambda(lambda arg: K.tf.gather_nd(arg[0], K.tf.stack(
	    [K.tf.range(K.tf.shape(arg[1])[0]), K.tf.cast(arg[1], K.tf.int32)], axis=1)))([passage_encoding, x])
	start_feature = RepeatVector(args.c_max_len)(start_feature)

	start_position = Lambda(lambda x: K.tf.one_hot(K.argmax(x), args.c_max_len))(answer_start)
	start_position = Reshape((args.c_max_len, 1), input_shape=(args.c_max_len, ))(start_position)
	start_position = Bidirectional(GRU(8, return_sequences=True))(start_position)

	# Answer end prediction
	output_end = Lambda(lambda arg: concatenate([arg[i] for i in range(len(arg))]))([
		start_position,
		start_feature,
		passage_encoding,
		q_char_attention_vector,
		q_word_attention_vector,
		question_attention_vector,
		gru_cinq,
		punctuation_vector,
		multiply([P_char_encode, q_char_attention_vector]),
		multiply([P_word_encode, q_word_attention_vector]),
		multiply([passage_encoding, question_attention_vector]),
		multiply([passage_encoding, start_feature])])

	output_end = TimeDistributed(Dense(args.hidden_size, activation='relu'))(output_end)
	output_end = TimeDistributed(Dense(1))(output_end)

	output_end = Flatten()(output_end)

	output_end = Activation(K.softmax)(output_end)

	# define model in/out and compile
	inputs = [cinq_vector, punctuation_vector, 
	P_Input_word, Q_Input_word, P_Input_char, Q_Input_char, answer_start_input]
	outputs = [output_start, output_end]
	model = Model(inputs, outputs)
	model.compile(optimizer=args.optimizer,
		loss=args.loss,
		metrics=['acc'])

	return model

def cinq_wordEmb_start_model(args, embedding_matrix):

	'''Inputs P and Q'''
	P_Input_word = Input(shape=(args.c_max_len, ), name='Pword')
	Q_Input_word = Input(shape=(args.q_max_len, ), name='Qword')

	P_Input_char = Input(shape=(args.c_max_len, ), name='Pchar')
	Q_Input_char = Input(shape=(args.q_max_len, ), name='Qchar')

	'''Input context vector'''
	cinq_vector = Input(shape=(args.c_max_len, args.context_vector_level))
	gru_cinq = Bidirectional(GRU(args.hidden_size, return_sequences=True))(cinq_vector)

	punctuation_vector = Input(shape=(args.c_max_len,  args.punctuation_level))

	P_char_encode, P_word_encode, passage_encoding, \
	Q_char_encode, Q_word_encode, question_encoding \
	= sentence_encode(args, P_Input_word, Q_Input_word, P_Input_char, Q_Input_char, embedding_matrix)

	q_char_attention_vector, q_word_attention_vector, question_attention_vector \
	= get_attentions(args, P_char_encode, P_word_encode, passage_encoding, 
	Q_char_encode, Q_word_encode, question_encoding)

	# Answer start prediction
	answer_start_charEmb = Lambda(lambda arg: concatenate([arg[i] for i in range(len(arg))]))([
		P_char_encode,
		P_word_encode,
		passage_encoding,
		q_char_attention_vector,
		q_word_attention_vector,
		question_attention_vector,
		gru_cinq,
		punctuation_vector,
		multiply([P_char_encode, q_char_attention_vector]),
		multiply([P_word_encode, q_word_attention_vector]),
		multiply([passage_encoding, question_attention_vector])])

	answer_start = TimeDistributed(Dense(args.hidden_size, activation='relu'))(answer_start_charEmb)
	answer_start = TimeDistributed(Dense(1))(answer_start)

	answer_start = Flatten()(answer_start)

	output_start = Activation(K.softmax)(answer_start)

	# define model in/out and compile
	outputs = [output_start]
	inputs = [
	cinq_vector, punctuation_vector, 
	P_Input_word, Q_Input_word, 
	P_Input_char, Q_Input_char]
	model = Model(inputs, outputs)
	model.compile(optimizer=args.optimizer,
		loss=args.loss,
		metrics=['acc'])

	return model

def cinq_wordEmb_end_model(args, embedding_matrix):

	'''Inputs answer_start '''
	answer_start_input = Input(shape=(args.c_max_len, ))
	answer_start = Reshape((args.c_max_len, 1), input_shape=(args.c_max_len, ))(answer_start_input)

	'''Inputs P and Q'''
	P_Input_word = Input(shape=(args.c_max_len, ), name='Pword')
	Q_Input_word = Input(shape=(args.q_max_len, ), name='Qword')

	P_Input_char = Input(shape=(args.c_max_len, ), name='Pchar')
	Q_Input_char = Input(shape=(args.q_max_len, ), name='Qchar')

	'''Input context vector'''
	cinq_vector = Input(shape=(args.c_max_len, args.context_vector_level))
	gru_cinq = Bidirectional(GRU(args.hidden_size, return_sequences=True))(cinq_vector)

	punctuation_vector = Input(shape=(args.c_max_len,  args.punctuation_level))

	P_char_encode, P_word_encode, passage_encoding, \
	Q_char_encode, Q_word_encode, question_encoding \
	= sentence_encode(args, P_Input_word, Q_Input_word, P_Input_char, Q_Input_char, embedding_matrix)

	q_char_attention_vector, q_word_attention_vector, question_attention_vector \
	= get_attentions(args, P_char_encode, P_word_encode, passage_encoding, 
	Q_char_encode, Q_word_encode, question_encoding)

	# Answer end prediction
	answer_end = Lambda(lambda arg: concatenate([arg[i] for i in range(len(arg))]))([
		answer_start,
		P_char_encode,
		P_word_encode,
		passage_encoding,
		q_char_attention_vector,
		q_word_attention_vector,
		question_attention_vector,
		gru_cinq,
		punctuation_vector,
		multiply([P_char_encode, q_char_attention_vector]),
		multiply([P_word_encode, q_word_attention_vector]),
		multiply([passage_encoding, question_attention_vector]),
		multiply([P_char_encode, answer_start]),
		multiply([P_word_encode, answer_start]),
		multiply([passage_encoding, answer_start])])

	answer_end = TimeDistributed(Dense(args.hidden_size, activation='relu'))(answer_end)
	answer_end = TimeDistributed(Dense(1))(answer_end)

	answer_end = Flatten()(answer_end)

	output_end = Activation(K.softmax)(answer_end)

	# define model in/out and compile
	outputs = [output_end]
	inputs = [cinq_vector, punctuation_vector, 
	P_Input_word, Q_Input_word, P_Input_char, Q_Input_char, answer_start_input]
	model = Model(inputs, outputs)
	model.compile(optimizer=args.optimizer,
		loss=args.loss,
		metrics=['acc'])

	return model

