from models.cinq_wordEmb_model import *

def question_classfier_model(args, embedding_matrix, isTeacherForcing=True):
	
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

	question_classfier = GRU(args.hidden_size, return_sequences=False)(question_encoding)
	question_classfier = Dense(args.latent_dim)(question_classfier)
	question_classfier = RepeatVector(args.c_max_len)(question_classfier)

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
	output_start = TimeDistributed(Dense(args.latent_dim))(output_start)

	output_start = Multiply()([output_start, question_classfier])
	output_start = Lambda(lambda a: K.sum(a, axis=2))(output_start)

	# output_start = Flatten()(output_start)

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
	
	if args.RNN_Type == 'gru':
		start_position = Bidirectional(GRU(8, return_sequences=True))(start_position)
	elif args.RNN_Type == 'lstm':
		start_position = Bidirectional(LSTM(8, return_sequences=True))(start_position)

	# Answer end prediction
	output_end = Lambda(lambda arg: concatenate([arg[i] for i in range(len(arg))]))([
		start_position,
		start_feature,
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
		multiply([passage_encoding, question_attention_vector]),
		multiply([passage_encoding, start_feature])])

	output_end = TimeDistributed(Dense(args.hidden_size, activation='relu'))(output_end)
	output_end = TimeDistributed(Dense(args.latent_dim))(output_end)

	output_end = Multiply()([output_end, question_classfier])
	output_end = Lambda(lambda a: K.sum(a, axis=2))(output_end)

	# output_end = Flatten()(output_end)

	output_end = Activation(K.softmax)(output_end)

	# define model in/out and compile
	inputs = [cinq_vector, punctuation_vector, 
	P_Input_word, Q_Input_word, P_Input_char, Q_Input_char, answer_start_input]
	outputs = [output_start, output_end]
	model = Model(inputs, outputs)
	model.compile(optimizer=args.optimizer,
		loss=args.loss,
		loss_weights=[0.9, 0.1],
		metrics=['acc'])

	return model