import numpy as np

def cq_compare_2d(contexts, questions, c_max_len, q_max_len):
	cq_compare_vectors = []
	for i in range(len(contexts)):
		if i % 1000 == 0:
			print("progress .. (",i,"/",len(contexts),")")
		context = contexts[i]
		question = questions[i]

		vector_row = []
		for k in range(c_max_len):
			vector_col = []
			for j in range(q_max_len):
				if k >= len(context) or j >= len(question):
					vector_col.append(0)
				else:
					c = context[k]
					q = question[j]
					if c == q:
						vector_col.append(1)
					else:
						vector_col.append(0)
			vector_row.append(vector_col)
		cq_compare_vectors.append(vector_row)
	return np.array(cq_compare_vectors)


def context_compare_vector(contexts, questions, c_max_len):
	context_vectors = []
	for i in range(len(contexts)):
		#if i % 1000 == 0:
			#print("progress .. (",i,"/",len(questions),")")
		context = contexts[i]
		question = questions[i]

		vector = []
		for k in range(c_max_len):
			if k >= len(context):
				vector.append(0)
			else:
				c = context[k]
				if c in question:
					vector.append(1)
				else:
					vector.append(0)

		context_vectors.append(np.array(vector))
				
	return np.array(context_vectors)

def context_compare_vector_level(contexts, questions, c_max_len, level=2):
	context_vectors = []
	for i in range(len(contexts)):
		#if i % 1000 == 0:
			#print("progress .. (",i,"/",len(questions),")")
		context = contexts[i]
		question = questions[i]

		vector = []
		for k in range(c_max_len):
			if k >= len(context)-level:
				vector.append(0)
			else:
				c = context[k:k+level]
				if c in question:
					vector.append(1)
				else:
					vector.append(0)

		context_vectors.append(np.array(vector))
				
	return np.array(context_vectors)

def context_compare_punctuation_vector(contexts, c_max_len, punctuations="，。·、?%；\n~!《》「」[]:\""):
	print ('Compare marks...')
	context_vectors = []
	for i in range(len(contexts)):
		#if i % 1000 == 0:
			#print("progress .. (",i,"/",len(contexts),")")
		context = contexts[i]

		vector = []
		for k in range(c_max_len):
			if k >= len(context):
				vector.append(0)
			else:
				c = context[k]
				if c in punctuations:
					vector.append(1)
				else:
					vector.append(0)

		context_vectors.append(np.array(vector))
				
	return np.array(context_vectors)

def context_punctuation_vector_all_level(contexts, c_max_len):
	ccv_list = []
	ccv_list.append(context_compare_punctuation_vector(contexts, c_max_len, ",，"))
	ccv_list.append(context_compare_punctuation_vector(contexts, c_max_len, "。"))
	ccv_list.append(context_compare_punctuation_vector(contexts, c_max_len, "、"))
	ccv_list.append(context_compare_punctuation_vector(contexts, c_max_len, "《》"))
	ccv_list.append(context_compare_punctuation_vector(contexts, c_max_len, "「」"))
	ccv_list.append(context_compare_punctuation_vector(contexts, c_max_len, ":："))
	ccv_list.append(context_compare_punctuation_vector(contexts, c_max_len, ";；"))
	ccv_list.append(context_compare_punctuation_vector(contexts, c_max_len, "~"))
	ccv_list.append(context_compare_punctuation_vector(contexts, c_max_len, "·?%；\n!《》「」\""))
	context_vectors = [[[ccv[k][j] for ccv in ccv_list] for j in range(len(ccv_list[0][0]))] for k in range(len(ccv_list[0]))]
	return np.array(context_vectors)

def context_cinq_vector_level(contexts, questions, c_max_len, level=5):
	print ('Compare word level %d...' %(level))
	ccv_list = []
	context_vectors = []
	for i in range(len(contexts)):
		if i % 1000 == 0:
			print("progress .. (",i,"/",len(questions),")")
		context = contexts[i]
		question = questions[i]

		context_vector = []
		for k in range(c_max_len):
			vector = []
			for lv in range(1,level+1):
				cinq = 0
				cinq_weight = 0
				if k >= len(context)-lv:
					cinq = 0
				else:
					c = context[k:k+lv]
					if c in question:
						cinq = 1
						cinq_weight = len(context.split(c))-1
					else:
						cinq = 0
				vector.append(cinq)
				vector.append(cinq_weight)
			context_vector.append(vector)
		context_vectors.append(context_vector)
	return np.array(context_vectors)

def context_compare_vector_level_compress(contexts, questions, c_max_len, level=5):
	ccv_list = []
	for i in range(level):
		print ('Compare word level %d...' %(i+1))
		ccv_list.append(context_compare_vector_level(contexts, questions, c_max_len, level=i+1))
	
	ccv_list.append(context_compare_punctuation_vector(contexts, c_max_len, ",，"))
	ccv_list.append(context_compare_punctuation_vector(contexts, c_max_len, "。"))
	ccv_list.append(context_compare_punctuation_vector(contexts, c_max_len, "、"))
	ccv_list.append(context_compare_punctuation_vector(contexts, c_max_len, "《》"))
	ccv_list.append(context_compare_punctuation_vector(contexts, c_max_len, "「」"))
	ccv_list.append(context_compare_punctuation_vector(contexts, c_max_len, ":："))
	ccv_list.append(context_compare_punctuation_vector(contexts, c_max_len, ";；"))
	ccv_list.append(context_compare_punctuation_vector(contexts, c_max_len, "~"))
	ccv_list.append(context_compare_punctuation_vector(contexts, c_max_len, "·?%；\n!《》「」\""))
	context_vectors = [[[ccv[k][j] for ccv in ccv_list] for j in range(len(ccv_list[0][0]))] for k in range(len(ccv_list[0]))]
	return np.array(context_vectors)





