import numpy as np
import pandas as pd
import json
from math import log, floor


def randomize_dataset(dataset_list):
	permutation = np.random.permutation(len(dataset_list[0]))

	new_dataset_list = []
	for dataset in dataset_list:
		dataset = dataset[permutation]
		new_dataset_list.append(dataset)

	return new_dataset_list

def split_by_valid_ratio(data, valid_ratio):
	valid_split = int(len(data) * valid_ratio)
	valid_data 	= data[:valid_split]
	train_data 	= data[valid_split:]
	return np.array(valid_data), np.array(train_data)

# loading data
def loadTrainingData(args):
	with open(args.training_data_path, encoding='utf-8') as data_file:    
		data = json.load(data_file)["data"]

	contexts, questions, answers, ids = [], [], [], []
	for d in data:
		paragraphs = d['paragraphs']
		for paragraph in paragraphs:
			context = paragraph['context']
			qas = paragraph['qas']
			for qa in qas:
				answer = qa['answers'][0]
				question = qa['question']
				qa_id = qa['id']

				contexts.append(context)
				questions.append(question)
				answers.append(answer)
				ids.append(qa_id)
				

	# questions = cleanSentence(questions)
	# contexts = cleanSentence(contexts)
	fastConst = args.fastConst
	if args.fastMode:
		contexts  = contexts[:fastConst]
		questions  = questions[:fastConst]
		answers  = answers[:fastConst]
		ids  = ids[:fastConst]

	# randomize
	randomize = args.randomize
	if randomize:
		permutation = np.random.permutation(len(contexts))
		contexts	= np.array(contexts)[permutation]
		questions	= np.array(questions)[permutation]
		answers	= np.array(answers)[permutation]
		ids	= np.array(ids)[permutation]	

	return contexts, questions, answers, ids

def loadTestingData(args):
	with open(args.testing_data_path, encoding='utf-8') as data_file:    
		data = json.load(data_file)["data"]

	contexts, questions, ids = [], [], []
	counter = 0

	for d in data:
		paragraphs = d['paragraphs']
		for paragraph in paragraphs:
			context = paragraph['context']
			qas = paragraph['qas']
			for qa in qas:
				question = qa['question']
				qa_id = qa['id']

				contexts.append(context)
				questions.append(question)
				ids.append(qa_id)

				counter+=1
				if args.fastMode and counter>=2000:
					break

	fastConst = args.fastConst
	if args.fastMode:
		contexts  = contexts[:fastConst]
		questions  = questions[:fastConst]
		ids  = ids[:fastConst]

	# questions = cleanSentence(questions)
	# contexts = cleanSentence(contexts)
	return contexts, questions, ids

def cleanSentence(sentence):
	punctuation_list = [",", ".", "!", "?","，","。","\n",]
	for i in range(len(sentence)):
		s = sentence[i]
		for p in punctuation_list:
			s = s.replace(p, "")

		sentence[i] = s
	return sentence

def answer_convert(contexts, answers):
	answers_start = []
	answers_end   = []
	for i in range(len(contexts)):
		a = answers[i]
		answers_start.append(a['answer_start'])
		answers_end.append(a['answer_start'] + len(a['text']))
	return np.array(answers_start), np.array(answers_end)

# encoding
def one_hot_encoding(x, dim=5):
	if type(x) == list:
		return [1 if (i in x) else 0 for i in range(dim)]
	return [1 if x == i else 0 for i in range(dim)]

def one_hot_decoding(v, dim=5):
	x = 0
	for idx, value in enumerate(v):
		if value == 1:
			x = idx
			break
	return x

def answer_2_probs(answer_starts, answers_end, c_max_len):
	probs = []
	for i in range(len(answer_starts)):
		x = np.zeros((c_max_len,))
		for ptr in range(answer_starts[i], answers_end[i]):
			if ptr > c_max_len:
				print("Error answer_2_probs: answers_end shouldn't larger than cmaxlen")
			if ptr < 0:
				print("Error answer_2_probs: answers_end shouldn't small than 0")
			x[ptr] = 1
		probs.append(x)
	return np.array(probs)

def output_predicts(args, Y_str, Y_end, ids, contexts):
	print("Y_str.shape", np.array(Y_str).shape)
	print("Y_str[0]",Y_str[:20], Y_end[:20])

	Y_submit = []
	for i in range(len(Y_str)):
	    
	    ptr = Y_str[i]
	    s = str(ptr)
	    # for k in range(4):
	    while ptr < Y_end[i]-1:
	        ptr+=1
	        s += " "
	        s += str(ptr)

	    Y_submit.append(s)

	df = pd.DataFrame(data={"id":ids,"answer":Y_submit})
	df.to_csv(args.result_path, index=False, columns=["id", "answer"], encoding='utf-8')
	print("output file at: ", args.result_path)

	if not args.result_text_path == '':
	    Y_text = []
	    for i in range(len(Y_str)):
	        s = contexts[i][Y_str[i]:Y_end[i]]
	        Y_text.append(s)

	    df = pd.DataFrame(data={"id":ids,"answer":Y_text})
	    df.to_csv(args.result_text_path, index=False, columns=["id", "answer"], encoding='utf-8')
	    print("output file at: ", args.result_text_path)

def output_predicts_with_probs(args, Y_probs, ids, contexts):
	print("Y_probs.shape", np.array(Y_probs).shape)
	print("Y_probs[0]",Y_probs[:20])

	Y_submit = []
	Y_text = []
	for i in range(len(Y_probs)):
		Y_idx = [idx for idx in range(len(Y_probs[i])) if Y_probs[i][idx] >= 0.5]

		s = ""
		s_text = ""
		for idx in Y_idx:
			if not s == "":
				s += " "
			s += str(idx)
			s_text += contexts[i][idx]

		Y_submit.append(s)
		Y_text.append(s_text)


	df = pd.DataFrame(data={"id":ids,"answer":Y_submit})
	df.to_csv(args.result_path, index=False, columns=["id", "answer"], encoding='utf-8')
	print("output file at: ", args.result_path)

	if not args.result_text_path == '':
	    df = pd.DataFrame(data={"id":ids,"answer":Y_text})
	    df.to_csv(args.result_text_path, index=False, columns=["id", "answer"], encoding='utf-8')
	    print("output file at: ", args.result_text_path)

def _shuffle(X1, X2, Y1, Y2):
    randomize = np.arange(len(X1))
    np.random.shuffle(randomize)
    return (X1[randomize], X2[randomize], Y1[randomize], Y2[randomize])

def split_data_wrapper(X1_all, Y1_all, Y2_all, percentage):
    wrapper = Y2_all

    X1_train, Y1_train, Y2_train, wrapper, X1_valid, Y1_valid, Y2_valid, wrapper = split_data(X1_all, Y1_all, Y2_all, wrapper, percentage)

    return X1_train, Y1_train, Y2_train, X1_valid, Y1_valid, Y2_valid


def split_data(X1_all, X2_all, Y1_all, Y2_all, percentage):
    all_data_size = len(X1_all)
    valid_data_size = int(floor(all_data_size * percentage))

    X1_all, X2_all, Y1_all, Y2_all = _shuffle(X1_all, X2_all, Y1_all, Y2_all)

    X1_train, X2_train, Y1_train, Y2_train = X1_all[0:valid_data_size], X2_all[0:valid_data_size], Y1_all[0:valid_data_size], Y2_all[0:valid_data_size]
    X1_valid, X2_valid, Y1_valid, Y2_valid = X1_all[valid_data_size:], X2_all[valid_data_size:], Y1_all[valid_data_size:], Y2_all[valid_data_size:]

    return X1_train, X2_train, Y1_train, Y2_train, X1_valid, X2_valid, Y1_valid, Y2_valid
