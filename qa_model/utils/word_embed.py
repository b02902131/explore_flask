import numpy as np
import pandas as pd

import jieba, gensim
import os

mod_dir = os.path.dirname(__file__)
print("mod_dir: ", mod_dir)

try:
	# jieba custom setting.
	jieba.set_dictionary(os.path.join(mod_dir, '../jieba_dict/dict.txt.big'))

	# load stopwords set
	# stopwordset = set()
	# with open('jieba_dict/stopwords.txt','r',encoding='utf-8') as sw:
	# 	for line in sw:
	# 		stopwordset.add(line.strip('\n'))
except:
	print("try to load jieba, but failed")

def load_char2idx():
	char2idx = np.load(os.path.join(mod_dir, "../word_vec/char2idx.npy")).item()
	idx2char = np.load(os.path.join(mod_dir, "../word_vec/idx2char.npy"))
	return char2idx, idx2char

def addQuestionWord(word2idx):
	wordset = [
	"哪個人", "哪一個人", "何人", "誰", \
	"哪個原因", "原因", "為什麼", \
	"哪一年", "何時", "哪一天", \
	"何處", "何地", "哪條", "哪ㄧ區", "哪邊", \
	"哪一個", "哪一種", "為", \
	"幾", "哪個", "為何", \
	"何", "何種", "什麼", "哪一條", "哪", \
	"哪一", "為?", "第幾", "幾個", "多少", "？"]
	for word in wordset:
		if not word in word2idx:
			idx = len(word2idx)
			word2idx[word] = idx
	return word2idx

def load_gensim(gensim_path="med250.model.bin", wv_path="wv_20171213_152540_.npy", wv_only=True):
	print("... load gensim")
	try:
		word2idx = np.load(os.path.join(mod_dir,"../word_vec/word2idx.npy")).item()
		embeddings_matrix = np.load(os.path.join(mod_dir,"../word_vec/embeddings_matrix.npy"))
		gensim_model = []
		print("... loaded from .npy file")
	except:
		print("... .npy files do not exist, load from model")
		word2idx = {"_PAD": 0}
		word2idx = addQuestionWord(word2idx)
		# print("word2idx preload word...", word2idx)
		nb_preserve_word = len(word2idx)
		if wv_only:
			print("load wv at: ", wv_path)
			wv = np.load(wv_path).item()

			vocab_list = [(k, wv[k]) for k, v in wv.vocab.items()]
			embeddings_matrix = np.zeros((len(wv.vocab.items()) + nb_preserve_word, len(wv[wv.index2word[0]])))


		else:
			print("load gensim at: ", gensim_path)
			gensim_model = gensim.models.Word2Vec.load(gensim_path)

			vocab_list = [(k, gensim_model.wv[k]) for k, v in gensim_model.wv.vocab.items()]
			embeddings_matrix = np.zeros((len(gensim_model.wv.vocab.items()) + nb_preserve_word, gensim_model.vector_size))

		for i in range(len(vocab_list)):
			word = vocab_list[i][0]
			if not word in word2idx:
			    idx = len(word2idx)
			    word2idx[word] = idx
			    embeddings_matrix[idx] = vocab_list[i][1]
			else:
				idx = word2idx[word]
				embeddings_matrix[idx] = vocab_list[i][1]

		print("check len, ", len(word2idx), len(embeddings_matrix))

		np.save("word_vec/word2idx.npy", word2idx)
		np.save("word_vec/embeddings_matrix.npy", embeddings_matrix)

	return word2idx, embeddings_matrix

def sentence_char_encode(jieba_sentences, char2idx, max_len, char_max_len):
	x = []
	for jieba_sentence in jieba_sentences:
		seqs = []
		for i in range(max_len):
			if i < len(jieba_sentence):
				char_seqs = []
				for k in range(char_max_len):
					if k < len(jieba_sentence[i]) and jieba_sentence[i][k] in char2idx:
						char_seqs.append(char2idx[jieba_sentence[i][k]])
					else:
						char_seqs.append(char2idx[" "])
				seqs.append(char_seqs)
			else:
				char_seqs = []
				for k in range(char_max_len):
					char_seqs.append(char2idx[" "])
				seqs.append(char_seqs)
		x.append(seqs)
	return np.array(x)

def sentence_encode_wordLevel(sentences, jieba_sentences, word2idx, max_len):
	x = []
	for idx, jieba_sentence in enumerate(jieba_sentences):
		seqs = []
		for i in range(len(jieba_sentence)):
			if jieba_sentence[i] in word2idx:
				seqs.append(word2idx[jieba_sentence[i]])
			else:
				seqs.append(word2idx["_PAD"])
			for k in range(len(jieba_sentence[i])-1):	# n char word append n-1 zeors
					seqs.append(word2idx["_PAD"])
		while len(seqs) < max_len:
			seqs.append(word2idx["_PAD"])
		# if len(seqs) > max_len:
		# 	print(sentences[idx])
		# 	print("idx", idx)
		# 	print("len(seqs)",len(seqs))
		# 	print("len(sentence)", len(sentences[idx]))
		x.append(np.array(seqs))
	return np.array(x)

def sentence_encode_charLevel(sentences, char2idx, max_len):
	x = []
	for sentence in sentences:
		seqs = []
		for i in range(max_len):
			if i < len(sentence) and sentence[i] in char2idx:
				seqs.append(char2idx[sentence[i]])
			else:
				seqs.append(char2idx[" "])
		x.append(seqs)
	return np.array(x)

def sentence_embedding(jieba_sentences, word2idx, max_len, embeddings_matrix):
	x = []
	for jieba_sentence in jieba_sentences:
		seqs = []
		for i in range(max_len):
			if i < len(jieba_sentence) and jieba_sentence[i] in word2idx:
				seqs.append(embeddings_matrix[word2idx[jieba_sentence[i]]])
			else:
				seqs.append(embeddings_matrix[word2idx["_PAD"]])
		x.append(seqs)
	return np.array(x)

def sentence_encode(jieba_sentences, word2idx, max_len, embedding_dim):
	x = []
	for jieba_sentence in jieba_sentences:
		seqs = []
		for i in range(max_len):
			if i < len(jieba_sentence) and jieba_sentence[i] in word2idx:
				seqs.append(word2idx[jieba_sentence[i]])	# a int index
			else:
				seqs.append(word2idx["_PAD"])
		x.append(seqs)
	return np.array(x)

def answer_decode(contexts, jieba_contexts, jieba_idx):
	idx = []
	for i in range(len(contexts)):
		c = contexts[i]
		j_c = jieba_contexts[i]
		j_idx = jieba_idx[i]
		counter = 0
		for k in range(j_idx):
			if k < len(j_c)-1:
				counter += len(j_c[k])
		idx.append(counter)
	return np.array(idx)

def answer_encode(contexts, jieba_contexts, answers):
	answers_start = []
	answers_end   = []
	for i in range(len(contexts)):
		a = answers[i]	# format like: {'answer_start': 139, 'text': '2200'}
		ans_str = a['answer_start']
		ans_end = a['answer_start'] + len(a['text'])
		q = contexts[i]
		jq = jieba_contexts[i]
		counter = 0
		idx_str = 0
		idx_end = 0
		while counter<ans_end:
			counter+=len(jq[idx_end])
			idx_end+=1

			if counter<=ans_str:
				idx_str=idx_end

		answers_start.append(idx_str)
		answers_end.append(idx_end)
		# print(counter,jq[idx_str:idx_end],a['text'])

	return np.array(answers_start), np.array(answers_end)

if __name__ == "__main__":
	pass
