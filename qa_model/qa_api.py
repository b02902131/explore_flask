# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import sys, os, re, json

from argparse import ArgumentParser

from .utils.load import loadTrainingData, loadTestingData, cleanSentence
from .utils.load import answer_convert, one_hot_encoding, one_hot_decoding
from .utils.load import output_predicts, randomize_dataset, split_by_valid_ratio
from .utils.compare_CQ import context_punctuation_vector_all_level, context_cinq_vector_level
from .utils.word_embed import load_char2idx, sentence_encode_charLevel, load_gensim, sentence_encode_wordLevel

from .models.cinq_wordEmb_model import cinq_wordEmb_start_model, cinq_wordEmb_end_model, cinq_wordEmb_model_2

from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback

import tensorflow as tf

import json
import jieba, gensim
# jieba custom setting.
mod_dir = os.path.dirname(__file__)
print("mod_dir: ", mod_dir)
jieba.set_dictionary(os.path.join(mod_dir,'jieba_dict/dict.txt.big'))

class Arguments():
    """docstring for Arguments."""
    def __init__(self):
    	# training arguments
    	self.isTeacherForcing=False
    	self.batch_size=64
    	self.nb_epoch=200
    	self.val_ratio=0.2
    	self.data_folder=os.path.join(mod_dir,'data/')
    	self.training_data_path=os.path.join(mod_dir,'data/train-v1.1.json')
    	self.wv_path=os.path.join(mod_dir,"word_vec/wv_20180118_234210_mincount300_dim100.npy")
    	self.vocab_size=20000
    	self.char_size=5493
    	self.char_emb_dim=28
    	self.embedding_trainable=False
    	self.embedding_dim=64
    	self.hidden_size=64
    	self.optimizer='adam'
    	self.loss='categorical_crossentropy'
    	self.char_level_embeddings=True
    	self.context_vector_level=4
    	self.punctuation_level=9
    	self.randomize=True
    	self.latent_dim=6
    	self.RNN_Type='gru'
    	self.word_trainable=False

    	# testing arguments
    	self.testing_data_path=os.path.join(mod_dir,'data/test-v1.1.json')

    	# put model in the same directory
    	self.load_model=os.path.join(mod_dir,'save_model/model_0119_wdim100_cdim28.hdf5')
    	self.load_model2=None
    	self.save_dir=os.path.join(mod_dir,'save_model')
    	self.model_name='model'
    	self.result_path=os.path.join(mod_dir,'result/result.csv')
    	self.result_text_path=os.path.join(mod_dir,'result/result_text.csv')
    	self.learningRate=0.001

    	self.onDeepQ=False
    	self.fastMode=False

    	self.seg_len=20
    	self.shiftting=0
    	self.enlarge=0
    	self.useJieba=False
    	self.density=2
    	self.fastConst=10
    	self.c_max_len=1160
    	self.q_max_len=90

args = Arguments()

word2idx, embeddings_matrix = load_gensim(wv_path=args.wv_path, wv_only=True)
char2idx, idx2char = load_char2idx()
args.char_size = len(char2idx)+1
args.vocab_size = len(embeddings_matrix)
args.embedding_dim = len(embeddings_matrix[0])

def initModel():
    print ('Build up model for answer start and end point...')
    model = cinq_wordEmb_model_2(args, embeddings_matrix, isTeacherForcing=False)

    path = args.load_model
    if os.path.exists(path):
        print('load model from %s' % path)
        model.load_weights(path, by_name=True)
    else:
        raise ValueError("Can't find the file %s" % path)

    return model

class myQAModel():
    def __init__(self):
        self.model = initModel()
        self.graph = tf.get_default_graph()

    def infer(self, contexts, questions, getIndex=False):
        print ('jieba cut ...')
        jieba_questions = [list(jieba.cut(question, cut_all=False)) for question in questions]
        jieba_contexts = [list(jieba.cut(context, cut_all=False)) for context in contexts]

        print ('Sentence encode --wordLevel...')
        questions_word = sentence_encode_wordLevel(questions, jieba_questions, word2idx, args.q_max_len)
        contexts_word = sentence_encode_wordLevel(contexts, jieba_contexts, word2idx, args.c_max_len)

        print ('Sentence encode --charLevel...')
        contexts_char = sentence_encode_charLevel(contexts, char2idx=char2idx, max_len=args.c_max_len)
        questions_char = sentence_encode_charLevel(questions, char2idx=char2idx, max_len=args.q_max_len)

        print ('Heuristic --comparing_CQ...')
        cinq_vector = context_cinq_vector_level(contexts, questions, args.c_max_len, args.context_vector_level)
        punctuation_vector = context_punctuation_vector_all_level(contexts, args.c_max_len)

        model = self.model

        answers_start_place_holder = np.zeros((cinq_vector.shape[0], args.c_max_len), dtype=np.float64)
        print ('Predict start point...')

        with self.graph.as_default():
            Y_pred = model.predict([
            	cinq_vector, punctuation_vector,
            	contexts_word, questions_word,
            	contexts_char, questions_char,
            	answers_start_place_holder],
            	batch_size=args.batch_size,
            	verbose=1)
        Y_str = [np.argmax(y) for y in Y_pred[0]]
        Y_end = [np.argmax(y) for y in Y_pred[1]]

        Y_text = []
        for i in range(len(Y_str)):
            s = contexts[i][Y_str[i]:Y_end[i]]
            Y_text.append(s)

        if getIndex == True:
            return Y_str, Y_end
        else:
            return Y_text

def infer(contexts, questions, getIndex=False):
    print ('jieba cut ...')
    jieba_questions = [list(jieba.cut(question, cut_all=False)) for question in questions]
    jieba_contexts = [list(jieba.cut(context, cut_all=False)) for context in contexts]

    print ('Sentence encode --wordLevel...')
    questions_word = sentence_encode_wordLevel(questions, jieba_questions, word2idx, args.q_max_len)
    contexts_word = sentence_encode_wordLevel(contexts, jieba_contexts, word2idx, args.c_max_len)

    print ('Sentence encode --charLevel...')
    contexts_char = sentence_encode_charLevel(contexts, char2idx=char2idx, max_len=args.c_max_len)
    questions_char = sentence_encode_charLevel(questions, char2idx=char2idx, max_len=args.q_max_len)

    print ('Heuristic --comparing_CQ...')
    cinq_vector = context_cinq_vector_level(contexts, questions, args.c_max_len, args.context_vector_level)
    punctuation_vector = context_punctuation_vector_all_level(contexts, args.c_max_len)

    model = initModel()

    answers_start_place_holder = np.zeros((cinq_vector.shape[0], args.c_max_len), dtype=np.float64)
    print ('Predict start point...')
    Y_pred = model.predict([
    	cinq_vector, punctuation_vector,
    	contexts_word, questions_word,
    	contexts_char, questions_char,
    	answers_start_place_holder],
    	batch_size=args.batch_size,
    	verbose=1)
    Y_str = [np.argmax(y) for y in Y_pred[0]]
    Y_end = [np.argmax(y) for y in Y_pred[1]]

    Y_text = []
    for i in range(len(Y_str)):
        s = contexts[i][Y_str[i]:Y_end[i]]
        Y_text.append(s)

    if getIndex == True:
        return Y_str, Y_end
    else:
        return Y_text


if __name__ == '__main__':
    contexts = ["小明今天吃滷肉飯","小明今天吃滷肉飯", "白沙坑文德宮，舊名保安宮，是位於臺灣彰化縣花壇鄉白沙村的土地祠，其元宵節在白沙村、文德村、長沙村的燈排遶境被列為彰化縣無形資產。",\
    "中國錢幣學會（英語：China Numismatic Society，縮寫：CNS），成立於1982年6月26日，是中華人民共和國的一個全國性錢幣學與貨幣史領域的群眾性學術團體，是由中國全國錢幣學、貨幣史及金融史研究單位及研究者，錢幣工作者，錢幣收藏者和愛好者自願結成的全國性、學術性、非營利性的社會組織，是國際錢幣學委員會的正式成員。該學會住所為北京市，其秘書處設在中國錢幣博物館。該會的業務主管單位為中國人民銀行，相關業務同時接受中國文化部和國家文物局的指導。會員分為單位會員和個人會員，會員需每年繳納會費。[1]",\
    "保山—腾冲高速公路简称保腾高速，是云南省的一条省级高速公路，编号为S10，连接保山和腾冲两座城市，起点是保龙高速上的小田坝互通，止于腾冲市中和镇毛家营村南侧岔路口，连接已建成通车的腾冲至缅甸密支那公路[2]，保山至腾冲过路费为82元[3]。保腾高速全长63.86公里，项目概算投资46亿余元，平均每公里造价7,197.21万元，双向4车道，设计时速80公里/小时[1]，路基宽度为24.5米，桥涵及构造物设计荷载等级为公路CI级[4]。全线共有特大桥两座、大桥114座、中小桥48座、隧道3座，桥隧总长占全线的29.5%，重点控制性工程是峡谷跨径世界第一、海拔高度亚洲第一的龙江大桥[5]。保腾高速较原有的89.54公里保腾二级路短22公里[6]，高速的修通，实现昆明至腾冲全程高速化、昆明到缅甸密支那全程公路高等级化[2]，对开发滇西旅游、拉动区域经济增长、建设云南面向南亚的国际大通道有重要意义[5]。高速公路的通车，也使得腾冲的房地产迎来提价峰值[7]。",\
    "江原道（朝鮮語：강원도／江原道 Gangwon do */?）是位于朝鲜半岛中东部的一个韩国道级行政区域，首府是春川市。江原道东临韩国东海，南与庆尚北道和忠清道相连，西邻京畿道，北面隔着朝韩非军事区与朝鲜江原道接壤，面积16,872平方公里，占韩国国土面积17%，人口超150万。[1]:273 江原道是韩国的旅游胜地。雪岳山国立公园、五台山国立公园、雉岳山国立公园三座韩国国立公园都位于江原道境内[2]。位于江原道春川市的南怡岛是韩剧《冬季恋歌》的拍摄地，也是韩流旅游的重要目的地之一[3]。江原道也是韩国的滑雪胜地，建有9个滑雪度假村，是1999年亚洲冬季运动会、2009年冬季两项世界锦标赛、2013年世界冬季特殊奥林匹克运动会、2018年冬季奥林匹克运动会的举办地[4]。",\
    "吉盧語（Jru' ）（國際音標：[ɟruʔ]）是屬於南亞語系的一種語言，使用人口約28000人，主要分布在寮國南部。[3]吉盧語也被稱為“Loven”、“Laven”或“Boloven”，來自寮語的Laven或Loven，而這又源自高棉語的布拉萬高原。"
    ]
    questions = ["小明今天吃了什麼?", "誰今天吃了滷肉飯?","臺灣彰化縣花壇鄉哪一座土地祠其元宵節在該鄉白沙村、文德村、長沙村的燈排遶境被列為彰化縣無形資產？",\
    "1982年成立的哪一個社會組織是中華人民共和國在錢幣學與貨幣史領域的全國性學術團體？", "雲南省的哪一條省級高速是中國第一條修建在火山區的高速公路？",\
    "韓國的哪個道是1999年亞洲冬季運動會和2018年冬季奧林匹克運動會的舉辦地？", "哪個語言屬於拉斐語的一種變體，在寮國約有28,000名使用者？"]

    Y_text = infer(contexts, questions)
    df = pd.DataFrame(data={"answer":Y_text})
    df.to_csv(os.path.join(mod_dir,"result/result_text.csv"), index=False, columns=["answer"], encoding='utf-8')
    print("output file at: ", os.path.join(mod_dir,"result/result_text.csv"))
