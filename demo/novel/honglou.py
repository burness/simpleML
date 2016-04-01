#-*-coding:utf-8-*-
'''
Coding Just for Fun
Created by burness on 16/4/1.
'''
from util import *
# text_list = segment_novel('./data/红楼梦utf-8.txt','./dict/honglou_dict.txt')
out_model_file = './model/honglou.model'
# train_word2vec(text_list,out_model_file)
model = Word2Vec.load(out_model_file)
print '宝玉:'
for e in model.most_similar(positive=[u'宝玉'],topn=100):
    print e[0],e[1]