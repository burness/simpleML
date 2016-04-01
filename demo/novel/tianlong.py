#-*-coding:utf-8-*-
'''
Coding Just for Fun
Created by burness on 16/4/1.
'''
from util import *
# text_list = segment_novel('./data/金庸小说全集下载txt/jinyong.txt')
out_model_file = './model/jinyong.model'
# train_word2vec(text_list,out_model_file)

model = Word2Vec.load(out_model_file)
print '令狐冲'
for e in model.most_similar(positive=[u'令狐冲'],topn=20):
    print e[0],e[1]