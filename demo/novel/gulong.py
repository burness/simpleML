#-*-coding:utf-8-*-
'''
Coding Just for Fun
Created by burness on 16/4/1.
'''
from util import *
# text_list = segment_novel('./data/gulong.txt','gulong_dict2.txt')
out_model_file = './model/gulong2.model'
# train_word2vec(text_list,out_model_file)
model = Word2Vec.load(out_model_file)
print '沈浪:'
for e in model.most_similar(positive=[u'沈浪'],topn=30):
    print e[0],e[1]