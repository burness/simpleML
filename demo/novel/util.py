'''
Coding Just for Fun
Created by burness on 16/4/1.
'''
import jieba
from gensim.models import Word2Vec
import logging
def segment_novel(novel_path, dict_path=None):
    text_list = []
    if dict_path is not None:
        jieba.load_userdict(dict_path)
    with open(novel_path, 'r') as fread:
        for line_num, line in enumerate(fread.readlines()):
            # if line_num> 100:
            #     break
            seg_list = jieba.cut(str(line),cut_all=False)
            word_list = [item for item in seg_list if len(item) > 1]
            text_list.append(word_list)
        logging.info(text_list)
    return text_list

def train_word2vec(corpora,out_model):
    w2v_model = Word2Vec(corpora)
    w2v_model.save(out_model)