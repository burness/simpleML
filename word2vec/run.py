'''
Coding Just for Fun
Created by burness on 16/3/6.
'''
from data_utils import *
from w2v import w2v
random.seed(314)
data = StanfordSentiment()
tokens = data.tokens()
nWords = len(tokens)
dimVectors = 10

C = 5
word2vec_model = w2v(data, C=C)

random.seed(2016)
np.random.seed(2016)
wordVectors = np.concatenate(((np.random.rand(nWords, dimVectors) - .5) / dimVectors,
                              np.zeros((nWords, dimVectors))), axis=0)
wordVectors0 = word2vec_model.sgd(lambda vec: word2vec_model.word2vec_sgd_wrapper(word2vec_model.cbow, tokens, vec, data, C, word2vec_model.softmax_cost_grad),
                   wordVectors, 0.3, 2000, None, True, PRINT_EVERY=10)

wordVectors = (wordVectors0[:nWords,:] + wordVectors0[nWords:,:])

print "\n=== For autograder ==="
checkWords = ["the", "a", "an", "movie", "ordinary", "but", "and"]
checkIdx = [tokens[word] for word in checkWords]
checkVecs = wordVectors[checkIdx, :]
print checkVecs