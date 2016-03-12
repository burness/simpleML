'''
Coding Just for Fun
Created by burness on 16/3/6.
'''
import numpy as np
import glob
import os.path as op
import cPickle as pickle
import random





class w2v():
    '''
    Author by Burness Duan
    '''
    def __init__(self, dataset, C):
        self.version = 1.0
        self.SAVE_PARAMS_EVERY = 1000
        self.dataset = dataset

    def sigmoid(self, x):
        x = 1.0 / (1.0+np.exp(-x))
        return x

    def softmax(self, x):
        N = x.shape[0]
        x -= np.max(x, axis=1).reshape(N, 1)
        x = np.exp(x) / np.sum(np.exp(x), axis=1).reshape(N, 1)
        return x

    def softmax_cost_grad(self, predicted, target, output_vec):
        '''
        :param predicted: predicted word vec
        :param target: the index of the target word
        :param output_vec:
        :return:
        '''
        V,D = output_vec.shape
        scores = self.softmax(output_vec.dot(predicted).reshape(1,V)).reshape(V,)
        cost = -np.log(scores[target])

        # one hot vec
        labels = np.zeros(V)
        labels[target] = 1
        diff_scores = scores - labels
        grad_pred = diff_scores.dot(output_vec)
        grad = diff_scores.reshape(V, 1).dot(predicted.reshape(D,1).T)

        return cost, grad_pred, grad

    def negsampling_cost_grad(self, predicted, target, output_vec, K=10):
        '''
        :param predicted:
        :param target:
        :param output_vec:
        :param K:
        :return:
        '''
        sample_indexs = []
        for i in xrange(10):
            index = self.dataset.sampleTokenIdx()
            sample_indexs.append(index)
        sample_vecs = output_vec[sample_indexs,:]
        w_r_out = self.sigmoid(output_vec[target].dot(predicted))
        w_r_k = self.sigmoid(-sample_vecs.dot(predicted))

        cost = -np.log(w_r_out) - np.sum(np.log(w_r_k))
        grad_pred = output_vec[target]*(w_r_out - 1) + (1-w_r_k).dot(sample_vecs)
        grad = np.zeros(output_vec.shape)
        for i in xrange(K):
            grad[sample_indexs[i]] += predicted*(1-w_r_k)[i]

        return cost, grad_pred, grad

    def skip_gram(self, current_word, C, context_words, tokens, input_vecs, output_vecs, word2vec_cost_grad = softmax_cost_grad):
        '''
        :param current_word:
        :param C:
        :param context_words:
        :param tokens:
        :param input_vecs:
        :param output_vecs:
        :param word2vec_cost_grad:
        :return:
        '''
        current_index = tokens[current_word]
        current_vec = input_vecs[current_index]
        cost = 0
        grad_in = np.zeros(input_vecs.shape)
        grad_out = np.zeros(output_vecs.shape)

        for context_word in context_words:
            target = tokens[context_word]
            curr_cost, curr_grad_in, curr_grad_out = word2vec_cost_grad(current_vec, target,output_vecs)
            cost += curr_cost
            grad_in[current_index] += curr_grad_in
            grad_out += curr_grad_out

        return cost, grad_in, grad_out


    def cbow(self, current_word, C, context_words, tokens, input_vecs, output_vecs, word2vec_cost_grad = softmax_cost_grad):
        cost = 0.0
        grad_in = 0.0
        grad_out = 0.0
        return cost, grad_in, grad_out

    def normalize(self, x):
        '''
        :param x:
        :return:
        '''
        N = x.shape[0]
        x /= np.sqrt(np.sum(x**2, axis=1)).reshape(N,1)
        return x

    def load_saved_params(self):
        st = 0
        for f in glob.glob("saved_params_*.npy"):
            iter = int(op.splitext(op.basename(f))[0].split('_')[2])
            if (iter > st):
                st =iter

        if st > 0:
            with open('saved_params_%d.npy' % st, 'r') as f:
                params =pickle.load(f)
                state = pickle.load(f)
            return st, params, state
        else:
            return st, None, None

    def save_params(self, iter, params):
        with open('saved_parmas_%d.npy' % iter, 'w') as f:
            pickle.dump(params, f)
            pickle.dump(random.getstate(), f)

    def sgd(self, f, x0, step, iterations, postprocessing = None, use_saved = False, PRINT_EVERY=10):
        ANNEAL_EVERY = 20000

        if use_saved:
            start_iter, oldx, state = self.load_saved_params()

            if start_iter > 0:
                x0 = oldx
                step *= 0.5 ** (start_iter / ANNEAL_EVERY)

            if state:
                random.setstate(state)
        else:
            start_iter = 0

        x = x0

        if not postprocessing:
            postprocessing = lambda x: x
        expcost = None

        for iter in xrange(start_iter+1, iterations+1):
            cost, grad = f(x)
            x = x - step*grad
            x = postprocessing(x)

            if iter % PRINT_EVERY == 0:
                print 'Iteration ' + str(iter) + '. Cost = '+str(cost)

            if iter % self.SAVE_PARAMS_EVERY == 0 and use_saved:
                self.save_params(iter, x)

            if iter % ANNEAL_EVERY == 0:
                step *= 0.5
        return x




    def word2vec_sgd_wrapper(self, word2vec_model, tokens, word_vecs, dataset, C, word2vec_cost_grad = softmax_cost_grad):
        batch_size = 50
        cost = 0.0
        grad = np.zeros(word_vecs.shape)
        N = word_vecs.shape[0]
        input_vecs = word_vecs[:N/2,:]
        output_vecs = word_vecs[N/2:,:]
        for i in xrange(batch_size):
            C1 = random.randint(1, C)
            # print C1
            # cent_word, context = self.get_random_context(C1)
            cent_word, context = dataset.getRandomContext(C1)

            if word2vec_model == self.skip_gram:
                denom = 1
            else:
                denom = 1

            c, gin, gout = word2vec_model(cent_word, C1, context, tokens, input_vecs, output_vecs, word2vec_cost_grad)
            cost += c/batch_size/denom
            grad[:N/2,:] += gin/batch_size/denom
            grad[N/2:,:] += gout / batch_size /denom
            return cost, grad