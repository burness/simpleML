#-*-coding:utf-8-*-
'''
Coding Just for Fun
Created by burness on 16/2/21.
'''

from __future__ import division, print_function
import os
import numpy as np
filepath = os.path.dirname(os.path.abspath(__file__))

class NeuralNetwork():
    '''
        Simple implementation of a Nerual Network trained using
        Stochastic Gradient Descent with momentum.
    '''

    def __init__(self, layers, num_epochs=10000, learning_rate=0.10, alpha=0.9,
                 activation_func='sigmoid', epsilon=0.001, print_details=True):
        activation_funcs = {
            'sigmoid': self.sigmoid,
            'tanh': self.tanh,
            'linear': self.linear,
            'rectifier': self.rectifier
        }
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.layers = layers
        self.epsilon = epsilon
        self.print_details = print_details
        self.activation_func = activation_funcs[activation_func]

    def custom_print(self, text):
        '''
            Only print if printing is activated
        '''
        if self.print_details:
            print(text)

    def sigmoid(self, x):
        '''
            Activtion function for neurons.
            To be used on net_j: sigmoid(net_j)
        '''
        return 1./(1. + np.exp(-x))
    def tanh(self, x):
        '''
            Activtion function for neurons.
            To be used on net_j: sigmoid(net_j)
        '''
        return np.tanh(x)
    def linear(self, x):
        '''
            Activtion function for neurons.
            To be used on net_j: sigmoid(net_j)
        '''
        return x
    def rectifier(self, x):
        '''
            Activtion function for neurons.
            To be used on net_j: sigmoid(net_j)
        '''
        return np.max(0, x)

    def feedforward(self, x, W):
        '''
            Calculate output of network by feeding x through network.
            Outputs: y_hat = g(x)
        '''
        # Go through each Layer
        o = [x]
        for i in range(0, len(W)):
            net = np.dot(np.concatenate((o[i], [1])), W[i])
            o.append(self.activation_func(net))
        return o


    def predict(self, X):
        '''
            Predict class by using hard threshold on the network output.
        '''
        if np.prod(X.shape) > len(X):
            # X is matrix
            Y_hat = []
            for i in range(0,X.shape[0]):
                o = self.feedforward(X[i,:], self.W)
                Y_hat.append(o[len(o)-1])
            Y_hat = np.array(Y_hat)
            tmp = np.zeros(Y_hat.shape)
            idx = np.argmax(Y_hat, axis=1)
            rows = np.arange(0,Y_hat.shape[0])
            tmp[rows,idx] = 1
            return tmp.astype(int)
        else:
            o = self.feedforward(X, self.W)
            y_hat = o[len(o)-1]
            idx = np.argmax(y_hat)
            tmp = np.zeros(y_hat.shape)
            tmp[idx] = 1
            return tmp.astype(int)

    def E(self, t, y):
        '''
            Calculates error between vector t and y
        '''
        return 1./2. * np.sum( (t - y) ** 2 )

    def create_d_W_array(self, W):
        '''
            Create an array of size: size(W) containing only zeros.
        '''
        d_W = []
        for i, w in enumerate(W):
            d_W.append(np.zeros(w.shape))
        return d_W

    def backpropagate(self, W, o, t, learning_rate, alpha=0.9):
        '''
            Perform single optimization step for weights W of network from input x.
            Weights are updated in steepest descent direction using backpropagation.
        '''
        W_count = len(W)
        d_W = self.create_d_W_array(W)
        deltas = [[] for i in range(0, W_count)]
        # Iterate through all weight layers
        for l_rev, w in enumerate(reversed(W)):
            # Define parameters
            l = W_count - l_rev - 1
            n_neurons = w.shape[0]
            n_out = w.shape[1]

            # Get output
            o_in  = np.concatenate((o[l], [1]))
            o_out = o[l+1]

            # Define bool
            is_output_layer = (l == (W_count - 1))

            # Iterate through each edge going out of neuron i
            for j in range(0, n_out):
                # Define neuron output values
                o_j = o_out[j]

                # 输出层
                if is_output_layer:
                    # neuron j is in output layer
                    d_j = (t[j] - o_j)*o_j*(1-o_j)
                    deltas[l_rev].append(d_j)
                else:
                    # 隐层
                    d_j = o_j*(1-o_j)*np.sum(deltas[l_rev-1]*W[l+1][j,:])
                    deltas[l_rev].append(d_j)

                # Iterate through each neuron in layer l
                for i in range(0, n_neurons):
                    # Calculate delta_j
                    o_i = o_in[i]

                    # Compute step
                    d_w_ij = learning_rate*d_j*o_i + alpha * d_W[l][i,j]

                    # Update weight
                    w[i,j] = w[i,j] + d_w_ij
            deltas[l_rev] = np.array(deltas[l_rev])
        return W
    # TODO: modify the backpropagate to be a faster version
    # def backpropagate_faster(self, W, o, t, learning_rate, alpha=0.9):
    #     W_count = len(W)
    #     d_W = self.create_d_W_array(W)
    #     print(len(d_W))
    #     deltas = [[] for i in range(0, W_count)]
    #     # Iterate through all weight layers
    #     delta = (o[-1]-t).reshape([o[-1].shape[0],1])
    #     for l_rev, w in enumerate(reversed(W)):
    #
    #         # Define parameters
    #         w = w.reshape()
    #         print(l_rev)
    #         l = W_count - l_rev - 1
    #         n_neurons = w.shape[0]
    #         n_out = w.shape[1]
    #
    #         # Get output
    #         o[l]  = np.concatenate((o[l], [1]))
    #         o_out = o[l+1]
    #         print(delta.shape)
    #         print(o[l].shape)
    #         print(l_rev,l,W_count)
    #         print(w.shape)
    #         # Define bool
    #         is_output_layer = (l == (W_count - 1))
    #         if l_rev == 0:
    #             delta = (o[-1]-t).reshape([o[-1].shape[0],1])
    #             gradWl = np.dot( delta,o[l].reshape(o[l].shape[0],1).T)
    #             deltas[l_rev] = deltas[l_rev].append(delta)
    #         else:
    #             delta = np.dot(w,delta)
    #             gradWl = np.dot( delta,o[l].reshape(o[l].shape[0],1).T)
    #             deltas[l_rev] = deltas[l_rev].append(delta)
    #
    #         print(gradWl.shape)
    #     w += learning_rate*gradWl.T
    #     return W




    def shuffle_data(self, X,Y):
        '''
            Shuffle training data.
        '''
        rnd_idx = np.arange(0, len(Y))
        np.random.shuffle(rnd_idx)
        return X[rnd_idx,:], Y[rnd_idx]

    def fit(self, X, Y):
        '''
            Learn network weights from data.
            Returns training error for each epoch as np.array.
        '''
        n = X.shape[0]
        # Initialize weights
        W = []
        for i in range(0,len(self.layers)-1):
            # Add weights for the imaginary neuron (threshold theta)
            # Hence the N[i]+1
            w = np.random.rand(self.layers[i]+1, self.layers[i+1]) - 0.5
            W.append(w)

        # Iterate through every epoch
        converged = False
        MSEs = []
        for epoch in range(0, self.num_epochs):
            # Iterate through all data points
            error_vals = []

            # Shuffle data points
            X, Y = self.shuffle_data(X,Y)
            for i in range(0, n):
                x, y = X[i,:], Y[i]

                # Compute output of network
                o = self.feedforward(x, W)
                y_hat = o[len(o)-1]

                # Compute error
                error_vals.append(self.E(y, y_hat))

                W = self.backpropagate(W, o, y, self.learning_rate, self.alpha)
                W_faster = self.backpropagate_faster(W, o, y, self.learning_rate, self.alpha)
                print(W,W_faster)

            MSE = np.mean(np.array(error_vals))
            MSEs.append(MSE)
            self.custom_print('epoch: %d\tMSE: %.8f' % (epoch, MSE))
            if MSE < self.epsilon and epoch > 2: # Force atleast 2 epochs to be run
                converged = True
                break
        if converged:
            self.custom_print('Learning converged after %d epochs with an in-sample (training) MSE of: %.8f' % (epoch + 1, MSE))
        else:
            self.custom_print('Learning completed maximum number of epochs of %d with a final in-sample (training) MSE of %.8f' % (epoch + 1, MSE))
        self.W = W
        return np.array(MSEs)