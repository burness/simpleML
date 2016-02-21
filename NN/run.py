'''
Coding Just for Fun
Created by burness on 16/2/21.
'''
from __future__ import division, print_function
import csv, os, sys
import numpy as np
import pandas as pd
from NN import NeuralNetwork
filepath = os.path.dirname(os.path.abspath(__file__))

def read_data(filename):
    """
        Read data from file.
        Will also return header if header=True
    """
    data = pd.read_csv(filename).as_matrix()
    return data


def save_data(filename, X):
    """
        Save data to file.
    """
    X.to_csv(filename)


def compute_acc(Y_hat, Y):
    """
        Compute accuracy of model predictions.
    """
    idx_hat, idx_true = np.argmax(Y_hat, axis=1), np.argmax(Y, axis=1)
    return np.sum(idx_hat == idx_true)/(float(len(idx_hat)))

def main(filename='data/iris-virginica.txt'):
    # Load data
    data = read_data('%s/%s' % (filepath, filename))

    X, y = data[:,:-1].astype(float), data[:,-1]


    class_vec = list(set(y))
    K = len(class_vec)


    Y = pd.get_dummies(y).astype(int).as_matrix()


    # Define parameters
    n = X.shape[0]
    d = X.shape[1]
    #
    # # Define layer sizes
    print(n,d,K)
    layers = [d, 5, K]

    model = NeuralNetwork(layers=layers, num_epochs=1000, learning_rate=0.10, alpha=0.9,
                          activation_func='sigmoid', epsilon=0.001, print_details=True)
    model.fit(X, Y)

    Y_hat = model.predict(X)
    accuracy = compute_acc(Y_hat, Y)
    print('Model training accuracy:\t%.2f' % (accuracy))

if __name__ == '__main__':
    main(filename='../data/iris.data')