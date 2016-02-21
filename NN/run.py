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
    data, header = [], None
    # with open(filename, 'rt') as csvfile:
    #     spamreader = csv.reader(csvfile, delimiter=',')
    #     if has_header:
    #         header = spamreader.next()
    #     for row in spamreader:
    #         data.append(row)
    # return (np.array(data), np.array(header))
    data = pd.read_csv(filename).as_matrix()
    return data


def save_data(filename, X, header):
    """
        Save data to file.
    """
    with open(filename, 'wt') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(header)
        spamwriter.writerows(X)

def convert_y(y, class_vec):
    """
        Binarise class vectors.
        Converts class into a binary class vector.
    """
    class_idx = { key: i for i,key in enumerate(class_vec)}
    n, K = len(y), len(class_vec)
    Y = np.zeros((n, K))
    for i in range(0, n):
        #print class_idx[y[i]]
        Y[i, class_idx[y[i]]] = 1
    return Y.astype(int)

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