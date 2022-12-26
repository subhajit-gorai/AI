
# perceptron.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set

return - a list containing predicted labels for dev_set
"""

import numpy as np

def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters

    m, features = np.shape(train_set)
    # x_ = np.ones((m,1))
    # train_set_ = np.append(train_set, x_, axis=1)
    train_labels_ = np.reshape(train_labels, (m, 1))
    train_labels_ = np.where(train_labels_ == 0, -1, 1)
    # print(train_labels_)
    # print("shapes")
    # print(f" training set {np.shape(train_set)}")
    # print(f"train labels {np.shape(train_labels)}")
    # print(f"training labels {m} and features {features}")
    # print(f"learning rate {learning_rate}")
    W = np.zeros((features, 1))
    b = 0
    for i in range(max_iter):
        for j in range(m):
            instance = train_set[j, :].reshape((features, 1))
            f_ = instance * W
            f = np.sum(f_) + b
            if f > 0:
                f = 1
            else:
                f = -1
            if f != train_labels_[j, 0]:
                W = W + learning_rate * train_labels_[j, 0] * instance
                b = b + learning_rate * train_labels_[j, 0] * 1
                # print(W)
        # old imple
        #
        #
        # # #
        # f = np.matmul(train_set_, W)
        # y_out = np.where(f <= 0, 0, 1)
        # print(f"shape of y_out {np.shape(y_out)}")
        # y_diff = np.where(y_out == train_labels_, 0, train_labels_)
        # W = np.sum(W.transpose() + (train_set_ * y_diff * learning_rate), axis=0).reshape((features+1, 1))
        #
        # for j in range(m):
        #     y_out = -1
        #     if f[i, 0] > 0:
        #         y_out = 1
        #     if y_out != train_labels[i, 0]:
        #         W = W - learning_rate * train_labels[i, 0] * train_set
        # # #
    # b = W[features, 0]

    b *= 1.0
    return W.reshape(-1), b

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train perceptron model and return predicted labels of development set
    W, b = trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    print(f"shape  {W.shape} {b} ")
    W = np.reshape(W, (W.shape[0],1))
    # b = np.reshape(b, (b.shape[0],1))
    m, _ = np.shape(dev_set)
    # x_ = np.ones((m, 1))
    # dev_set_ = np.append(dev_set, x_, axis=1)
    f = np.matmul(dev_set , W) + b
    f = np.reshape(f, (m,1))
    y_pred = np.where(f <= 0, 0, 1)
    # print(f"flatten ", y_pred.flatten().tolist())
    return y_pred.flatten().tolist()

