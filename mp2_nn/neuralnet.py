# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester
# Modified by Joao Marques for the fall 2021 semester
# Modified by Kaiwen Hong for the Spring 2022 semester

"""
This is the main entry point for MP2. You should only modify code
within this file and neuralnet.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param x - an (N,D) tensor
            @param y - an (N,D) tensor
            @param l(x,y) an () tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 2 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size

        """
        super(NeuralNet, self).__init__()
        self.hidden_units = 32
        self.model = nn.Sequential(
            nn.Linear(in_size, self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units, out_size))
        self.lrate = lrate
        self.loss_fn = loss_fn
        self.optims = optim.SGD(self.model.parameters(), self.lrate, weight_decay=0.003)
        # raise NotImplementedError("You need to write this part!")

    # def set_parameters(self, params):
    #     """ Sets the parameters of your network.

    #     @param params: a list of tensors containing all parameters of the network
    #     """
    #     raise NotImplementedError("You need to write this part!")

    # def get_parameters(self):
    #     """ Gets the parameters of your network.

    #     @return params: a list of tensors containing all parameters of the network
    #     """
    #     raise NotImplementedError("You need to write this part!")

    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        return self.model(x)

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) at this timestep as a float
        """
        yhat = self.forward(x)
        loss_value = self.loss_fn(yhat, y)
        loss_value.backward()
        return loss_value.item()


def fit(train_set, train_labels, dev_set, n_iter, batch_size=100):
    """ Fit a neural net. Use the full batch size.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param n_iter: an int, the number of epoches of training
    @param batch_size: size of each batch to train on. (default 100)

    NOTE: This method _must_ work for arbitrary M and N.

    @return losses: array of total loss at the beginning and after each iteration.
            Ensure that len(losses) == n_iter.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """

    losses = []

    # preprocessing
    mean = train_set.mean()
    std = train_set.std()
    train = (train_set - mean) / std

    # creating batches
    num_batches = len(train_set) // batch_size
    train_list = [train[batch_size * y : batch_size * (y + 1)] for y in range(num_batches)]
    label_list = [train_labels[batch_size * y : batch_size * (y + 1)] for y in range(num_batches)]

    lrate = 0.009
    loss_fn = nn.CrossEntropyLoss()
    network = NeuralNet(lrate, loss_fn, len(train_set[0]), 2)


    for t in range(n_iter):
        x = train_list[t % num_batches]
        y = label_list[t % num_batches]
        # making gradient zero
        network.optims.zero_grad()

        # forward and backward step
        steps = network.step(x,y)

        #update params of the model
        network.optims.step()

        losses.append(steps)



    dev = (dev_set-mean) / std
    yi = network(dev).detach().numpy()
    yhat = np.argmax(yi, axis=1)
    return losses, yhat, network

    # raise NotImplementedError("You need to write this part!")
    # return [], [], None
