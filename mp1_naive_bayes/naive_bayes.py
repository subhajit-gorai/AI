# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import numpy as np
import math
from tqdm import tqdm
from collections import Counter
import reader

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""




"""
  load_data calls the provided utility to load in the dataset.
  You can modify the default values for stemming and lowercase, to improve performance when
       we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset_main(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels

pos_uni_total = 0
neg_uni_total = 0
pos_uni_unique = 0
neg_uni_unique = 0


pos_bi_total = 0
neg_bi_total = 0
pos_bi_unique = 0
neg_bi_unique = 0


def create_word_maps_uni(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: words 
        values: number of times the word appears
    neg_vocab:
        In data where labels are 0
        keys: words 
        values: number of times the word appears 
    """
    global pos_uni_total
    global neg_uni_total
    global pos_uni_unique
    global neg_uni_unique

    pos_uni_total = 0
    neg_uni_total = 0
    pos_uni_unique = 0
    neg_uni_unique = 0

    #print(len(X),'X')
    pos_vocab = {}
    neg_vocab = {}


    #TODO:
    # raise RuntimeError("Replace this line with your code!")
    length = len(y)
    vocab = {}
    for i in range(length):
        if y[i] == 1:
            vocab = pos_vocab
            pos_uni_total = pos_uni_total + len(X[i])
        else:
            vocab = neg_vocab
            neg_uni_total = neg_uni_total + len(X[i])
        words = X[i]
        for word in words:
            if word not in vocab:
                vocab[word] = 0
            vocab[word] = vocab[word] + 1
    pos_uni_unique = len(pos_vocab)
    neg_uni_unique = len(neg_vocab)
    return dict(pos_vocab), dict(neg_vocab)


def create_word_maps_bi(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: pairs of words
        values: number of times the word pair appears
    neg_vocab:
        In data where labels are 0
        keys: words 
        values: number of times the word pair appears 
    """
    global pos_bi_total
    global neg_bi_total
    global pos_bi_unique
    global neg_bi_unique

    pos_bi_total = 0
    neg_bi_total = 0
    pos_bi_unique = 0
    neg_bi_unique = 0

    #print(len(X),'X')
    pos_vocab = {}
    neg_vocab = {}
    vocab = {}
    ##TODO:
    # raise RuntimeError("Replace this line with your code!")
    length = len(y)
    for i in range(length):
        l = len(X[i])
        if y[i] == 1:
            vocab = pos_vocab
            pos_bi_total += l-1
        else:
            vocab = neg_vocab
            neg_bi_total += l-1
        wl = len(X[i])
        for j in range(0, wl-1, 1):
            w1 = X[i][j]
            # if w1 not in vocab:
            #     vocab[w1] = 0
            # vocab[w1] = vocab[w1] + 1
            # for k in range(j+1, wl, 1):
            w2 = X[i][j+1]

            # if w1 > w2:
            #     t = w1
            #     w1 = w2
            #     w2 = t

            w = w1 + " " + w2
            if w not in vocab:
                vocab[w] = 0
            vocab[w] = vocab[w] + 1

    pos_bi_unique = len(pos_vocab)
    neg_bi_unique = len(neg_vocab)

    for i in range(length):
        if y[i] == 1:
            vocab = pos_vocab
        else:
            vocab = neg_vocab
        email = X[i]
        for word in email:
            if word not in vocab:
                vocab[word] = 0
            vocab[word] = vocab[word] + 1
    return dict(pos_vocab), dict(neg_vocab)



# Keep this in the provided template
def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def get_uni_prob(email, pos_vocab, neg_vocab, laplace):
    global pos_uni_total
    global pos_uni_unique
    global neg_uni_total
    global neg_uni_unique
    pos_prob = 0
    neg_prob = 0
    for word in email:
        pos_word_count = 0
        if word in pos_vocab:
            pos_word_count = pos_vocab[word]
        pos_word_prob = (pos_word_count + laplace) / (pos_uni_total + laplace * (1 + pos_uni_unique))
        pos_prob += np.log(pos_word_prob)

        neg_word_count = 0
        if word in neg_vocab:
            neg_word_count = neg_vocab[word]
        neg_word_prob = (neg_word_count + laplace) / (neg_uni_total + laplace * (1 + neg_uni_unique))
        neg_prob += np.log(neg_word_prob)

    return pos_prob, neg_prob

def get_bi_prob(email, pos_vocab, neg_vocab, laplace):
    global pos_uni_total
    global pos_uni_unique

    global neg_uni_total
    global neg_uni_unique

    global pos_bi_total
    global pos_bi_unique
    global neg_bi_total
    global neg_bi_unique
    pos_prob = 0
    neg_prob = 0
    wl = len(email)
    for j in range(0, wl-1, 1):
        # for k in range(j+1, wl, 1):
        w1 = email[j]
        w2 = email[j+1]
        # if w1 > w2:
        #     t = w1
        #     w1 = w2
        #     w2 = t
        w = w1 + " " + w2
        pos_word_count = 0
        if w in pos_vocab:
            pos_word_count = pos_vocab[w]
        pos_word_prob = (pos_word_count + laplace) / (pos_bi_total + pos_uni_total + laplace * (1 + pos_bi_unique + pos_uni_unique))
        pos_prob += np.log(pos_word_prob)

        neg_word_count = 0
        if w in neg_vocab:
            neg_word_count = neg_vocab[w]
        neg_word_prob = (neg_word_count + laplace) / (neg_bi_total + neg_uni_total + laplace * (1 + neg_bi_unique + neg_uni_unique))
        neg_prob += np.log(neg_word_prob)
    # print(f"bigram probabilites pos = {pos_prob} and neg = {neg_prob}")
    return pos_prob, neg_prob


def naiveBayes(train_set, train_labels, dev_set, laplace=0.001, pos_prior=0.8, silently=False):
    '''
    Compute a naive Bayes unigram model from a training set; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    global pos_uni_total
    global neg_uni_total
    global pos_uni_unique
    global neg_uni_unique
    dev_labels = []
    # Keep this in the provided template
    print_paramter_vals(laplace,pos_prior)
    pos_vocab, neg_vocab = create_word_maps_uni(train_set, train_labels)
    # raise RuntimeError("Replace this line with your code!")
    for email in dev_set:
        pos_prob = np.log(pos_prior)
        neg_prob = np.log(1.0 - pos_prior)
        pos_email_prob, neg_email_prob = get_uni_prob(email, pos_vocab, neg_vocab, laplace)
        pos_prob += pos_email_prob
        neg_prob += neg_email_prob

        if pos_prob >= neg_prob:
            dev_labels.append(1)
        else:
            dev_labels.append(0)

    return dev_labels


# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.001, bigram_laplace=0.005, bigram_lambda=0.5,pos_prior=0.8,silently=False):
    '''
    Compute a unigram+bigram naive Bayes model; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    unigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    bigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating bigram probs
    bigram_lambda (scalar float) = interpolation weight for the bigram model
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    dev_labels = []
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)
    pos_vocab_uni, neg_vocab_uni = create_word_maps_uni(train_set, train_labels)
    pos_vocab_bi, neg_vocab_bi = create_word_maps_bi(train_set,train_labels)
    for email in dev_set:
        pos_prob_uni = np.log(pos_prior)
        neg_prob_uni = np.log(1.0 - pos_prior)
        pos_email_prob_uni, neg_email_prob_uni = get_uni_prob(email, pos_vocab_uni, neg_vocab_uni, unigram_laplace)

        pos_prob_uni += pos_email_prob_uni
        neg_prob_uni += neg_email_prob_uni

        pos_prob_bi = np.log(pos_prior)
        neg_prob_bi = np.log(1.0 - pos_prior)

        pos_email_prob_bi, neg_email_prob_bi = get_bi_prob(email, pos_vocab_bi, neg_vocab_bi, bigram_laplace)
        pos_prob_bi += pos_email_prob_bi
        neg_prob_bi += neg_email_prob_bi

        pos_prob = (pos_prob_uni * (1-bigram_lambda)) + (pos_prob_bi * bigram_lambda)
        neg_prob = (neg_prob_uni * (1-bigram_lambda)) + (neg_prob_bi * bigram_lambda)

        if pos_prob >= neg_prob:
            dev_labels.append(1)
        else:
            dev_labels.append(0)

    # max_vocab_size = None
    #
    # raise RuntimeError("Replace this line with your code!")
    return dev_labels



