# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath
"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    word_to_tag_counts = {}
    tag_counts = {}
    max_tag = None
    max_word_to_tag = {}
    print(f"{test[0]}")
    for line in train:
        for word, tag in line:
            if word not in word_to_tag_counts:
                word_to_tag_counts[word] = {}
            if tag not in word_to_tag_counts[word]:
                word_to_tag_counts[word][tag] = 0
            word_to_tag_counts[word][tag] = word_to_tag_counts[word][tag] + 1

            if tag not in tag_counts:
                tag_counts[tag] = 0
            tag_counts[tag] = tag_counts[tag] + 1

    max_tag_count = 0
    for tag in tag_counts:
        if tag_counts[tag] >= max_tag_count:
            max_tag_count = tag_counts[tag]
            max_tag = tag

    for word in word_to_tag_counts:
        max_tag_for_word_count = 0
        for tag in word_to_tag_counts[word]:
            if max_tag_for_word_count <= word_to_tag_counts[word][tag]:
                max_tag_for_word_count = word_to_tag_counts[word][tag]
                max_word_to_tag[word] = tag

    test_tag = []
    for line in test:
        solution = []
        for word in line:
            if word in max_word_to_tag:
                tag = max_word_to_tag[word]
                solution.append((word, tag))
            else:
                solution.append((word, max_tag))
        test_tag.append(solution)

    return test_tag