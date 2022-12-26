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
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
"""

import numpy as np

def get_initial_transition_prob(tag2, total_lines, prob_sent_start_with_tag, count_by_tag, total_number_of_tags, laplace_lambda):
    if tag2 in prob_sent_start_with_tag:
        return prob_sent_start_with_tag[tag2]
    numerator = laplace_lambda
    denominator = total_lines + (laplace_lambda * total_number_of_tags)
    if denominator == 0:
        print(f" /0 exception get_intitial_transition_prob")
    return np.log(numerator/denominator)


def get_transition_prob(tag1, total_lines, prob_sent_start_with_tag,  tag2, transition_prob, count_by_tag, total_number_of_tags, laplace_lambda):
    if tag1 == "START":
        return get_initial_transition_prob(tag2, total_lines, prob_sent_start_with_tag, count_by_tag, total_number_of_tags, laplace_lambda)
    elif (tag1, tag2) in transition_prob:
        return transition_prob[(tag1, tag2)]
    else:
        numerator = laplace_lambda
        denominator = count_by_tag[tag1] + (total_number_of_tags * laplace_lambda)
        return np.log(numerator/denominator)

def get_emmission_prob(tag, word, emmission_given_tag, count_by_tag, total_vocabulary_words, laplace_lambda, prob_of_tag_for_hapax_word, hyper_param_hapax):
    if tag in emmission_given_tag and word in emmission_given_tag[tag]:
        return emmission_given_tag[tag][word]

    numerator = laplace_lambda
    # hyper_param_hapax = 30
    if tag in prob_of_tag_for_hapax_word:
        numerator = numerator * hyper_param_hapax * prob_of_tag_for_hapax_word[tag]
    denominator = count_by_tag[tag] + (laplace_lambda * (total_vocabulary_words + 1))
    # print(f"deno is {denominator}")
    # if denominator <= 0.00000001 or numerator <= 0.000000001:
    #     print(f" /0 exception {count_by_tag[tag] } -- {numerator} -- {denominator}")
    return np.log(numerator/denominator)


def get_the_states(dp, ind, tag, sol):
    if ind == 1:
        return

    # print(f"inside get states {ind} {tag}")
    _, prev_tag = dp[ind][tag]
    get_the_states(dp, ind-1, prev_tag, sol)
    sol.append(prev_tag)

def predict_pos_tag(line, total_lines, prob_sent_start_with_tag, tag_inverse_index, transition_prob, count_by_tag, total_number_of_tags, emmission_given_tag, total_vocabulary_words, laplace_lambda, prob_of_tag_for_hapax_word, hyper_param_hapax):
    dp = {}
    size = len(line)
    i = 0
    # print(f"{len(line)} **** the line is {line}")
    # print(f"{total_number_of_tags}")
    # print(f"{count_by_tag}")
    # print(f"{tag_inverse_index}")
    prev_tag = "None"
    for word in line:
        if word != "START" and word != "END":
            if i == 1: # first word
                for j in range(total_number_of_tags):
                    # print(f"key error {tag_inverse_index[j]}")
                    pi = get_transition_prob("START", total_lines, prob_sent_start_with_tag, tag_inverse_index[j], transition_prob, count_by_tag, total_number_of_tags, laplace_lambda)
                    emit = get_emmission_prob(tag_inverse_index[j], word, emmission_given_tag, count_by_tag, total_vocabulary_words, laplace_lambda, prob_of_tag_for_hapax_word, hyper_param_hapax)
                    if i not in dp:
                        dp[i] = {}
                    dp[i][tag_inverse_index[j]] = (pi + emit, None)
            else:
                for j in range(total_number_of_tags):
                    current_tag = tag_inverse_index[j]
                    max_val = None
                    optimal_prev_tag = None
                    for k in range(total_number_of_tags):
                        prev_tag = tag_inverse_index[k]
                        pi = get_transition_prob(prev_tag, total_lines, prob_sent_start_with_tag, current_tag, transition_prob, count_by_tag,
                                                 total_number_of_tags, laplace_lambda)
                        emit = get_emmission_prob(current_tag, word, emmission_given_tag, count_by_tag,
                                                  total_vocabulary_words, laplace_lambda, prob_of_tag_for_hapax_word, hyper_param_hapax)
                        old_sol, _ = dp[i-1][prev_tag]
                        final = pi + emit + old_sol
                        if max_val == None or max_val < final:
                            max_val = final
                            optimal_prev_tag = prev_tag

                    if i not in dp:
                        dp[i] = {}
                    dp[i][current_tag] = (max_val, optimal_prev_tag)
        prev_word = word
        i += 1
    max_val = None
    optimal_tag = None
    solution = []
    for k in range(total_number_of_tags):
        tag = tag_inverse_index[k]
        sol, _ = dp[size-2][tag]
        if max_val == None or sol > max_val:
            max_val = sol
            optimal_tag = tag
    # print(f" rthe final dp is {dp}")

    get_the_states(dp, size-2, optimal_tag, solution)
    solution.append(optimal_tag)

    result = []
    index = 0
    for word in line:
        if word == "START" or word == "END":
            result.append((word, word))
        else:
            result.append((word, solution[index]))
            index += 1

    return result

def viterbi_2(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    laplace_lambda = 0.001
    hyper_param_hapax = 600
    total_number_of_tags = 0
    count_by_tag = {}
    count_by_words = {}
    emmission_given_tag = {}
    transition_prob = {}  # (tag1, tag2) --> what's the probability
    possible_tag_for_word = {}
    prob_sent_start_with_tag = {}
    total_lines = len(train)
    prob_of_tag_for_hapax_word = {}
    total_hapax_words = 0
    first_tag = True
    for line in train:
        prev_word = "START"
        prev_tag = "START"
        first_tag = True
        for (word, tag) in line:
            # print(f"word tag pairs are {word, tag}")
            if word == "START" or word == "END":
                continue

            if first_tag:
                if tag not in prob_sent_start_with_tag:
                    prob_sent_start_with_tag[tag] = 0
                prob_sent_start_with_tag[tag] = prob_sent_start_with_tag[tag] + 1

            first_tag = False
            if tag not in count_by_tag:
                count_by_tag[tag] = 0
            count_by_tag[tag] = count_by_tag[tag] + 1
            possible_tag_for_word[word] = tag
            
            if word not in count_by_words:
                count_by_words[word] = 0    
            count_by_words[word] = count_by_words[word] + 1

            if tag not in emmission_given_tag:
                emmission_given_tag[tag] = {}
            if word not in emmission_given_tag[tag]:
                emmission_given_tag[tag][word] = 0
            emmission_given_tag[tag][word] = emmission_given_tag[tag][word] + 1

            if prev_tag != 'START':
                if (prev_tag, tag) not in transition_prob:
                    transition_prob[(prev_tag, tag)] = 0
                transition_prob[(prev_tag, tag)] = transition_prob[(prev_tag, tag)] + 1

            prev_word = word
            prev_tag = tag

    tag_inverse_index = {}
    total_vocabulary_words = len(possible_tag_for_word)
    total_number_of_tags = len(count_by_tag)
    idx = 0
    for tag in count_by_tag:
        tag_inverse_index[idx] = tag
        idx += 1

    for (prev_tag, current_tag) in transition_prob:
        numerator = transition_prob[(prev_tag, current_tag)] + laplace_lambda
        denominator = count_by_tag[prev_tag] + (total_number_of_tags * laplace_lambda)
        transition_prob[(prev_tag, current_tag)] = np.log(numerator / denominator)


    for tag in emmission_given_tag:
        for word in emmission_given_tag[tag]:
            numerator = emmission_given_tag[tag][word] + laplace_lambda
            denominator = count_by_tag[tag] + (laplace_lambda * (total_vocabulary_words + 1))
            emmission_given_tag[tag][word] = np.log(numerator / denominator)
             
    for word in count_by_words:
        if count_by_words[word] == 1:
            total_hapax_words += 1
            tag = possible_tag_for_word[word]
            if tag not in prob_of_tag_for_hapax_word:
                prob_of_tag_for_hapax_word[tag] = 0
            prob_of_tag_for_hapax_word[tag] = prob_of_tag_for_hapax_word[tag] + 1

    for tag in prob_of_tag_for_hapax_word:
        prob_of_tag_for_hapax_word[tag] = (prob_of_tag_for_hapax_word[tag]/total_hapax_words)

    # print(f" the probability of hapax words \n {prob_of_tag_for_hapax_word} \n")
    for tag in prob_sent_start_with_tag:
        numerator = prob_sent_start_with_tag[tag] + laplace_lambda
        denominator = total_lines + (laplace_lambda * total_number_of_tags)
        prob_sent_start_with_tag[tag] = np.log(numerator / denominator)

    test_tagging = []
    for line in test:
        sol = predict_pos_tag(line, total_lines, prob_sent_start_with_tag, tag_inverse_index, transition_prob,
                              count_by_tag, total_number_of_tags,
                              emmission_given_tag, total_vocabulary_words, laplace_lambda, prob_of_tag_for_hapax_word, hyper_param_hapax)
        # print(f"the line is {line}")
        # print(f"the tag are {sol}")
        test_tagging.append(sol)
        # if len(test_tagging) > 3:
        #     break

    return test_tagging