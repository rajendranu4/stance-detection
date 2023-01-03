import errno
import math
import os
import random
import sys
from collections import defaultdict
from time import time

import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from transformers import AutoTokenizer
import translate
import sentencepiece as spm

model_type = sys.argv[1]
adversarial_attack = sys.argv[2]
perturb_perc = sys.argv[3]

if model_type == 'baseline':
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
elif model_type == 'contrastive':
    pretrained_model_tokenizer_path = r"C:\Users\rajen\Documents\GitHub\DeCLUTR\output_transformers_e6"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_tokenizer_path)
else:
    print("Model type not recognized")


def create_adversarial_spelling(samples, MAX_SEQ_LEN=512):
    """
    Swaps letters in words longer than three tokens. Adapted from:
    Reference:
        Naik, Aakanksha, Abhilasha Ravichander, Norman Sadeh, Carolyn Rose, and Graham Neubig.
        "Stress Test Evaluation for Natural Language Inference." In Proceedings of the 27th International
        Conference on Computational Linguistics, pp. 2340-2353. 2018.
    """
    def perturb_word_swap(word):
        char_ind = int(np.random.uniform(1, len(word) - 2))
        new_word = list(word)
        first_char = new_word[char_ind]
        new_word[char_ind] = new_word[char_ind + 1]
        new_word[char_ind + 1] = first_char
        new_word = "".join(new_word)
        return new_word

    def perturb_word_kb(word):
        keyboard_char_dict = {"a": ['s'], "b": ['v', 'n'], "c": ['x', 'v'], "d": ['s', 'f'], "e": ['r', 'w'],
                              "f": ['g', 'd'], "g": ['f', 'h'], "h": ['g', 'j'], "i": ['u', 'o'], "j": ['h', 'k'],
                              "k": ['j', 'l'], "l": ['k'], "m": ['n'], "n": ['m', 'b'], "o": ['i', 'p'], "p": ['o'],
                              "q": ['w'], "r": ['t', 'e'], "s": ['d', 'a'], "t": ['r', 'y'],
                              "u": ['y', 'i'], "v": ['c', 'b'], "w": ['e', 'q'], "x": ['z', 'c'], "y": ['t', 'u'],
                              "z": ['x']}

        new_word = list(word)
        acceptable_subs = []
        for ind, each_char in enumerate(new_word):
            if each_char.lower() in keyboard_char_dict.keys():
                acceptable_subs.append(ind)

        if len(acceptable_subs) == 0:
            return word

        char_ind = random.choice(acceptable_subs)

        first_char = new_word[char_ind]

        new_word[char_ind] = random.choice(keyboard_char_dict[first_char.lower()])
        final_new_word = "".join(new_word)
        return final_new_word

    def swap(sent):
        tokens = sent.split(" ")
        token_used = -1
        # we cut the sent later at a len of 100, so we don't want to have the adversarial changes after the cut
        rand_indices = random.sample(range(50 if len(tokens) > 50 else len(tokens)), (50 if len(tokens) > 50 else len(tokens)))
        counter = 0
        for rand_index in rand_indices:
            counter = counter + 1
            if counter > 8:
                break
            if len(tokens[rand_index]) > 3:
                tokens[rand_index] = perturb_word_kb(tokens[rand_index])
                token_used = rand_index
                #break


        for rand_index in rand_indices:
            if len(tokens[rand_index]) > 3 and rand_index != token_used:
                tokens[rand_index] = perturb_word_swap(tokens[rand_index])
                break

        return " ".join(tokens)

    def undo_wp(sent_wp):
        sent_redo = ""
        for index, t in enumerate(sent_wp):
            if t.startswith("##"):
                sent_redo += t[2:]
            elif index == 0:
                sent_redo += t
            else:
                sent_redo += " " + t
        return sent_redo

    def add_lost_info(sent_orig, sent_swap, orig_wp_len):

        sent_orig_len = len(tokenizer.tokenize(sent_orig))
        sent_wp = tokenizer.tokenize(sent_swap)

        additional_len = len(sent_wp) - sent_orig_len

        sent_wp = sent_wp[:orig_wp_len+additional_len]
        sent_wp = undo_wp(sent_wp)
        return sent_wp

    print("Swap letters from test set sentences.")
    for index, text, label in samples.itertuples():
        '''temp_swap = swap(sample['premise'])
        sample['premise'] = add_lost_info(sample['premise'], temp_swap,
                                          min(len(tokenizer.tokenize(sample['premise'])), MAX_SEQ_LEN-3))
        '''
        if index in perturb_indices:
            swap_text = swap(text)
            samples.at[index, 'text'] = swap_text

    return samples


def create_adversarial_negation(samples, perturb_indices, MAX_SEQ_LEN=512):
    """
    Add tautology "and false is not true" at the beginning of the hypothesis or premise
    Reference:
        Naik, Aakanksha, Abhilasha Ravichander, Norman Sadeh, Carolyn Rose, and Graham Neubig.
        "Stress Test Evaluation for Natural Language Inference." In Proceedings of the 27th
        International Conference on Computational Linguistics, pp. 2340-2353. 2018.
    """

    print("Add negation word to test set sentences.")
    for index, text, label in samples.itertuples():
        if index in perturb_indices:
            text_added_tautology = "false is not true and " + text
            samples.at[index, 'text'] = text_added_tautology

    return samples


test_set_path = r"C:\Users\rajen\Documents\GitHub\DeCLUTR\path\to\your\dataset"
test_set_file = "debateforum_test"
test_set_file_extension = ".csv"

test_set = pd.read_csv(test_set_path + '\\' + test_set_file + test_set_file_extension)
print(test_set.head())
test_set = test_set.sample(frac=1)

number_samples_to_be_perturbed = math.floor(test_set.shape[0] * int(perturb_perc) / 100)
perturb_indices = random.sample(range(test_set.shape[0]), number_samples_to_be_perturbed)

if adversarial_attack == 'spelling':
    adversarial_test_file = test_set_file + '_perturbed_spelling_' + model_type + "_" + str(perturb_perc)
    perturbed_test_set = create_adversarial_spelling(test_set, perturb_indices)
    perturbed_test_set.to_csv(test_set_path + "\\" +adversarial_test_file + test_set_file_extension)

elif adversarial_attack == 'negation':
    adversarial_test_file = test_set_file + '_perturbed_negation_' + model_type + "_" + str(perturb_perc)
    perturbed_test_set = create_adversarial_negation(test_set, perturb_indices)
    perturbed_test_set.to_csv(test_set_path + "\\" +adversarial_test_file + test_set_file_extension)

else:
    print("Unknown Adversarial Attack")