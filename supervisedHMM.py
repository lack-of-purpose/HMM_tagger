from collections import deque
import math
import pathlib
import random
import os
import sys
from HMMmodel import HMMModel
from HMMModel2 import HMMModel as HMMModell

class Dataset:
    def __init__(self, training, testing, heldout):
        self.text_train = self.data_preparation(training)
        self.text_test = self.data_preparation(testing)
        self.text_held = self.data_preparation(heldout)

        self.tags_train = self.cut_off_words(training)
        self.tags_test = self.cut_off_words(testing)
        self.tags_held = self.cut_off_words(heldout)

        self.words_train = self.cut_off_tags(training)
        self.words_test = self.cut_off_tags(testing)
        self.words_held = self.cut_off_tags(heldout)

        self.supervised_train = self.text_train[:20000]
        self.supervised_tags_train = self.tags_train[:10000]
        self.supervised_words_train = self.words_train[:10000]
        self.unsupervised_train = self.cut_off_tags(training)[10000:]

    def data_preparation(self, data):
        prep_data = []
        for line in data:
            word_tag = line.split('/')
            word = word_tag[0]
            tag = word_tag[1].strip()
            prep_data.append(word)
            prep_data.append(tag) 

        return prep_data

    def cut_off_tags(self, input_list):
        new_list = []
        new_word = ""
        for word in input_list:
            for char in word:
                if char == "/":
                    break
                new_word += char
            new_list.append(new_word)
            new_word = ""
        return new_list

    def cut_off_words(self, input_list):
        new_list = []
        new_word = ""
        for word in input_list:
            word_parts = word.split('/', 1)
            new_word = word_parts[1]
            tag = new_word.strip()
            new_list.append(tag)
            new_word = ""
        return new_list

def data_split(text):
    size = len(text)
    trainingSize = size - 20000 - 40000
    training = text[:trainingSize]
    testing = text[-40000:]
    heldout = text[-60000:-40000]

    return training, testing, heldout

def supervised_model(dataset, test_size, language):
    #Create supervised HMM model
    supervised_model = HMMModel(dataset.text_train, dataset.words_train, dataset.tags_train, dataset.words_train, dataset.tags_held, dataset.words_held, language, 'V')

    wordtest = dataset.words_test[:test_size]
    tagtest = dataset.tags_test[:test_size]
    best_path = supervised_model.pruned_viterbi(wordtest)

    count = 0

    for i in range(0, len(best_path)):
        if best_path[i] == tagtest[i]:
            count += 1

    print(count/len(tagtest))

def unsupervised_model(dataset, test_size, language):
    unsupervised_model = HMMModel(dataset.supervised_train, dataset.unsupervised_train, dataset.supervised_tags_train, dataset.supervised_words_train, dataset.tags_held, dataset.words_held, language, 'BW')

    wordtest = dataset.words_test[:test_size]
    tagtest = dataset.tags_test[:test_size]
    best_path = unsupervised_model.pruned_viterbi(wordtest)

    count = 0

    for i in range(0, len(best_path)):
        if best_path[i] == tagtest[i]:
            count += 1

    print(count/len(tagtest))

def main():
    # Create lists from files
    with open(os.path.join(sys.path[0], "TEXTEN2.ptg.txt"), "r") as ff:
        english = ff.readlines()
    training_en, testing_en, heldout_en = data_split(english)

    with open(os.path.join(sys.path[0], "TEXTCZ2.ptg.txt"), encoding="ISO-8859-2") as f:
        czech = f.readlines()
    training_cz, testing_cz,  heldout_cz = data_split(czech)

    dataset_en = Dataset(training_en, testing_en, heldout_en)
    
    dataset_cz = Dataset(training_cz, testing_cz, heldout_cz)

    #English supervised model with Viterbi
    #Second parameter is a length of test data, max. 40000
    supervised_model(dataset_en, 40000, 'EN')

    #English unsupervised model with Baum-Welch and Viterbi

    unsupervised_model(dataset_en, 40000, 'EN')

    #Czech supervised model with Viterbi
    #Second parameter is a length of test data, max. 40000. Better to choose small length for czech text, because Viterbi takes ong time even after pruning
    supervised_model(dataset_cz, 1000, 'CZ')

    #Czech unsupervised model with Baum-Welch and Viterbi

    unsupervised_model(dataset_cz, 1000, 'CZ')

if __name__ == "__main__":
    main()
