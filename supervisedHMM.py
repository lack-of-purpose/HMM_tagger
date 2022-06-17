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

        self.supervised_train = self.text_train[:10000]
        self.supervised_tags_train = self.cut_off_words(training[:10000])
        self.supervised_words_train = self.cut_off_tags(training[:10000])
        self.unsupervised_train = self.cut_off_tags(training)[10000:15000]

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

def get_tagset(i, tagset):
    if i == 0 or i == -1:
        return ['st']
    else:
        return tagset

def viterbi(test, transition, emission, tagset, known_words):
    v_t = {}
    backpointers = {}
    tags = []
    #test.insert(0,'.')
    length = len(test)
    start_tag = 'st'
    max = 0
    for tag in tags:
        key = tag + ' ' + '.;###'
        max_prob = transition.get(key, 0)
        if max_prob > max:
            max = max_prob
    v_t[(0, start_tag, start_tag)] = 1
    for n in range(1, length+1):
        for tag1 in get_tagset(n-1, tagset):
            for tag2 in get_tagset(n, tagset):
                max = float("-Inf")
                m_tag = None
                for tag3 in get_tagset(n-2, tagset):
                    trans_key = tag2 + ' ' + tag3 + ';' + tag1
                    emis_key = test[n-1] + ' ' + tag2
                    if emission.get(emis_key, 0) != 0:
                        interim_pi = v_t.get((n-1, tag3, tag1), -10)*transition.get(trans_key, -10)*emission.get(emis_key, -10)
                        if interim_pi > max:
                            max = interim_pi
                            m_tag = tag3
                v_t[(n, tag1, tag2)] = max
                backpointers[(n, tag1, tag2)] = m_tag

    max = -float("inf")
    m_tag1 = None
    m_tag2 = None
    for tag1 in tagset:
        for tag2 in tagset:
            trans_key = '.' + ' ' + tag1 + ';' + tag2
            inter = v_t.get((length, tag1, tag2), -10)*transition.get(trans_key, -10)
            if inter > max:
                max = inter
                m_tag1 = tag1
                m_tag2 = tag2
    seq_tags = deque()
    seq_tags.append(m_tag2)
    seq_tags.append(m_tag1)
    for i, n in enumerate(range(length-2, 0, -1)):
        seq_tags.append[backpointers[(n+2, seq_tags[i+1], seq_tags[i])]]
    seq_tags.reverse()

    sentence = deque()
    for i in range(0, length):
        sentence.append(test[i] + '/' + seq_tags[i])
    sentence.append('\n')
    tagged = []
    tagged.append(' '.join(sentence))
    return tagged

def main():
    # Create lists from files
    with open(os.path.join(sys.path[0], "TEXTEN2.ptg.txt"), "r") as ff:
        english = ff.readlines()
    #english = open("TEXTEN2.ptg.txt").readlines()
    training_en, testing_en, heldout_en = data_split(english)

    with open(os.path.join(sys.path[0], "TEXTCZ2.ptg.txt"), encoding="ISO-8859-2") as f:
    #with open("TEXTCZ2.ptg.txt", encoding="ISO-8859-2") as f:
        czech = f.readlines()
    training_cz, testing_cz,  heldout_cz = data_split(czech)

    dataset_en = Dataset(training_en, testing_en, heldout_en)
    
    dataset_cz = Dataset(training_cz, testing_cz, heldout_cz)

    #supervised_model = HMMModel(dataset_en.text_train, dataset_en.words_train, dataset_en.text_held, dataset_en.tags_train, dataset_en.words_train, dataset_en.tags_held, dataset_en.words_held, 'V')

    supervised_model = HMMModel(dataset_cz.text_train, dataset_cz.words_train, dataset_cz.text_held, dataset_cz.tags_train, dataset_cz.words_train, dataset_cz.tags_held, dataset_cz.words_held, 'V')
    #unsupervised_model = HMMModel(dataset_en.supervised_train, dataset_en.unsupervised_train, dataset_en.text_held, dataset_en.supervised_tags_train, dataset_en.supervised_words_train, dataset_en.tags_held, dataset_en.words_held, 'BW')

    #unsupervised_model = HMMModel(dataset_cz.supervised_train, dataset_cz.unsupervised_train, dataset_cz.text_held, dataset_cz.supervised_tags_train, dataset_cz.supervised_words_train, dataset_cz.tags_held, dataset_cz.words_held, 'BW')

    wordtest = dataset_cz.words_test[20000:30000]
    tagtest = dataset_cz.tags_test[20000:30000]
    #bestpath1 = supervised_model.np_viterbi(wordtest)
    bestpath2 = supervised_model.pruned_viterbi(wordtest)

    #count = 0

    #for i in range(0, len(bestpath1)):
    #    if bestpath1[i] == tagtest[i]:
    #        count += 1

    #print(count/len(dataset_en.tags_test[:100]))

    count = 0

    for i in range(0, len(bestpath2)):
        if bestpath2[i] == tagtest[i]:
            count += 1

    print(count/len(dataset_cz.tags_test[20000:30000]))



    #tagged = viterbi(test, smoothed_transition_probabilities, all_emissions, tagset, known_words)
    #smoothed_emission_probabilities = smoothed_emission_probs(all_emissions, t_unigram_prob, lambdas_emission)
    
    print("end")

if __name__ == "__main__":
    main()
