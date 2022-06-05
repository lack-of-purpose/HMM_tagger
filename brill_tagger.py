import math
import pathlib
import random
import os
import sys
import numpy as np
import nltk
from nltk.tag import brill 
from nltk.tag import brill_trainer
from nltk.tag import DefaultTagger
from nltk.corpus import treebank

def brill_tagger(tag, train):
    brill.Template._cleartemplates()
    '''
    templates = [
        brill.Template(brill.Pos([-1])), 
        brill.Template(brill.Pos([-1]), brill.Word([0])),
        brill.Template(brill.Pos([-1]), brill.Pos([1])),
        brill.Template(brill.Pos([1, 2])),
        brill.Template(brill.Word([1])),
        brill.Template(brill.Word([-2, -1])),
        brill.Template(brill.Word([1, 2])),
        brill.Template(brill.Word([-1]), brill.Word([1])),
        ]
    '''
    templates = [
        brill.Template(brill.Pos([-1]), brill.Word([-1])), 
        brill.Template(brill.Pos([1]), brill.Word([1])),
        brill.Template(brill.Pos([-2, -1]), brill.Word([-2, -1])),
        brill.Template(brill.Pos([1, 2]), brill.Word([1, 2])),
        brill.Template(brill.Pos([-1, 1]), brill.Word([-1, 1])),
        brill.Template(brill.Pos([-1, 2]), brill.Word([-1, 2])),
        brill.Template(brill.Pos([-2, 1]), brill.Word([-2, 1])),
        brill.Template(brill.Pos([-3, -2, -1]), brill.Word([-3, -2, -1])),
        brill.Template(brill.Word([0])),
        ]
    f = brill_trainer.BrillTaggerTrainer(tag, templates, deterministic = True)

    return f.train(train, max_rules=10)

def first_split(text):
    size = len(text)
    trainingSize = size - 20000 - 40000
    Training = text[:trainingSize]
    Test = text[-40000:]
    Heldout = text[-60000:-40000]

    return Training, Test, Heldout

def second_split(text):
    size = len(text)
    trainingSize = size - 20000 - 40000
    Training = text[-trainingSize:]
    Test = text[:40000]
    Heldout = text[40000:60000]

    return Training, Test, Heldout

def third_split(text):
    size = len(text)
    trainingSize = size - 20000 - 40000
    Training = text[40000:trainingSize]
    Test = text[:40000]
    Heldout = text[-20000:]

    return Training, Test, Heldout

def fourth_split(text):
    size = len(text)
    trainingSize = size - 20000 - 40000
    Training = text[50000:trainingSize]
    Test = text[10000:50000]
    Heldout = text[-10000:10000]

    return Training, Test, Heldout

def fifth_split(text):
    size = len(text)
    trainingSize = size - 20000 - 40000
    Training = text[55000:trainingSize]
    Test = text[15000:55000]
    Heldout = text[-5000:15000]

    return Training, Test, Heldout

def data_preparation(data):
    prep_data = []
    temp = []
    for line in data:
        word_tag = line.split('/')
        word = word_tag[0]
        tag = word_tag[1].strip()
        word_and_tag = (word, tag)
        temp.append(word_and_tag)
        if word == '.' or word == '!' or word == '?':
            prep_data.append(temp)
            temp = []

    return prep_data

def main():
    # Create lists from files
    #English = open("TEXTEN2.ptg.txt").readlines()
    with open(os.path.join(sys.path[0], "TEXTEN2.ptg.txt"), "r") as ff:
        English = ff.readlines()
    size = len(English)
    trainingSize = size - 20000 - 40000
    TrainingEN = English[:trainingSize]
    TestEN = English[-40000:]
    HeldoutEN = English[-60000:-40000]

    with open(os.path.join(sys.path[0], "TEXTCZ2.ptg.txt"), encoding="ISO-8859-2") as f:
    #with open("TEXTCZ2.ptg.txt", encoding="ISO-8859-2") as f:
        Czech = f.readlines()
    size = len(Czech)
    trainingSize = size - 20000 - 40000
    TrainingCZ = Czech[:trainingSize]
    TestCZ = Czech[-40000:]
    HeldoutCZ = Czech[-60000:-40000]

    training_en_1, testing_en_1, heldout_en_1 = first_split(English)
    training_en_2, testing_en_2, heldout_en_2 = second_split(English)
    training_en_3, testing_en_3, heldout_en_3 = third_split(English)
    training_en_4, testing_en_4, heldout_en_4 = fourth_split(English)
    training_en_5, testing_en_5, heldout_en_5 = fifth_split(English)

    training_cz_1, testing_cz_1, heldout_cz_1 = first_split(Czech)
    training_cz_2, testing_cz_2, heldout_cz_2 = second_split(Czech)
    training_cz_3, testing_cz_3, heldout_cz_3 = third_split(Czech)
    training_cz_4, testing_cz_4, heldout_cz_4 = fourth_split(Czech)
    training_cz_5, testing_cz_5, heldout_cz_5 = fifth_split(Czech)


    train_en_1 = data_preparation(training_en_1)
    train_en_2 = data_preparation(training_en_2)
    train_en_3 = data_preparation(training_en_3)
    train_en_4 = data_preparation(training_en_4)
    train_en_5 = data_preparation(training_en_5)

    test_en_1 = data_preparation(testing_en_1)
    test_en_2 = data_preparation(testing_en_2)
    test_en_3 = data_preparation(testing_en_3)
    test_en_4 = data_preparation(testing_en_4)
    test_en_5 = data_preparation(testing_en_5)

    train_cz_1 = data_preparation(training_cz_1)
    train_cz_2 = data_preparation(training_cz_2)
    train_cz_3 = data_preparation(training_cz_3)
    train_cz_4 = data_preparation(training_cz_4)
    train_cz_5 = data_preparation(training_cz_5)

    test_cz_1 = data_preparation(testing_cz_1)
    test_cz_2 = data_preparation(testing_cz_2)
    test_cz_3 = data_preparation(testing_cz_3)
    test_cz_4 = data_preparation(testing_cz_4)
    test_cz_5 = data_preparation(testing_cz_5)

    tagger = DefaultTagger('NN')

    #treebank_train = treebank.tagged_sents()[:1000]
    #treebank_test = treebank.tagged_sents()[1000:]
    
    #tagged_training_data = treebank.tagged_sents(fileids=TrainingEN)

    #brill = brill_tagger(tag, train_en)

    #english

    '''
    brill_en_1 = brill_tagger(tagger, train_en_1)
    print('rules')
    print(brill_en_1.rules()[1:5])
    print('stats')
    print(brill_en_1.train_stats())
    print('statistics')
    brill_en_1.print_template_statistics(printunused=False)
    eval_en_1 = brill_en_1.accuracy(test_en_1)
    print ("Accuracy of english brill tag 1 : ", eval_en_1)

    brill_cz_1 = brill_tagger(tagger, train_cz_1)
    print('rules')
    print(brill_cz_1.rules()[1:5])
    print('stats')
    print(brill_cz_1.train_stats())
    print('statistics')
    brill_cz_1.print_template_statistics(printunused=False)
    eval_cz_1 = brill_cz_1.accuracy(test_cz_1)
    print ("Accuracy of czech brill tag 1 : ", eval_cz_1)
    '''

    brill_en_5 = brill_tagger(tagger, train_en_5)
    print('rules')
    print(brill_en_5.rules()[1:5])
    print('stats')
    print(brill_en_5.train_stats())
    print('statistics')
    brill_en_5.print_template_statistics(printunused=False)
    eval_en_5 = brill_en_5.accuracy(test_en_5)
    print ("Accuracy of english brill tag 1 : ", eval_en_5)

    brill_en_2 = brill_tagger(tagger, train_en_2)
    eval_en_2 = brill_en_2.evaluate(test_en_2)
    print ("Accuracy of english brill tag 2 : ", eval_en_2)

    brill_en_3 = brill_tagger(tagger, train_en_3)
    eval_en_3 = brill_en_3.evaluate(test_en_3)
    print ("Accuracy of english brill tag 3 : ", eval_en_3)

    brill_en_4 = brill_tagger(tagger, train_en_4)
    eval_en_4 = brill_en_4.evaluate(test_en_4)
    print ("Accuracy of english brill tag 4 : ", eval_en_4)

    brill_en_5 = brill_tagger(tagger, train_en_5)
    eval_en_5 = brill_en_5.evaluate(test_en_5)
    print ("Accuracy of english brill tag 5 : ", eval_en_5)

    #czech
    #brill_cz_1 = brill_tagger(tagger, train_cz_1)
    #eval_cz_1 = brill_cz_1.evaluate(test_cz_1)
    #print ("Accuracy of czech brill tag 1 : ", eval_cz_1)

    brill_cz_2 = brill_tagger(tagger, train_cz_2)
    eval_cz_2 = brill_cz_2.evaluate(test_cz_2)
    print ("Accuracy of czech brill tag 2 : ", eval_cz_2)

    brill_cz_3 = brill_tagger(tagger, train_cz_3)
    eval_cz_3 = brill_cz_3.evaluate(test_cz_3)
    print ("Accuracy of czech brill tag 3 : ", eval_cz_3)
 
    brill_cz_4 = brill_tagger(tagger, train_cz_4)
    eval_cz_4 = brill_cz_4.evaluate(test_cz_4)
    print ("Accuracy of czech brill tag 4 : ", eval_cz_4)

    brill_cz_5 = brill_tagger(tagger, train_cz_5)
    eval_cz_5 = brill_cz_5.evaluate(test_cz_5)
    print ("Accuracy of czech brill tag 5 : ", eval_cz_5)
    
    
    #print(brill_en.rules)

if __name__ == "__main__":
    main()