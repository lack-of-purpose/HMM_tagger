import os
import sys
import nltk
from nltk.tag import brill 
from nltk.tag import brill_trainer
from nltk.tag import DefaultTagger
from nltk.corpus import treebank

def brill_tagger(tag, train):
    brill.Template._cleartemplates()
    # Create templates for tagger
    templates = [
        brill.Template(brill.Word([0])),
        brill.Template(brill.Pos([-1]), brill.Word([-1])), 
        brill.Template(brill.Pos([1]), brill.Word([1])),
        brill.Template(brill.Pos([-2]), brill.Word([-2])),
        brill.Template(brill.Pos([-2, -1]), brill.Word([-2, -1])),
        brill.Template(brill.Pos([2]), brill.Word([2])),
        brill.Template(brill.Pos([1, 2]), brill.Word([1, 2])),
        brill.Template(brill.Pos([-1, 1]), brill.Word([-1, 1])),
        brill.Template(brill.Pos([-1, 2]), brill.Word([-1, 2])),
        brill.Template(brill.Pos([-2, 1]), brill.Word([-2, 1])),
        brill.Template(brill.Pos([-3, -2, -1]), brill.Word([-3, -2, -1])),
        ]
    # Create Brill tagger
    f = brill_trainer.BrillTaggerTrainer(tag, templates, deterministic = True)

    # Return trained Brill tagger
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

def call_brill_tagger(tagger, language, number_of_split, train, test):
    # Train Brill tagger
    brill = brill_tagger(tagger, train)

    # Print some statistics
    print('rules')
    print(brill.rules()[1:5])
    print('stats')
    print(brill.train_stats())
    print('statistics')
    brill.print_template_statistics(printunused=False)

    # Evaluate Brill tagger on test set and print accuracy
    eval = brill.accuracy(test)
    print(f'Accuracy of {language} brill tagger {number_of_split} :', eval)

def main():
    # Create lists from files
    with open(os.path.join(sys.path[0], "TEXTEN2.ptg.txt"), "r") as ff:
        english = ff.readlines()
    size = len(english)
    training_size = size - 20000 - 40000
    training_en = english[:training_size]
    test_en = english[-40000:]
    heldout_en = english[-60000:-40000]

    with open(os.path.join(sys.path[0], "TEXTCZ2.ptg.txt"), encoding="ISO-8859-2") as f:
        czech = f.readlines()
    size = len(czech)
    training_size = size - 20000 - 40000
    training_cz = czech[:training_size]
    test_cz = czech[-40000:]
    heldout_cz = czech[-60000:-40000]

    # 5 different splits of data
    training_en_1, testing_en_1, heldout_en_1 = first_split(english)
    training_en_2, testing_en_2, heldout_en_2 = second_split(english)
    training_en_3, testing_en_3, heldout_en_3 = third_split(english)
    training_en_4, testing_en_4, heldout_en_4 = fourth_split(english)
    training_en_5, testing_en_5, heldout_en_5 = fifth_split(english)

    training_cz_1, testing_cz_1, heldout_cz_1 = first_split(czech)
    training_cz_2, testing_cz_2, heldout_cz_2 = second_split(czech)
    training_cz_3, testing_cz_3, heldout_cz_3 = third_split(czech)
    training_cz_4, testing_cz_4, heldout_cz_4 = fourth_split(czech)
    training_cz_5, testing_cz_5, heldout_cz_5 = fifth_split(czech)

    # Prepare data for Brill tagger (split to sentences)
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

    # Initialize default tagger
    tagger = DefaultTagger('NN')

    ###  ENGLISH  ###

    call_brill_tagger(tagger, 'english', 1, train_en_1, test_en_1)
    call_brill_tagger(tagger, 'english', 2, train_en_2, test_en_2)
    call_brill_tagger(tagger, 'english', 3, train_en_3, test_en_3)
    call_brill_tagger(tagger, 'english', 4, train_en_4, test_en_4)
    call_brill_tagger(tagger, 'english', 5, train_en_5, test_en_5)

    ###  CZECH  ###

    call_brill_tagger(tagger, 'czech', 1, train_cz_1, test_cz_1)
    call_brill_tagger(tagger, 'czech', 2, train_cz_2, test_cz_2)
    call_brill_tagger(tagger, 'czech', 3, train_cz_3, test_cz_3)
    call_brill_tagger(tagger, 'czech', 4, train_cz_4, test_cz_4)
    call_brill_tagger(tagger, 'czech', 5, train_cz_5, test_cz_5)

if __name__ == "__main__":
    main()