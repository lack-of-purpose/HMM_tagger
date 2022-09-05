#Brill Tagger 
brill_tagger.py file - run to get tagging accuracies for 5 different splits of English and Czech texts using Brill tagger from NLTK.

#HMM Tagger 
HMMTagger.py file - main file for HMM tagger. Contains calls of supervised and unsupervised taggers for both languages in main function.

Functions 'supervised_model' and 'unsupervised_model' with parameters:
- dataset (english or czech)
- beginning of test sequence (0 to start from the very beginning)
- end of test sequence (maximum 40000), better to choose small value for Czech text for faster execution
- language ('EN' or 'CZ')

#HMM Model
HMMModel.py file contains computation of transition and emission models, smoothing, implementations of Viterbi and Baum-Welch.