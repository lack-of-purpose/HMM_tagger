from collections import deque
import math
import pathlib
import random
import numpy as np

class HMMModel:
    _EPSYLON = 0.0001

    
    def __init__(self, training_data, heldout_data, training_data_tags, training_data_words, heldout_data_tags, heldout_data_words) -> None:
        self.tagset = set()
        for tag in training_data_tags:      
            self.tagset.add(tag)
        self.tagset_size = len(self.tagset)
        self.t_uniform_probability = 1/self.tagset_size

        self.initial_probabilities = np.full(self.tagset_size, self.t_uniform_probability, dtype=float)

        self.known_words = set()
        for word in training_data_words:      
            self.known_words.add(word)

        self.w_unigram_amount = len(training_data_words)
        self.wt_bigram_amount = self.w_unigram_amount - 1

        self.t_unigram_amount = len(training_data_tags)
        self.t_bigram_amount = self.t_unigram_amount - 1
        self.t_trigram_amount = self.t_unigram_amount - 2

        self.tag_list = list(self.tagset)
        self.add_unknown_words(training_data, training_data_words, self.known_words)
        self.word_list = list(self.known_words)

        self.tag_transition = np.zeros((self.tagset_size, self.tagset_size, self.tagset_size), dtype=float)
        self.word_emission = np.zeros((self.tagset_size, len(self.known_words)), dtype=float)

        self.all_transitions = self.generate_all_possible_transitions(self.tag_list)
        self.all_emissions = self.generate_all_possible_wordtag_pairs(self.tag_list, self.word_list)

        #n-gram counts (tags and words/tags)
        self.t_unigram_en = self.t_unigram_count(training_data_tags, self.t_unigram_amount)
        self.t_bigram_en = self.t_bigram_count(training_data_tags, self.t_bigram_amount)
        self.t_trigram_en = self.t_trigram_count(training_data_tags, self.t_trigram_amount)

        self.w_unigram_en = self.t_unigram_count(training_data_words, self.t_unigram_amount)
        self.wt_bigram_en = self.wt_bigram_count(training_data, self.wt_bigram_amount)

        # probabilities
        self.transition_probabilities = self.transition_probs(self.t_trigram_en, self.t_bigram_en)
        self.emission_probabilities = self.emission_probs(self.t_unigram_en, self.wt_bigram_en)

        self.w_unigram_prob =self.unigram_probabilities(self.w_unigram_en, self.w_unigram_amount)

        self.t_unigram_prob = self.unigram_probabilities(self.t_unigram_en, self.t_unigram_amount)

        self.t_bigram_prob = self.bigram_probabilities(self.t_unigram_en, self.t_bigram_en)

        #self.t_trigram_prob = self.trigram_probabilities(self.t_trigram_en, self.t_bigram_en)

        self.move_counts(self.all_transitions, self.transition_probabilities)
        self.move_counts(self.all_emissions, self.emission_probabilities)

        self.lambdas_transition = self.trigram_smoothing(heldout_data_tags, self._EPSYLON, self.t_unigram_prob, self.t_bigram_prob, self.transition_probabilities, self.t_uniform_probability)
        self.lambdas_emission = self.bigram_smoothing(heldout_data, self._EPSYLON, self.w_unigram_prob, self.emission_probabilities, self.t_uniform_probability)

        self.smoothed_transition_probabilities = self.smoothed_transition_probs(self.all_transitions, self.t_bigram_prob, self.t_unigram_prob, self.t_uniform_probability, self.lambdas_transition)
        self.smoothed_emission_probabilities = self.smoothed_emission_probs(self.all_emissions, self.w_unigram_prob, self.t_uniform_probability, self.lambdas_emission)

        for key in self.smoothed_transition_probabilities:
            key_split = key.split(' ')
            key_conditions = key_split[1].split(';')
            tag1 = self.tag_list.index(key_conditions[0])
            tag2 = self.tag_list.index(key_conditions[1])
            tag3 = self.tag_list.index(key_split[0])
            self.tag_transition[tag1, tag2, tag3] = self.smoothed_transition_probabilities[key]

        for key in self.smoothed_emission_probabilities:
            key_split = key.split(' ')
            tag2 = self.word_list.index(key_split[0])
            tag1 = self.tag_list.index(key_split[1])
            self.word_emission[tag1, tag2] = self.smoothed_emission_probabilities[key]

        print("End")

    def np_viterbi(self, test):
        test_sequence = np.asarray(test)
        T = test_sequence.shape[0]
        M = self.tag_transition.shape[0]
 
        #omega = np.zeros((T, M))
        #omega = np.zeros((T, M, M))
        omega = np.full((T, M), np.NINF, dtype=float)
        index = self.word_list.index(test_sequence[0])
        #omega[0, :] = np.log(self.initial_probabilities * self.word_emission[:, index])
        omega[0, :] = np.log(self.initial_probabilities * self.word_emission[:, index])

        #prev = np.zeros((T - 1, M))
        prev = np.zeros((T - 1, M))
        prev = np.full((T - 1, M), np.NINF, dtype=int)

        for t in range(1, T):
            for j in range(M):
                    # Same as Forward Probability
                    if test_sequence[t] in self.word_list:
                        index_t = self.word_list.index(test_sequence[t])
                    else:
                        index_t = self.word_list.index('UNK')
                    probability = omega[t - 1,:] + np.log(self.tag_transition[:, :, j]) + np.log(self.word_emission[j, index_t])
 
                    # This is our most probable state given previous state at time t (1)
                    #tag1 = np.argmax(probability)
                    index = np.unravel_index(probability.argmax(), probability.shape)
                    #tag1 = index[0]
                    tag2 = index[1]
                    prev[t - 1, j] = tag2
 
                    # This is the probability of the most probable state (2)
                    omega[t, j] = np.max(probability)
 
        # Path Array
        S = np.zeros(T,dtype=np.int)

        # Find the most probable last hidden state
        last_state = np.argmax(omega[T - 1, :])
 
        S[0] = last_state
 
        # Find the most probable last hidden state
        #index = np.unravel_index(omega[T - 1, :, :].argmax(), omega[T - 1, :, :].shape)
        #last_state = index[0] #np.argmax(omega[T - 1, :, :])
        #prev_last_state = index[1]

        #S[0] = last_state
        #S[1] = prev_last_state

        backtrack_index = 1
        for i in range(T - 2, -1, -1):
            S[backtrack_index] = prev[i, int(last_state)]
            last_state = prev[i, int(last_state)]
            backtrack_index += 1

        '''
        backtrack_index = 1
        for i in range(T - 2, -1, -1):
            S[backtrack_index] = prev[i, int(last_state), int(prev_last_state)]
            temp = prev_last_state
            prev_last_state = last_state
            last_state = prev[i, int(last_state), int(temp)] 
            backtrack_index += 1
        '''

        # Flip the path array since we were backtracking
        S = np.flip(S, axis=0)
 
        # Convert numeric values to actual hidden states
        result = []
        for s in S:
            result.append(self.tag_list[s])
 
        return result

    def viterbi(self, test):
        bestpath = []
        viterbi = {}
        backtrack = {}
        n = len(self.tagset)
        max = 0
        max_tag = ''
        for tag in list(self.tagset):
            key = test[0] + ' ' + tag
            viterbi[(tag, 0)] = self.t_uniform_probability*self.smoothed_emission_probabilities[key]
            #if temp > max:
                #max = temp
                #max_tag = tag
        #viterbi[(tag, 0)] = max
            #backtrack[(tag, 0)] = 'null'
            backtrack[0] = 'null'

        max = 0
        max_tag = ''
        for tag3 in list(self.tagset):
            for curr_tag in list(self.tagset):
                for prev_tag in list(self.tagset):
                    trans_key = curr_tag + ' ' + tag3 + ';' + prev_tag
                    if test[1] in self.known_words:
                        emiss_key = test[1] + ' ' + curr_tag
                    else:
                        emiss_key = 'UNK' + ' ' + curr_tag
                    temp = viterbi[(prev_tag, 0)]*self.smoothed_transition_probabilities[trans_key]*self.smoothed_emission_probabilities[emiss_key]
                    if temp > max:
                        max = temp
                        max_tag = prev_tag
                viterbi[(prev_tag, curr_tag, 1)] = max
                #backtrack[(tag3, prev_tag, 1)] = max_tag
                backtrack[1] = max_tag  # got back tags
                max = 0

        max = 0
        max_tag = ''
        temp = 0
        for i in range(2, n):
            for tag3 in list(self.tagset):
                for curr_tag in list(self.tagset):
                    for prev_tag in list(self.tagset):
                        trans_key = curr_tag + ' ' + tag3 + ';' + prev_tag
                        if test[i] in self.known_words:
                            emiss_key = test[i] + ' ' + curr_tag
                        else:
                            emiss_key = 'UNK' + ' ' + curr_tag
                        if (tag3, prev_tag, i-1) in viterbi:
                            temp = viterbi[(tag3, prev_tag, i-1)]*self.smoothed_transition_probabilities[trans_key]*self.smoothed_emission_probabilities[emiss_key]
                        if temp > max:
                            max = temp
                            max_tag = prev_tag
                        temp = 0
                    viterbi[(prev_tag, curr_tag, i)] = max
                    #backtrack[(tag3, prev_tag, i)] = max_tag
                    backtrack[i] = max_tag
                    max = 0

        max = 0
        max_prev_tag = ''
        max_curr_tag = ''
        temp = 0
        for prev_tag in list(self.tagset):
            for curr_tag in list(self.tagset):
                if (prev_tag,curr_tag,n-1) in viterbi:
                    temp = viterbi[(prev_tag,curr_tag,n-1)]
                if temp > max:
                    max = temp
                    max_prev_tag = prev_tag
                    max_curr_tag = curr_tag

        seq_tags = deque()
        seq_tags.append(max_curr_tag)
        seq_tags.append(max_prev_tag)
        for i in range(n-2, 0, -1):
            seq_tags.append(backtrack[i])

        seq_tags.reverse()

        bestpath = seq_tags

        return bestpath

    def add_unknown_words(self, training_data, training_data_words, known_words):
        unigrams = self.t_unigram_count(training_data_words, self.t_unigram_amount)
        count = 0
        known_words.add('UNK')
        for word in training_data_words:
            if unigrams[word] == 1:
                training_data_words[count] = 'UNK'
                training_data[count*2] = 'UNK'
                if word in known_words:
                    known_words.remove(word)
            count += 1

    def t_unigram_count(self, train_tags, unigram_amount):
        unigram_cnt = {}
        for i in range(unigram_amount):
            if train_tags[i] in unigram_cnt:
                unigram_cnt[train_tags[i]] = unigram_cnt[train_tags[i]]+1
            else:
                unigram_cnt[train_tags[i]] = 1

        return unigram_cnt

    def t_bigram_count(self, train_tags, bigram_amount):
        t_bigram_joint_count = {}
        for i in range(bigram_amount):
            bigram_joint = train_tags[i]+ ' ' +train_tags[i+1]
            if bigram_joint in t_bigram_joint_count:
                t_bigram_joint_count[bigram_joint] = t_bigram_joint_count[bigram_joint] + 1
            else:
                t_bigram_joint_count[bigram_joint] = 1

        return t_bigram_joint_count

    def t_trigram_count(self, train_tags, trigram_amount):
        t_trigram_joint_count = {}
        for i in range(trigram_amount):
            trigram_joint = train_tags[i]+ ' ' +train_tags[i+1]+ ' ' +train_tags[i+2]
            if trigram_joint in t_trigram_joint_count:
                t_trigram_joint_count[trigram_joint] = t_trigram_joint_count[trigram_joint] + 1
            else:
                t_trigram_joint_count[trigram_joint] = 1

        return t_trigram_joint_count

    def wt_bigram_count(self, train_text, bigram_amount):
        wt_bigram_joint_count = {}
        for i in range(0,bigram_amount,2):
            bigram_joint = train_text[i]+ ' ' +train_text[i+1]
            if bigram_joint in wt_bigram_joint_count:
                wt_bigram_joint_count[bigram_joint] = wt_bigram_joint_count[bigram_joint] + 1
            else:
                wt_bigram_joint_count[bigram_joint] = 1

        return wt_bigram_joint_count

    def unigram_probabilities(self, unigram_count, unigram_amount):
        unigram_prob = {}

        for key in unigram_count:
            unigram_prob[key] = unigram_count[key]/unigram_amount

        return unigram_prob

    def bigram_probabilities(self, unigram_count, bigram_joint_count):
        bigram_prob = {}

        for key in bigram_joint_count:
            keyBigram = key.split(' ')
            probKey = keyBigram[1]+ ' ' +keyBigram[0]
            probValue = bigram_joint_count[key] / unigram_count[keyBigram[0]]
            bigram_prob[probKey] = probValue

        return bigram_prob   

    def trigram_probabilities(self, trigram_joint_count, bigram_joint_count):
        trigram_prob = {}

        for key in trigram_joint_count:
            keyBigram = key.split(' ')
            bigram_joint_key = keyBigram[0]+ ' ' +keyBigram[1]
            prob_key = keyBigram[2]+ ' ' +keyBigram[0]+ ';' +keyBigram[1]
            if trigram_joint_count[key] == 0:
                prob_value = 0
            else:
                if bigram_joint_key not in bigram_joint_count:
                    prob_value = 0
                else:
                    prob_value = trigram_joint_count[key] / bigram_joint_count[bigram_joint_key]
            trigram_prob[prob_key] = prob_value

        return trigram_prob  
    
    def generate_all_possible_transitions(self, tagset):
        all_transitions = {}
        for tag in tagset:
            #trigram = tag
            for next_tag in tagset:
                #trigram = trigram + ' ' + next_tag
                for last_tag in tagset:
                    trigram = last_tag + ' ' + tag + ';' + next_tag
                    #trigram = trigram + ' ' + last_tag
                    all_transitions[trigram] = 0
                    #parts = trigram.split(' ')
                    #trigram = parts[0] + ' ' + parts[1]
                #parts = trigram.split(' ')
                #trigram = parts[0]
            #trigram = ''
        return all_transitions

    def generate_all_possible_wordtag_pairs(self, tagset, words):
        all_emissions = {}
        for word in words:
            bigram = word
            for tag in tagset:
                bigram = bigram + ' ' + tag
                all_emissions[bigram] = 0
                parts = bigram.split(' ')
                bigram = parts[0]

        return all_emissions

    def transition_probs(self, trigram_joint_count, bigram_joint_count):
        trans_prob = {}
        for key in trigram_joint_count:
            key_bigram = key.split(' ')
            bigram_joint_key = key_bigram[0]+ ' ' +key_bigram[1]
            probKey = key_bigram[2]+ ' ' + key_bigram[0]+ ';' + key_bigram[1]
            probValue = trigram_joint_count[key] / bigram_joint_count[bigram_joint_key]
            trans_prob[probKey] = probValue

        return trans_prob

    def emission_probs(self, t_unigram_cnt, wt_bigram_joint_count):
        emission_prob = {}
        for key in wt_bigram_joint_count:
            key_bigram = key.split(' ')
            tag_key = key_bigram[1]
            probValue = wt_bigram_joint_count[key]/t_unigram_cnt[tag_key]
            emission_prob[key] = probValue
        return emission_prob

    def move_counts(self, all_trans, ngram_joint_count):
        for key in ngram_joint_count:
            all_trans[key] = ngram_joint_count[key]

    def bigram_smoothing(self, list_of_heldout_data, epsylon, unigram_prob, bigram_prob, uniform_prob):
        heldout_brigram_probs = {}
        heldout_text_size = len(list_of_heldout_data)
        # Starting values of lambdas l0, l1, l2
        lambdas = [0.3, 0.3, 0.4]
        while 1:
            #for i in range(heldoutTextSize-2):
            for i in range(1, heldout_text_size, 2):
                new_bigram_key = list_of_heldout_data[i]+ ' ' + list_of_heldout_data[i-1]
                new_unigram_key = list_of_heldout_data[i]
                if new_bigram_key in bigram_prob.keys():
                    biProb = bigram_prob[new_bigram_key]
                else:
                    biProb = 0
                if new_unigram_key in unigram_prob.keys():
                    uniProb = unigram_prob[new_unigram_key]
                else:
                    uniProb = 0
                new_bigram_prob = lambdas[2]*biProb + lambdas[1]*uniProb + lambdas[0]*uniform_prob
                heldout_brigram_probs[new_bigram_key] = new_bigram_prob

            new_lambdas = [0, 0, 0]
            expected_counts = [0, 0, 0]

            # Compute expected counts for L0 
            for key in heldout_brigram_probs:
                expected_counts[0] = expected_counts[0] + lambdas[0]*uniform_prob/heldout_brigram_probs[key]

            # Compute expected counts for L1,L2
            for i in range(1, heldout_text_size, 2):
                bigram = list_of_heldout_data[i]+ ' ' + list_of_heldout_data[i-1]
                unigram = list_of_heldout_data[i]
                if unigram in unigram_prob.keys():
                    expected_counts[1] = expected_counts[1] + lambdas[1]*unigram_prob[unigram]/heldout_brigram_probs[bigram]
                if bigram in bigram_prob.keys():
                    expected_counts[2] = expected_counts[2] + lambdas[2]*bigram_prob[bigram]/heldout_brigram_probs[bigram]

            # Compute next lambdas
            new_lambdas[0] = expected_counts[0]/(expected_counts[0]+expected_counts[1]+expected_counts[2])
            new_lambdas[1] = expected_counts[1]/(expected_counts[0]+expected_counts[1]+expected_counts[2])
            new_lambdas[2] = expected_counts[2]/(expected_counts[0]+expected_counts[1]+expected_counts[2])

            # Check termination condition for next lambdas
            delta0 = abs(new_lambdas[0] - lambdas[0])
            delta1 = abs(new_lambdas[1] - lambdas[1])
            delta2 = abs(new_lambdas[2] - lambdas[2])
            if delta0 < epsylon and delta1 < epsylon and delta2 < epsylon:
                lambdas = new_lambdas
                break
            else:
                lambdas = new_lambdas

        return lambdas

    def trigram_smoothing(self, list_of_heldout_data, epsylon, unigram_prob, bigram_prob, trigram_prob, uniform_prob):
        heldout_trigram_probs = {}
        heldout_text_size = len(list_of_heldout_data)
        # Starting values of lambdas l0, l1, l2, l3
        lambdas = [0.1, 0.2, 0.3, 0.4]
        while 1:
            #for i in range(heldoutTextSize-2):
            for i in range(2, heldout_text_size):
                new_trigram_key = list_of_heldout_data[i]+ ' ' +list_of_heldout_data[i-2]+ ';' +list_of_heldout_data[i-1]
                new_bigram_key = list_of_heldout_data[i]+ ' ' +list_of_heldout_data[i-1]
                new_unigram_key = list_of_heldout_data[i]
                if new_trigram_key in trigram_prob.keys():
                    tri_prob = trigram_prob[new_trigram_key]
                else:
                    tri_prob = 0
                if new_bigram_key in bigram_prob.keys():
                    bi_prob = bigram_prob[new_bigram_key]
                else:
                    bi_prob = 0
                if new_unigram_key in unigram_prob.keys():
                    uni_prob = unigram_prob[new_unigram_key]
                else:
                    uni_prob = 0
                new_trigram_prob = lambdas[3]*tri_prob + lambdas[2]*bi_prob + lambdas[1]*uni_prob + lambdas[0]*uniform_prob
                heldout_trigram_probs[new_trigram_key] = new_trigram_prob

            new_lambdas = [0, 0, 0, 0]
            expected_counts = [0, 0, 0, 0]

            # Compute expected counts for L0 
            for key in heldout_trigram_probs:
                expected_counts[0] = expected_counts[0] + lambdas[0]*uniform_prob/heldout_trigram_probs[key]

            # Compute expected counts for L1,L2,L3  
            for i in range(2,heldout_text_size):
                trigram = list_of_heldout_data[i]+ ' ' +list_of_heldout_data[i-2]+ ';' +list_of_heldout_data[i-1]
                bigram = list_of_heldout_data[i]+ ' ' +list_of_heldout_data[i-1]
                unigram = list_of_heldout_data[i]
                if unigram in unigram_prob.keys():
                    expected_counts[1] = expected_counts[1] + lambdas[1]*unigram_prob[unigram]/heldout_trigram_probs[trigram]
                if bigram in bigram_prob.keys():
                    expected_counts[2] = expected_counts[2] + lambdas[2]*bigram_prob[bigram]/heldout_trigram_probs[trigram]
                if trigram in trigram_prob.keys():
                    expected_counts[3] = expected_counts[3] + lambdas[3]*trigram_prob[trigram]/heldout_trigram_probs[trigram]

            # Compute next lambdas
            new_lambdas[0] = expected_counts[0]/(expected_counts[0]+expected_counts[1]+expected_counts[2]+expected_counts[3])
            new_lambdas[1] = expected_counts[1]/(expected_counts[0]+expected_counts[1]+expected_counts[2]+expected_counts[3])
            new_lambdas[2] = expected_counts[2]/(expected_counts[0]+expected_counts[1]+expected_counts[2]+expected_counts[3])
            new_lambdas[3] = expected_counts[3]/(expected_counts[0]+expected_counts[1]+expected_counts[2]+expected_counts[3])

            # Check termination condition for next lambdas
            delta0 = abs(new_lambdas[0] - lambdas[0])
            delta1 = abs(new_lambdas[1] - lambdas[1])
            delta2 = abs(new_lambdas[2] - lambdas[2])
            delta3 = abs(new_lambdas[3] - lambdas[3])
            if delta0 < epsylon and delta1 < epsylon and delta2 < epsylon and delta3 < epsylon:
                lambdas = new_lambdas
                break
            else:
                lambdas = new_lambdas

        return lambdas

    def smoothed_transition_probs(self, all_trans, bigram_probs, unigram_probs, uniform_prob, lambdas):
        all_uni_values = unigram_probs.values()
        max_uni_value = max(all_uni_values)
        sum_uni = sum(all_uni_values)
        all_bi_values = bigram_probs.values()
        max_bi_value = max(all_bi_values)
        sum_bi = sum(all_bi_values)
        all_tri_values = all_trans.values()
        max_tri_value = max(all_tri_values)
        sum_tri = sum(all_tri_values)
        smoothed_trans_prob = {}
        for key in all_trans:
            key_split = key.split(' ') 
            key_cond = key_split[1].split(';')
            bigram_key = key_split[0] + ' ' + key_cond[1]
            unigram_key = key_split[0]
            if bigram_key in bigram_probs:
                bigram_prob = bigram_probs[bigram_key]
            else:
                bigram_prob = 0
            prob_value = lambdas[3]*all_trans[key] + lambdas[2]*bigram_prob + lambdas[1]*unigram_probs[unigram_key] + lambdas[0]*uniform_prob
            smoothed_trans_prob[key] = prob_value

        return smoothed_trans_prob

    def smoothed_emission_probs(self, all_emis, unigram_probs, uniform_prob, lambdas):
        smoothed_emis_prob = {}
        for key in all_emis:
            key_split = key.split(' ')
            unigram_key = key_split[0]
            prob_value = lambdas[2]*all_emis[key] + lambdas[1]*unigram_probs[unigram_key] + lambdas[0]*uniform_prob
            smoothed_emis_prob[key] = prob_value

        return smoothed_emis_prob