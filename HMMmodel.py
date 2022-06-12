from collections import deque
import math
import pathlib
import random
from re import L
import itertools
import numpy as np

class HMMModel:
    _EPSYLON = 0.000001

    
    def __init__(self, training_data, unsupervised_training_data, heldout_data, training_data_tags, training_data_words, heldout_data_tags, heldout_data_words, case) -> None: 
        self.tagset = set()
        for tag in training_data_tags:      
            self.tagset.add(tag)
        self.tagset_size = len(self.tagset)
        self.t_uniform_probability = 1/self.tagset_size

        self.initial_probabilities = np.full(self.tagset_size, self.t_uniform_probability, dtype=float)

        self.known_words = set()
        for word in training_data_words:      
            self.known_words.add(word)
        self.wordset_size = len(self.known_words)
        self.w_uniform_probability = 1/self.wordset_size

        self.w_unigram_amount = len(training_data_words)
        self.wt_bigram_amount = self.w_unigram_amount

        self.t_unigram_amount = len(training_data_tags)
        self.t_bigram_amount = self.t_unigram_amount - 1
        self.t_trigram_amount = self.t_unigram_amount - 2

        self.tag_list = list(self.tagset)
        #self.add_unknown_words(training_data, training_data_words, self.known_words)
        self.word_list = list(self.known_words)

        self.tag_transition = np.zeros((self.tagset_size, self.tagset_size, self.tagset_size), dtype=float)
        self.word_emission = np.zeros((self.tagset_size, len(self.known_words)), dtype=float)

        #self.all_transitions = list(itertools.combinations(self.tag_list, 3))
        #self.all_transitions = self.generate_all_possible_transitions(self.tag_list)
        #self.all_emissions = self.generate_all_possible_wordtag_pairs(self.tag_list, self.word_list)

        #n-gram counts (tags and words/tags)
        self.t_unigram_en = self.t_unigram_count(training_data_tags, self.t_unigram_amount)
        self.t_bigram_en = self.t_bigram_count(training_data_tags, self.t_bigram_amount)
        self.t_trigram_en = self.t_trigram_count(training_data_tags, self.t_trigram_amount)

        self.w_unigram_en = self.t_unigram_count(training_data_words, self.t_unigram_amount)
        self.wt_bigram_en = self.wt_bigram_count(training_data, self.wt_bigram_amount)

        # probabilities
        self.transition_probabilities = self.transition_probs(self.t_trigram_en, self.t_bigram_en, self.tag_list)
        self.emission_probabilities = self.emission_probs(self.t_unigram_en, self.wt_bigram_en, self.tag_list, self.word_list)

        #self.transition_probs(self.t_trigram_en, self.t_bigram_en, self.tag_list)
        #self.emission_probs(self.t_unigram_en, self.wt_bigram_en, self.tag_list, self.word_list)

        self.w_unigram_prob =self.unigram_probabilities(self.w_unigram_en, self.w_unigram_amount)

        self.t_unigram_prob = self.unigram_probabilities(self.t_unigram_en, self.t_unigram_amount)

        self.t_bigram_prob = self.bigram_probabilities(self.t_unigram_en, self.t_bigram_en)

        self.tag_bigram_probs = np.zeros((self.tagset_size, self.tagset_size), dtype=float)
        self.tag_unigram_probs = np.zeros((self.tagset_size), dtype=float)
        self.word_unigram_probs = np.zeros((len(self.known_words)), dtype=float)

        for key in self.t_bigram_prob:
            key_split = key.split(' ')
            tag1 = self.tag_list.index(key_split[0])
            tag2 = self.tag_list.index(key_split[1])
            self.tag_bigram_probs[tag2, tag1] = self.t_bigram_prob[key]
        
        for key in self.t_unigram_prob:
            tag = self.tag_list.index(key)
            self.tag_unigram_probs[tag] = self.t_unigram_prob[key]

        for key in self.w_unigram_prob:
            word = self.word_list.index(key)
            self.word_unigram_probs[word] = self.w_unigram_prob[key]

        #self.tag_unigrams = np.fromiter(self.t_unigram_prob.values, dtype=float)

        #self.t_trigram_prob = self.trigram_probabilities(self.t_trigram_en, self.t_bigram_en)

        #self.move_counts(self.all_transitions, self.transition_probabilities)
        #self.move_counts(self.all_emissions, self.emission_probabilities)



        #if case == 'V':
        self.lambdas_transition = self.trigram_smoothing(heldout_data_tags, self._EPSYLON, self.t_unigram_prob, self.t_bigram_prob, self.transition_probabilities, self.t_uniform_probability)
        self.lambdas_emission = self.bigram_smoothing(heldout_data_tags, heldout_data_words, self._EPSYLON, self.w_unigram_prob, self.emission_probabilities, self.w_uniform_probability)

        #self.lambdas_emission = [1.181759621716475e-13, 3.77674053343082e-06, 0.9999962232593484]

        #self.cp_tag_transition = np.copy(self.tag_transition)

        self.np_smoothed_transition_probs(self.tag_bigram_probs, self.tag_unigram_probs, self.t_uniform_probability, self.lambdas_transition, self.tag_list)
        
        #self.check_smooth_trans(self.transition_probabilities, self.t_bigram_prob, self.t_unigram_prob, self.t_uniform_probability, self.lambdas_transition, self.tag_list)
        #self.check_smooth_emis(self.emission_probabilities, self.w_unigram_prob, self.w_uniform_probability, self.lambdas_emission, self.tag_list, self.word_list)
        #difference = self.cp_tag_transition - self.tag_transition

        #count = np.count_nonzero(difference)

        #sum = np.sum(difference)
        
        #self.np_smoothed_emission_probs(self.w_unigram_prob, self.t_uniform_probability, self.lambdas_emission, self.tag_list, self.word_list)

        self.np_smoothed_emission_probs(self.word_unigram_probs, self.w_uniform_probability, self.lambdas_emission, self.tag_list, self.word_list)
        #self.check_smooth_emis(self.emission_probabilities, self.w_unigram_prob, self.w_uniform_probability, self.lambdas_emission, self.tag_list, self.word_list)

        #count = np.count_nonzero(difference)
        #self.smoothed_transition_probabilities = self.np_smoothed_transition_probs(self.tag_transition, self.transition_probabilities, self.t_bigram_prob, self.t_unigram_prob, self.t_uniform_probability, self.lambdas_transition, self.tag_list)
        #self.smoothed_emission_probabilities = self.smoothed_emission_probs(self.word_emission, self.emission_probabilities, self.w_unigram_prob, self.t_uniform_probability, self.lambdas_emission)



        if case == 'BW':
            #for key in self.smoothed_transition_probabilities:
            #    key_split = key.split(' ')
            #    key_conditions = key_split[1].split(';')
            #    tag1 = self.tag_list.index(key_conditions[0])
            #    tag2 = self.tag_list.index(key_conditions[1])
            #    tag3 = self.tag_list.index(key_split[0])
            #    self.tag_transition[tag1, tag2, tag3] = self.smoothed_transition_probabilities[key]

            #for key in self.smoothed_emission_probabilities:
            #    key_split = key.split(' ')
            #    tag2 = self.word_list.index(key_split[0])
            #    tag1 = self.tag_list.index(key_split[1])
            #    self.word_emission[tag1, tag2] = self.smoothed_emission_probabilities[key]
            
            #self.alpha = self.forward(unsupervised_training_data)
            #self.beta = self.backward(unsupervised_training_data)
            trans, emis = self.baum_welch(unsupervised_training_data)


        #for key in self.smoothed_transition_probabilities:
        #    key_split = key.split(' ')
        #    key_conditions = key_split[1].split(';')
        #    tag1 = self.tag_list.index(key_conditions[0])
        #    tag2 = self.tag_list.index(key_conditions[1])
        #    tag3 = self.tag_list.index(key_split[0])
       #    self.tag_transition[tag1, tag2, tag3] = self.smoothed_transition_probabilities[key]

        #for key in self.smoothed_emission_probabilities:
        #    key_split = key.split(' ')
        #    tag2 = self.word_list.index(key_split[0])
        #    tag1 = self.tag_list.index(key_split[1])
        #    self.word_emission[tag1, tag2] = self.smoothed_emission_probabilities[key]

        print("End")

    def forward(self, train):
        train_sequence = np.asarray(train)
        transition_shape = self.tag_transition.shape[0]
        train_shape = train_sequence.shape[0]
        alpha = np.zeros((train_shape, transition_shape))
        if train_sequence[0] in self.word_list:
            index = self.word_list.index(train_sequence[0])
            output_prob = self.word_emission[:, index]
        else:
            output_prob = 0
        alpha[0, :] = self.initial_probabilities * output_prob
        norm = 0
        
        for t in range(1, train_shape):
            for j in range(transition_shape):
                if train_sequence[t] in self.word_list:
                    index_w = self.word_list.index(train_sequence[t])
                    output_prob = self.word_emission[j, index_w]
                else:
                    output_prob = self.w_uniform_probability
                product = np.dot(alpha[t-1,:],self.tag_transition[:, :, j])
                sum = product.sum()
                alpha[t, j] = sum * output_prob
                #np.dot(alpha[t-1],self.tag_transition[:, j])
                #probability = omega[t - 1,:] + np.log(self.tag_transition[:, :, j]) + log_emission
                norm += alpha[t, j]
            alpha[t, :] = alpha[t, :]/norm
            norm = 0
 
        return alpha

    def backward(self, train, alpha):
        train_sequence = np.asarray(train)
        transition_shape = self.tag_transition.shape[0]
        train_shape = train_sequence.shape[0]
        beta = np.zeros((train_shape, transition_shape))
 
        # setting beta(T) = 1
        beta[train_shape - 1] = np.ones((transition_shape))

        norm = 0
        # Loop in backward way from T-1 to
        # Due to python indexing the actual loop will be T-2 to 0
        for t in range(train_shape - 2, -1, -1):
            for j in range(transition_shape):
                if train_sequence[t+1] in self.word_list:
                    index_w = self.word_list.index(train_sequence[t+1])
                    output_prob = self.word_emission[:, index_w]
                else:
                    output_prob = self.w_uniform_probability
                #product1 = np.dot(beta[t + 1],output_prob)
                #product2 = np.dot(product1, self.tag_transition[:, :, j])
                #sum = product2.sum()
                #beta[t, j] = product2.sum()
                #beta[t, j] = (beta[t + 1] * b[:, V[t + 1]]).dot(a[j, :])
                mult = beta[t + 1] * output_prob
                product = np.dot(mult,self.tag_transition[:, :, j])
                beta[t, j] = np.sum(product)#/100000
                norm += beta[t, j]
            beta[t, :] = beta[t, :]/norm
            norm = 0
        return beta

    def baum_welch(self, train):
        train_sequence = np.asarray(train)
        M = self.tag_transition.shape[0]
        T = len(train_sequence)
 
        for n in range(100):
            alpha = self.forward(train)
            beta = self.backward(train, alpha)
 
            xi = np.zeros((M, M, M, T - 2))
            for t in range(T - 2):
                #denominator = np.dot(np.dot(alpha[t, :].T, self.tag_transition) * self.word_emission[:, train_sequence[t + 1]].T, beta[t + 1, :])
                if train_sequence[t+1] in self.word_list:
                    index_w = self.word_list.index(train_sequence[t+1])
                    output_prob = self.word_emission[:, index_w]
                else:
                    output_prob = self.w_uniform_probability
                denom1 = np.dot(alpha[t, :], self.tag_transition)
                denom2 = np.dot(denom1, output_prob) 
                denom3 = np.dot(denom2, beta[t+1, :])
                denominator = np.sum(denom3)
                #denominator = (alpha[t, :].T @ self.tag_transition * output_prob.T) @ beta[t + 1, :]
                for i in range(M):
                    for j in range(M):
                        num1 = np.dot(alpha[t, i], self.tag_transition[i, j, :])
                        num2 = np.dot(num1, output_prob)
                        numerator = np.dot(num2, beta[t+1, :])
                        #numerator = alpha[t, i] * self.tag_transition[:, :, i] * output_prob.T * beta[t + 1, :].T
                        xi[:, :, i, t] = numerator / denominator
 
            gamma = np.sum(xi, axis=1)
            self.tag_transition = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
 
            # Add additional T'th element in gamma
            gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
 
            K = self.word_emission.shape[1]
            denominator = np.sum(gamma, axis=1)
            for l in range(K):
                self.word_emission[:, l] = np.sum(gamma[:, train_sequence == l], axis=1)
 
            self.word_emission = np.divide(self.word_emission, denominator.reshape((-1, 1)))
 
        return {"transition":self.tag_transition, "emission":self.word_emission}

    def np_viterbi(self, test):
        test_sequence = np.asarray(test)
        T = test_sequence.shape[0]
        M = self.tag_transition.shape[0]
        #omega = np.zeros((T, M))
        #omega = np.zeros((T, M, M))
        omega = np.full((T, M, M), np.NINF, dtype=float)
        #index = self.word_list.index(test_sequence[0])
        if test_sequence[0] in self.word_list:
            index_0 = self.word_list.index(test_sequence[0])
            log_emission = self.word_emission[:, index_0]
        else:
            log_emission = self.w_uniform_probability
        omega[0, :, :] = np.log(self.initial_probabilities * log_emission)
        #prev = np.zeros((T - 1, M))
        prev = np.zeros((T, M, M))
        #prev = np.full((T, M), np.NINF, dtype=int)

        for t in range(1, T):
            for j in range(M):
                    # Same as Forward Probability
                    if test_sequence[t] in self.word_list:
                        index_t = self.word_list.index(test_sequence[t])
                        log_emission = np.log(self.word_emission[j, index_t])
                    else:
                        log_emission = np.log(self.w_uniform_probability) #self.word_list.index('UNK')
                    #probability = omega[t - 1,:] + np.log(self.tag_transition[:, :, j]) + log_emission
                    probability =  omega[t - 1, :, :] + np.log(self.tag_transition[:, :, j]) + log_emission
                    # This is our most probable state given previous state at time t (1)
                    index = np.unravel_index(probability.argmax(), probability.shape)
                    tag1 = index[0]
                    tag2 = index[1]
                    #prev[t - 1, j] = tag2
                    prev[t, tag2, j] = tag1
 
                    # This is the probability of the most probable state (2)
                    omega[t, tag2, j] = np.max(probability)
 
        # Path Array
        S = np.zeros(T,dtype=np.int)

        # Find the most probable last hidden state
        index = np.unravel_index(np.argmax(omega[T - 1, :, :]), omega[T - 1, :, :].shape)
        l = index[0]
        j = index[1]
        last_state = j
        before_last_state = l 

        #last_state = np.argmax(omega[T - 1, :])
 
        S[0] = last_state
        S[1] = before_last_state

        backtrack_index = 2
        for i in range(T - 2, -1, -1):
            if backtrack_index == T:
                break
            S[backtrack_index] = prev[i+1, int(before_last_state), int(last_state)]
            last_state = before_last_state
            before_last_state = S[backtrack_index]
            backtrack_index += 1

        #backtrack_index = 1
        #for i in range(T - 1, -1, -1):
        #    if backtrack_index == T:
        #        break
        #    S[backtrack_index] = prev[i, int(last_state)]
        #    last_state = prev[i, int(last_state)]
        #    backtrack_index += 1

        # Flip the path array since we were backtracking
        S = np.flip(S, axis=0)
 
        # Convert numeric values to actual hidden states
        result = []
        for s in S:
            result.append(self.tag_list[s])
 
        return result

    def bi_viterbi(self, test):
        test_sequence = np.asarray(test)
        T = test_sequence.shape[0]
        M = self.tag_transition.shape[0]
 
        #omega = np.zeros((T, M))
        #omega = np.zeros((T, M, M))
        omega = np.full((T, M), np.NINF, dtype=float)
        index = self.word_list.index(test_sequence[0])
        omega[0, :] = np.log(self.initial_probabilities * self.word_emission[:, index])

        #prev = np.zeros((T - 1, M))
        prev = np.zeros((T, M))
        #prev = np.full((T, M), np.NINF, dtype=int)

        for t in range(1, T):
            for j in range(M):
                    # Same as Forward Probability
                    if test_sequence[t] in self.word_list:
                        index_t = self.word_list.index(test_sequence[t])
                        log_emission = np.log(self.word_emission[j, index_t])
                    else:
                        log_emission = self.w_uniform_probability #self.word_list.index('UNK')
                    probability = omega[t - 1,:] + np.log(self.tag_transition[:, :, j]) + log_emission
                    # This is our most probable state given previous state at time t (1)
                    #tag1 = np.argmax(probability)
                    index = np.unravel_index(probability.argmax(), probability.shape)
                    tag1 = index[0]
                    tag2 = index[1]
                    prev[t - 1, j] = tag2
 
                    # This is the probability of the most probable state (2)
                    omega[t, j] = np.max(probability)
 
        # Path Array
        S = np.zeros(T,dtype=np.int)

        # Find the most probable last hidden state
        last_state = np.argmax(omega[T - 1, :])
 
        S[0] = last_state

        backtrack_index = 1
        for i in range(T - 1, -1, -1):
            if backtrack_index == T:
                break
            S[backtrack_index] = prev[i, int(last_state)]
            last_state = prev[i, int(last_state)]
            backtrack_index += 1

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
    
    def generate_all_possible_transitions(self, taglist):
        all_transitions = {}
        for tag in taglist:
            #trigram = tag
            for next_tag in taglist:
                #trigram = trigram + ' ' + next_tag
                for last_tag in taglist:
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

    def transition_probs(self, trigram_joint_count, bigram_joint_count, taglist):
        trans_probs = {}
        for key in trigram_joint_count:
            key_bigram = key.split(' ')
            bigram_joint_key = key_bigram[0]+ ' ' +key_bigram[1]
            probKey = key_bigram[2]+ ' ' + key_bigram[0]+ ';' + key_bigram[1]
            probValue = trigram_joint_count[key] / bigram_joint_count[bigram_joint_key]
            tag1 = taglist.index(key_bigram[0])
            tag2 = taglist.index(key_bigram[1])
            tag3 = taglist.index(key_bigram[2])
            self.tag_transition[tag1, tag2, tag3] = probValue
            trans_probs[probKey] = probValue

        return trans_probs

    def emission_probs(self, t_unigram_cnt, wt_bigram_joint_count, taglist, wordlist):
        emis_probs = {}
        for key in wt_bigram_joint_count:
            key_bigram = key.split(' ')
            tag_key = key_bigram[1]
            tag = taglist.index(key_bigram[1])
            word = wordlist.index(key_bigram[0])
            probValue = wt_bigram_joint_count[key]/t_unigram_cnt[tag_key]
            self.word_emission[tag, word] = probValue
            emis_probs[key] = probValue

        return emis_probs

    def bigram_smoothing(self, heldout_data_tags, heldout_data_words, epsylon, unigram_prob, bigram_prob, uniform_prob):
        heldout_brigram_probs = {}
        heldout_size = len(heldout_data_words)
        # Starting values of lambdas l0, l1, l2
        lambdas = [0.3, 0.3, 0.4]
        while 1:
            #for i in range(heldoutTextSize-2):
            for i in range(0, heldout_size):
                new_bigram_key = heldout_data_words[i]+ ' ' + heldout_data_tags[i]
                new_unigram_key = heldout_data_words[i]
                if new_unigram_key in unigram_prob.keys():
                    uniProb = unigram_prob[new_unigram_key]
                else:
                    uniProb = 0
                if new_bigram_key in bigram_prob.keys() and uniProb != 0:
                    biProb = bigram_prob[new_bigram_key]
                else:
                    biProb = 0
                if uniProb == 0:
                    biProb = uniform_prob
                new_bigram_prob = lambdas[2]*biProb + lambdas[1]*uniProb + lambdas[0]*uniform_prob
                heldout_brigram_probs[new_bigram_key] = new_bigram_prob

            new_lambdas = [0, 0, 0]
            expected_counts = [0, 0, 0]

            # Compute expected counts for L0 
            #for key in heldout_brigram_probs:
            #    expected_counts[0] = expected_counts[0] + lambdas[0]*uniform_prob/heldout_brigram_probs[key]

            # Compute expected counts for L1,L2
            #for key in heldout_brigram_probs:
            #    unigram = key.split(' ')[0]
            #    expected_counts[0] = expected_counts[0] + lambdas[0]*uniform_prob/heldout_brigram_probs[key]
            #    if unigram in unigram_prob.keys():
            #        expected_counts[1] = expected_counts[1] + lambdas[1]*unigram_prob[unigram]/heldout_brigram_probs[key]
            #    if key in bigram_prob.keys():
            #        expected_counts[2] = expected_counts[2] + lambdas[2]*bigram_prob[key]/heldout_brigram_probs[key]


            for i in range(0, heldout_size):
                bigram = heldout_data_words[i]+ ' ' + heldout_data_tags[i]
                unigram = heldout_data_words[i]
                expected_counts[0] = expected_counts[0] + lambdas[0]*uniform_prob/heldout_brigram_probs[bigram]
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
                if new_unigram_key in unigram_prob.keys():
                    uni_prob = unigram_prob[new_unigram_key]
                else:
                    uni_prob = 0
                if new_bigram_key in bigram_prob.keys():
                    bi_prob = bigram_prob[new_bigram_key]
                else:
                    bi_prob = 0
                if new_trigram_key in trigram_prob.keys():
                    tri_prob = trigram_prob[new_trigram_key]
                else:
                    tri_prob = 0
                if uni_prob == 0:
                    bi_prob = uniform_prob
                if bi_prob == 0:
                    tri_prob = uniform_prob
                new_trigram_prob = lambdas[3]*tri_prob + lambdas[2]*bi_prob + lambdas[1]*uni_prob + lambdas[0]*uniform_prob
                heldout_trigram_probs[new_trigram_key] = new_trigram_prob

            new_lambdas = [0, 0, 0, 0]
            expected_counts = [0, 0, 0, 0]

            # Compute expected counts for L0 
            #for key in heldout_trigram_probs:
            #    expected_counts[0] = expected_counts[0] + lambdas[0]*uniform_prob/heldout_trigram_probs[key]

            for i in range(2,heldout_text_size):
                trigram = list_of_heldout_data[i]+ ' ' +list_of_heldout_data[i-2]+ ';' +list_of_heldout_data[i-1]
                bigram = list_of_heldout_data[i]+ ' ' +list_of_heldout_data[i-1]
                unigram = list_of_heldout_data[i]
                expected_counts[0] = expected_counts[0] + lambdas[0]*uniform_prob/heldout_trigram_probs[trigram]
                if unigram in unigram_prob.keys():
                    expected_counts[1] = expected_counts[1] + lambdas[1]*unigram_prob[unigram]/heldout_trigram_probs[trigram]
                if bigram in bigram_prob.keys():
                    expected_counts[2] = expected_counts[2] + lambdas[2]*bigram_prob[bigram]/heldout_trigram_probs[trigram]
                if trigram in trigram_prob.keys():
                #if trigram_prob[trigram] != 0:
                    expected_counts[3] = expected_counts[3] + lambdas[3]*trigram_prob[trigram]/heldout_trigram_probs[trigram]

            # Compute expected counts for L1,L2,L3  
            #for i in range(2,heldout_text_size):
            #    trigram = list_of_heldout_data[i]+ ' ' +list_of_heldout_data[i-2]+ ';' +list_of_heldout_data[i-1]
            #    bigram = list_of_heldout_data[i]+ ' ' +list_of_heldout_data[i-1]
            #    unigram = list_of_heldout_data[i]
            #    if unigram in unigram_prob.keys():
            #        expected_counts[1] = expected_counts[1] + lambdas[1]*unigram_prob[unigram]/heldout_trigram_probs[trigram]
            #    if bigram in bigram_prob.keys():
            #        expected_counts[2] = expected_counts[2] + lambdas[2]*bigram_prob[bigram]/heldout_trigram_probs[trigram]
            #    #if trigram in trigram_prob.keys():
            #    if trigram_prob[trigram] != 0:
            #        expected_counts[3] = expected_counts[3] + lambdas[3]*trigram_prob[trigram]/heldout_trigram_probs[trigram]

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

    def smoothed_transition_probs(self, all_trans, trigram_probs, bigram_probs, unigram_probs, uniform_prob, lambdas):
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

    def np_smoothed_transition_probs(self, bigram_probs, unigram_probs, uniform_prob, lambdas, taglist):
        for j in range(self.tagset_size):
            bigram = np.copy(bigram_probs[:, j])
            unigram = np.copy(unigram_probs[:])
            self.tag_transition[:, j, :] = lambdas[3] * self.tag_transition[:, j, :] + lambdas[2] * bigram_probs[j, :] + lambdas[1] * unigram_probs[:] + lambdas[0]*uniform_prob 

    def check_smooth_trans(self, trigram_probs, bigram_probs, unigram_probs, uniform_prob, lambdas, taglist):    
        iterat = np.nditer(self.tag_transition, flags=['multi_index'], op_flags=['readwrite'])
        for prob in iterat:
            tag1 = taglist[iterat.multi_index[0]]
            tag2 = taglist[iterat.multi_index[1]]
            tag3 = taglist[iterat.multi_index[2]]
            trigram_key = tag3 + ' ' + tag1 + ';' + tag2
            bigram_key = tag3 + ' ' + tag2
            unigram_key = tag3
            if bigram_key in bigram_probs:
                bigram_prob = bigram_probs[bigram_key]
            else:
                bigram_prob = 0
            if trigram_key in trigram_probs:
                trigram_prob = trigram_probs[trigram_key]
            else:
                trigram_prob = 0
            prob_value = lambdas[3]*trigram_prob + lambdas[2]*bigram_prob + lambdas[1]*unigram_probs[unigram_key] + lambdas[0]*uniform_prob
            self.tag_transition[iterat.multi_index] = prob_value

    def np_smoothed_emission_probs(self, unigram_probs, uniform_prob, lambdas, taglist, wordlist):
        unigram = np.copy(unigram_probs[:])
        self.word_emission[:,:] = lambdas[2] * self.word_emission[:,:] + lambdas[1] * unigram_probs[:] + lambdas[0] * uniform_prob
        
    def check_smooth_emis(self, bigram_probs, unigram_probs, uniform_prob, lambdas, taglist, wordlist):
        iterat = np.nditer(self.word_emission, flags=['multi_index'], op_flags=['readwrite'])
        for prob in iterat:
            tag = taglist[iterat.multi_index[0]]
            word = wordlist[iterat.multi_index[1]]
            bigram_key = word + ' ' + tag
            unigram_key = word
            if bigram_key in bigram_probs:
                bigram_prob = bigram_probs[bigram_key]
            else:
                bigram_prob = 0
            prob_value = lambdas[2]*bigram_prob + lambdas[1]*unigram_probs[unigram_key] + lambdas[0]*uniform_prob
            self.word_emission[iterat.multi_index] = prob_value


    def smoothed_emission_probs(self, all_emis, bigram_probs, unigram_probs, uniform_prob, lambdas):
        smoothed_emis_prob = {}
        count = 0
        for key in all_emis:
            key_split = key.split(' ')
            unigram_key = key_split[0]
            prob_value = lambdas[2]*all_emis[key] + lambdas[1]*unigram_probs[unigram_key] + lambdas[0]*uniform_prob
            smoothed_emis_prob[key] = prob_value

        return smoothed_emis_prob