#from collections import deque
import math
#import pathlib
#import random
#from re import L
#import itertools
import numpy as np
#import scipy as sc
#import time 

class HMMModel:
    _EPSYLON = 0.000001

    #Initialize a model. Last parameter "case" has 2 values: V or BW(to call Baum-Welch in case of unsupervised model)
    def __init__(self, training_data, unsupervised_training_data, training_data_tags, training_data_words, heldout_data_tags, heldout_data_words, language, case) -> None: 
        
        #Create set of tags 
        self.tagset = set()
        for tag in training_data_tags:      
            self.tagset.add(tag)

        #Create set of words 
        self.known_words = set()
        for word in training_data_words:      
            self.known_words.add(word)

        #Amount of unigrams, bigrams, trigrams
        self.w_unigram_amount = len(training_data_words)
        self.wt_bigram_amount = self.w_unigram_amount

        self.t_unigram_amount = len(training_data_tags)
        self.t_bigram_amount = self.t_unigram_amount - 1
        self.t_trigram_amount = self.t_unigram_amount - 2

        self.remove_rare_tags(training_data, training_data_tags, self.tagset, language)

        #Calculate uniform probability for tags and words
        self.tagset_size = len(self.tagset)
        self.t_uniform_probability = 1/self.tagset_size
        self.wordset_size = len(self.known_words)
        self.w_uniform_probability = 1/self.wordset_size

        #Uniform initial probabilities for the beginning of test set
        self.initial_probabilities = np.full(self.tagset_size, self.t_uniform_probability, dtype=float)

        #Convert tag set and word set to lists
        self.tag_list = list(self.tagset)
        self.word_list = list(self.known_words)

        #Create 3d matrix for tag transitions and 2d matrix for word emissions
        self.tag_transition = np.zeros((self.tagset_size, self.tagset_size, self.tagset_size), dtype=float)
        self.word_emission = np.zeros((self.tagset_size, len(self.known_words)), dtype=float)

        #n-gram counts (tags) and words/tags)
        self.t_unigram_en = self.t_unigram_count(training_data_tags, self.t_unigram_amount)
        self.t_bigram_en = self.t_bigram_count(training_data_tags, self.t_bigram_amount)
        self.t_trigram_en = self.t_trigram_count(training_data_tags, self.t_trigram_amount)
        
        #unigram counts (words)
        self.w_unigram_en = self.t_unigram_count(training_data_words, self.t_unigram_amount)

        #bigram counts (word+tags)
        self.wt_bigram_en = self.wt_bigram_count(training_data, self.wt_bigram_amount)

        #Compute transition and emission probabilities for training set and add values to tag transition and word emission matrices
        self.transition_probabilities = self.transition_probs(self.t_trigram_en, self.t_bigram_en, self.tag_list)
        self.emission_probabilities = self.emission_probs(self.t_unigram_en, self.wt_bigram_en, self.tag_list, self.word_list)

        #Compute n-gram probabilities
        self.w_unigram_prob =self.unigram_probabilities(self.w_unigram_en, self.w_unigram_amount)
        self.t_unigram_prob = self.unigram_probabilities(self.t_unigram_en, self.t_unigram_amount)
        self.t_bigram_prob = self.bigram_probabilities(self.t_unigram_en, self.t_bigram_en)

        #Convert n-gram probabilities to numpy structures
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

        #Compute lambdas for transition and emission models smoothing
        self.lambdas_transition = self.trigram_smoothing(heldout_data_tags, self._EPSYLON, self.t_unigram_prob, self.t_bigram_prob, self.transition_probabilities, self.t_uniform_probability)
        self.lambdas_emission = self.bigram_smoothing(heldout_data_tags, heldout_data_words, self._EPSYLON, self.w_unigram_prob, self.emission_probabilities, self.w_uniform_probability)

        #Smooth tag transition matrix 
        self.np_smoothed_transition_probs(self.tag_bigram_probs, self.tag_unigram_probs, self.t_uniform_probability, self.lambdas_transition, self.tag_list)

        #Smooth word emission matrix
        self.np_smoothed_emission_probs(self.word_unigram_probs, self.w_uniform_probability, self.lambdas_emission, self.tag_list, self.word_list)


        print("Precomputation done")

        #Update tag transitions and word emissions with Baum-Welch. Due to enormous memory consumption
        #it cannot be done on whole training dataset, so it is done in cycle (dataset part length = 10000 on each iteration for english)
        #and then average is taken. (ensemble)
        if case == 'BW':
            #New structures for ensemble
            self.ensemble_tag_transition = np.zeros((self.tagset_size, self.tagset_size, self.tagset_size), dtype=float)
            self.ensemble_word_emission = np.zeros((self.tagset_size, len(self.known_words)), dtype=float)
            #copy of initial transition and emission models
            cp_tag_transition = np.copy(self.tag_transition)
            cp_word_emission = np.copy(self.word_emission)
            self.new_baum_welch(unsupervised_training_data[:5000])
            print("BW")
            self.ensemble_tag_transition += self.tag_transition
            self.ensemble_word_emission += self.word_emission
            self.tag_transition = np.copy(cp_tag_transition)
            self.word_emission = np.copy(cp_word_emission)
            i = 5000
            j = 10000
            while i <= 165000 and j <= 170000:
                self.new_baum_welch(unsupervised_training_data[i:j])
                print("BW")
                self.ensemble_tag_transition += self.tag_transition
                self.ensemble_word_emission += self.word_emission
                self.tag_transition = np.copy(cp_tag_transition)
                self.word_emission = np.copy(cp_word_emission)
                i = j
                j += 5000
            #Taking average of updated transition and emission models
            self.tag_transition = self.ensemble_tag_transition / 34
            self.word_emission = self.ensemble_word_emission / 34

            #smoothing
            self.np_smoothed_transition_probs(self.tag_bigram_probs, self.tag_unigram_probs, self.t_uniform_probability, self.lambdas_transition, self.tag_list)
            self.np_smoothed_emission_probs(self.word_unigram_probs, self.w_uniform_probability, self.lambdas_emission, self.tag_list, self.word_list)

        print("Model initialization done")

    def new_forward(self, train):
        train_sequence = np.asarray(train)
        transition_shape = self.tag_transition.shape[0]
        train_shape = train_sequence.shape[0]
        alpha = np.zeros((transition_shape, transition_shape, train_shape))
        if train_sequence[0] in self.word_list:
            index = self.word_list.index(train_sequence[0])
            output_prob = self.word_emission[:, index]
        else:
            output_prob = 0
        alpha[:, :, 0] = self.initial_probabilities * output_prob
        norm = 0
        
        for t in range(1, train_shape):
            for j in range(transition_shape):
                if train_sequence[t] in self.word_list:
                    index_t = self.word_list.index(train_sequence[t])
                    output_prob = self.word_emission[j, index_t]
                else:
                    output_prob = self.w_uniform_probability
                product1 = np.dot(alpha[:, :, t - 1], self.tag_transition[:, :, j])
                probability =  np.dot(product1, output_prob)
                alpha[:, j, t] = np.sum(probability, axis=1)
                norm += np.sum(alpha[:, j, t])
            
            alpha[:, :, t] = alpha[:, :, t]/norm
            norm = 0
 
        return alpha


    def new_backward(self, train):
        train_sequence = np.asarray(train)
        transition_shape = self.tag_transition.shape[0]
        train_shape = train_sequence.shape[0]
        beta = np.zeros((transition_shape, transition_shape, train_shape))
 
        # setting beta(T) = 1
        beta[:, :, train_shape - 1] = np.ones((transition_shape))

        norm = 0
        for t in range(train_shape - 2, -1, -1):
            for i in range(transition_shape):
                if train_sequence[t+1] in self.word_list:
                    index_w = self.word_list.index(train_sequence[t+1])
                    output_prob = self.word_emission[i, index_w]
                else:
                    output_prob = self.w_uniform_probability
                #beta[t, j] = (beta[t + 1] * b[:, V[t + 1]]).dot(a[j, :])
                product1 = np.dot(beta[:, :, t+1], self.tag_transition[i, :, :])
                probability = np.dot(product1, output_prob)
                beta[i, :, t] = np.sum(probability, axis=1)
                norm += np.sum(beta[i, :, t])


                #product1 = np.dot(beta[:, :, t+1], self.tag_transition[:, :, j])
                #probability = np.dot(product1, output_prob)
                #beta[:, j, t] = np.sum(probability, axis=1)
                #norm += np.sum(beta[:, j, t])

            beta[:, :, t] = beta[:, :, t]/norm
            norm = 0
        return beta
    
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

        #words = set()
        #for word in train_sequence:      
        #    words.add(word)
        #wordset_size = len(words)
        #word_list = list(words)

        for n in range(100):
            alpha = self.new_forward(train)
            beta = self.new_backward(train)

            #count = np.zeros((M, M, M, wordset_size))
            #temp = np.zeros((M, M, M))
            #for word in word_list:
            #    for t in range(T - 1):
            #        if train_sequence[t] != word:
            #            continue
            #        else:
            #            count[:, :, :, word_list.index(word)] += alpha[:, :, t-1] 

            xi = np.zeros((M, M, M, T - 1))
            gamma = np.zeros((M, T - 1))
            for t in range(T - 1):
                #denominator = np.dot(np.dot(alpha[t, :].T, self.tag_transition) * self.word_emission[:, train_sequence[t + 1]].T, beta[t + 1, :])
                if train_sequence[t+1] in self.word_list:
                    index_w = self.word_list.index(train_sequence[t+1])
                    output_prob = self.word_emission[:, index_w]
                else:
                    output_prob = self.w_uniform_probability
                #denom1 = np.dot(alpha[:, :, t], self.tag_transition)
                #denom2 = np.dot(denom1, output_prob) 
                #denom3 = np.dot(denom2, beta[:, :, t+1])
                #denominator = np.sum(denom3)
                denom1 = np.dot(alpha[:, :, t], beta[:, :, t])
                denominator = np.sum(denom1)
                #denominator = (alpha[t, :].T @ self.tag_transition * output_prob.T) @ beta[t + 1, :]
                for i in range(M):
                    for j in range(M):
                        #numerator = alpha[t, i] * a[i, :] * b[:, V[t + 1]].T * beta[t + 1, :].T
                        #xi[i, :, t] = numerator / denominator
                        num1 = np.dot(alpha[i, j, t], self.tag_transition[i, j, :])
                        num2 = np.dot(num1, output_prob)
                        numerator = np.dot(num2, beta[j, :, t+1])
                        #numerator = alpha[t, i] * self.tag_transition[:, :, i] * output_prob.T * beta[t + 1, :].T
                        xi[i, j, :, t] = numerator / denominator

            numer = np.sum(xi, 3)
            denom = np.sum(numer, 2)

            self.tag_transition = np.divide(numer, denom)

            #gamma = np.sum(xi, axis=2)

            #transition update
            #numer = np.sum(xi, 3)
            #denom = np.sum(gamma, 2)
            #self.tag_transition = np.divide(numer, denom)#numer / denom
            #self.tag_transition = np.sum(xi, 2) / np.sum(gamma, axis=1)

            for t in range(T - 1):
                numer = np.dot(alpha[:, :, t], beta[:, :, t])
                numerator = np.sum(numer, axis=0)
                denomenator = np.sum(numerator, axis=0)
                gamma[:, t] = numerator / denomenator
 
            K = self.word_emission.shape[1]
            denominator = np.sum(gamma, axis=2)
            for l in range(K):
                self.word_emission[:, l] = np.sum(gamma[:, :, train_sequence == l], axis=2)

            self.word_emission = np.divide(self.word_emission, denominator.reshape((-1, 1)))
 
        return {"transition":self.tag_transition, "emission":self.word_emission}


    def new_baum_welch(self, train):
            alpha_1 = np.zeros((self.tagset_size, self.tagset_size), dtype=float)
            alpha_1 += 1/((self.tagset_size))

            beta_last = np.ones((self.tagset_size, self.tagset_size), dtype=float)

            word_uniform = np.ones((self.tagset_size), dtype=float)/self.tagset_size

            words_dict = {}  # Create word dictionary to access words faster
            for i in range(0,len(self.word_list)):  
                words_dict[self.word_list[i]]=i  

            words_training_dict ={}  #Create word dictionary for word emission updates
            for i in range(0,len(train)):
                if train[i] in words_training_dict:
                    words_training_dict[train[i]].append(i)
                else:
                    words_training_dict[train[i]] = [i]

            a_conv = 100
            b_conv = 100

            while a_conv > 15 and b_conv > 0.0005:  #convergence condition
            #for p in range(0,10): #convergence 
                alpha = [alpha_1]
                beta = [None] * len(train)
                beta[len(train)-1] = beta_last

                #Alpha computation
                for w in range(1,len(train)):
                    a = np.zeros((self.tagset_size, self.tagset_size), dtype=float)
                    b = []
                    if train[w] in words_dict:
                        b = self.word_emission[:,words_dict[train[w]]]
                    else:
                        b = word_uniform

                    for k in range(0,self.tagset_size):
                        a[:,k] = np.sum(np.multiply(self.tag_transition[:,:,k],alpha[w-1]),0) * b[k]
                    a = a/np.sum(a, 0)
                    alpha.append(a)

                #beta
                for w in range(1,len(train)):
                    b_back = []
                    bet = np.zeros((self.tagset_size, self.tagset_size), dtype=float)
                    if train[len(train)-w-1] in words_dict:
                        b_back = self.word_emission[:,words_dict[train[len(train)-w-1]]]
                    else:
                        b_back = word_uniform

                    for i in range(0,self.tagset_size):
                            bet[i,:] = np.sum(np.multiply(np.multiply(beta[len(train)-w],b_back), self.tag_transition[i,:,:]),1)
                    bet = bet/np.sum(bet, 0)
                    beta[len(train)-w-1] = bet
        
                gamma = np.multiply(alpha,beta)
                sum_list = np.sum(np.sum(gamma,1),1)
                for i in range(0,len(alpha)):
                    gamma[i] = gamma[i]/sum_list[i]

                xis = [] 
                for w in range(0,len(alpha)-1):
                    xi = np.zeros((self.tagset_size, self.tagset_size, self.tagset_size), dtype=float)
                    #numerator
                    b = []
                    if train[w+1] in words_dict:
                        b = self.word_emission[:,words_dict[train[w+1]]]
                    else:
                        b = word_uniform
                
                    step_1 = np.multiply(self.tag_transition, alpha[w]) #a..k*alpha
                    step_2 = np.multiply(beta[w+1],b)
                    for i in range(0,self.tagset_size):
                        xi[i,:,:] = np.multiply(step_1[i,:,],step_2)

                    denominator = np.sum(xi)

                    xi = xi /denominator

                    xis.append(xi)
        
                cube_sum = np.sum(xis,0)
                k_sum = np.sum(cube_sum, 2)
                update_a = np.zeros((self.tagset_size, self.tagset_size, self.tagset_size), dtype=float)

                for k in range(0,self.tagset_size):
                    update_a[:,:,k] = cube_sum[:,:,k]/k_sum

                b_update = np.zeros((self.tagset_size, len(self.word_list)), dtype=float)
                denominator_gamma = np.sum(np.sum(gamma,0),1)
                for i in range(0,len(self.word_list)):
                    if self.word_list[i] in words_training_dict:
                        b_update[:,i] = np.divide(np.sum(np.sum(gamma[words_training_dict[self.word_list[i]]],0),1),denominator_gamma)
                    else:
                        b_update[:,i] = np.ones((self.tagset_size),dtype=float)/len(self.word_list)
        

                a_conv = np.sum(np.abs(self.tag_transition - update_a))  #15
                self.tag_transition = update_a
                b_conv = np.sum(np.abs(self.word_emission - b_update))   #0.0005
                self.word_emission = b_update
                print("a convergence:" + str(a_conv))
                print("b convergence:" + str(b_conv))

    def np_viterbi(self, test):
        test_sequence = np.asarray(test)
        T = test_sequence.shape[0]
        M = self.tag_transition.shape[0]
        N = 25
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

    def get_most_probable_states(self, viterbi, t, N):
        root = math.sqrt(N)
        new_omega = np.full((int(root), int(root)), np.NINF, dtype=float)
        tag_indices = list()
        u = 0
        v = 0
        count = 0
        #tag_indices.clear()
        for k in range(0, N):
            index = np.unravel_index(viterbi.argmax(), viterbi.shape)
            index = np.unravel_index(viterbi.argmax(), viterbi.shape)
            new_omega[u, v] = np.max(viterbi)
            tag1 = index[0]
            tag2 = index[1]
            tag_indices.append((tag1, tag2))
            viterbi[tag1, tag2] = np.NINF
            count += 1
            if count < int(root):
                v += 1
            else:
                u += 1
                v = 0
                count = 0

        return new_omega, tag_indices

    def get_new_transitions(self, new_trans, tag_indices, j, N):
        root = math.sqrt(N)
        u = 0
        v = 0
        count = 0
        for k in range(0, N):
            tag1 = tag_indices[k][0]
            tag2 = tag_indices[k][1]
            new_trans[u, v, j] = self.tag_transition[tag1, tag2, j]
            count += 1
            if count < int(root):
                v += 1
            else:
                u += 1
                v = 0
                count = 0

        return new_trans


    def pruned_viterbi(self, test):
        test_sequence = np.asarray(test)
        T = test_sequence.shape[0]
        M = self.tag_transition.shape[0]
        N = 9
        viterbi_prev = np.full((M, M), np.NINF, dtype=float)
        viterbi_next = np.full((M, M), np.NINF, dtype=float)
        #omega = np.full((T, M, M), np.NINF, dtype=float)
        if test_sequence[0] in self.word_list:
            index_0 = self.word_list.index(test_sequence[0])
            log_emission = self.word_emission[:, index_0]
        else:
            log_emission = self.w_uniform_probability
        viterbi_prev[:,:] = np.log(self.initial_probabilities * log_emission)
        #omega[0, :, :] = np.log(self.initial_probabilities * log_emission)
        prev = np.zeros((T, M, M))
        root = math.sqrt(N)
        #new_omega = np.full((8, 8), np.NINF, dtype=float)
        new_trans = np.full((int(root), int(root), M), np.NINF, dtype=float)
        #tag_indices = list()
        count_word = 0

        for t in range(1, T):
            #st = time.time()
            count_word += 1
            if count_word % 100 == 0:
                print(count_word)
            
            if t > 1:
                viterbi_prev = np.copy(viterbi_next)
                viterbi_next[:,:] = np.NINF
                new_omega, tag_indices = self.get_most_probable_states(viterbi_prev, t, N)
            for j in range(M):
                    if t > 1:
                        new_trans = self.get_new_transitions(new_trans, tag_indices, j, N)
                    
                    # Same as Forward Probability
                    if test_sequence[t] in self.word_list:
                        index_t = self.word_list.index(test_sequence[t])
                        log_emission = np.log(self.word_emission[j, index_t])
                    else:
                        log_emission = np.log(self.w_uniform_probability) 
                    if t > 1:
                        probability = new_omega + np.log(new_trans[:, :, j]) + log_emission
                        index = np.unravel_index(probability.argmax(), probability.shape)
                        tag_index = index[0]*int(root) + index[1]
                        tag1 = tag_indices[tag_index][0]
                        tag2 = tag_indices[tag_index][1]
                    else:
                        probability =  viterbi_prev[:, :] + np.log(self.tag_transition[:, :, j]) + log_emission
                        #probability =  omega[t - 1, :, :] + np.log(self.tag_transition[:, :, j]) + log_emission
                    # This is our most probable state given previous state at time t (1)
                        index = np.unravel_index(probability.argmax(), probability.shape)
                        tag1 = index[0]
                        tag2 = index[1]
                    prev[t, tag2, j] = tag1

                    # This is the probability of the most probable state (2)
                    viterbi_next[tag2, j] = np.max(probability)
                    #omega[t, tag2, j] = np.max(probability)

            #et =time.time()
            #elapsed_time = et - st
            #print(elapsed_time)
        # Path Array
        S = np.zeros(T,dtype=np.int)

        # Find the most probable last hidden state
        index = np.unravel_index(np.argmax(viterbi_next[:, :]), viterbi_next[:, :].shape)
        #index = np.unravel_index(np.argmax(omega[T - 1, :, :]), omega[T - 1, :, :].shape)
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

    def remove_rare_tags(self, training_data, training_data_tags, tagset, language):
        if language == 'CZ':
            n = 30
        else: 
            n = 10
        unigrams = self.t_unigram_count(training_data_tags, self.t_unigram_amount)
        count = 0
        tagset.add('UNK')
        for word in training_data_tags:
            if unigrams[word] <= n:
                training_data_tags[count] = 'UNK'
                training_data[count*2 + 1] = 'UNK'
                if word in tagset:
                    tagset.remove(word)
                if word in self.tagset:
                    self.tagset.remove(word)
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