import math
import numpy as np

class HMMModel:
    _EPSYLON = 0.000001

    # Initialize a model. Last parameter "case" has 2 values: V or BW (to make a condition for calling Baum-Welch)
    def __init__(self, training_data, unsupervised_training_data, training_data_tags, training_data_words, heldout_data_tags, heldout_data_words, language, case) -> None: 
        
        # Create set of tags 
        self.tagset = set()
        for tag in training_data_tags:      
            self.tagset.add(tag)

        # Create set of words 
        self.known_words = set()
        for word in training_data_words:      
            self.known_words.add(word)

        # Amount of unigrams, bigrams, trigrams
        self.w_unigram_amount = len(training_data_words)
        self.wt_bigram_amount = self.w_unigram_amount

        self.t_unigram_amount = len(training_data_tags)
        self.t_bigram_amount = self.t_unigram_amount - 1
        self.t_trigram_amount = self.t_unigram_amount - 2

        # Remove some rare tags to decrease memory consumption (does not affect the accuracy). Supervised tagger: removes tags which were seen 
        # less than 10 times for both languages; 
        # Unseprvised tagger: removes tags which were seen less than 20 times for Czech and doesn't remove tags for English
        if (language == 'EN' and case == 'V') or (language == 'CZ' and case == 'V') or (language == 'CZ' and case == 'BW'):
            self.remove_rare_tags(training_data, training_data_tags, self.tagset, case)

        # Calculate uniform probability for tags and words
        self.tagset_size = len(self.tagset)
        self.t_uniform_probability = 1/self.tagset_size
        self.wordset_size = len(self.known_words)
        self.w_uniform_probability = 1/self.wordset_size

        # Uniform initial probabilities for the beginning of test set
        self.initial_probabilities = np.full(self.tagset_size, self.t_uniform_probability, dtype=float)

        # Convert tag set and word set to lists
        self.tag_list = list(self.tagset)
        self.word_list = list(self.known_words)

        # Create 3d matrix for tag transitions and 2d matrix for word emissions
        self.tag_transition = np.zeros((self.tagset_size, self.tagset_size, self.tagset_size), dtype=float)
        self.word_emission = np.zeros((self.tagset_size, len(self.known_words)), dtype=float)

        # n-gram counts (tags) and words/tags)
        self.t_unigram_en = self.t_unigram_count(training_data_tags, self.t_unigram_amount)
        self.t_bigram_en = self.t_bigram_count(training_data_tags, self.t_bigram_amount)
        self.t_trigram_en = self.t_trigram_count(training_data_tags, self.t_trigram_amount)
        
        # unigram counts (words)
        self.w_unigram_en = self.t_unigram_count(training_data_words, self.t_unigram_amount)

        # bigram counts (word+tags)
        self.wt_bigram_en = self.wt_bigram_count(training_data, self.wt_bigram_amount)

        # Compute transition and emission probabilities for training set and add values to tag transition and word emission matrices
        self.transition_probabilities = self.transition_probs(self.t_trigram_en, self.t_bigram_en, self.tag_list)
        self.emission_probabilities = self.emission_probs(self.t_unigram_en, self.wt_bigram_en, self.tag_list, self.word_list)

        # Compute n-gram probabilities
        self.w_unigram_prob =self.unigram_probabilities(self.w_unigram_en, self.w_unigram_amount)
        self.t_unigram_prob = self.unigram_probabilities(self.t_unigram_en, self.t_unigram_amount)
        self.t_bigram_prob = self.bigram_probabilities(self.t_unigram_en, self.t_bigram_en)

        # Convert n-gram probabilities to numpy structures
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

        # Compute lambdas for transition and emission models smoothing
        self.lambdas_transition = self.trigram_smoothing(heldout_data_tags, self._EPSYLON, self.t_unigram_prob, self.t_bigram_prob, self.transition_probabilities, self.t_uniform_probability)
        self.lambdas_emission = self.bigram_smoothing(heldout_data_tags, heldout_data_words, self._EPSYLON, self.w_unigram_prob, self.emission_probabilities, self.w_uniform_probability)

        # Smooth tag transition matrix 
        self.smoothed_transition_probs(self.tag_bigram_probs, self.tag_unigram_probs, self.t_uniform_probability, self.lambdas_transition)

        # Smooth word emission matrix
        self.smoothed_emission_probs(self.word_unigram_probs, self.w_uniform_probability, self.lambdas_emission)

        # Update tag transitions and word emissions with Baum-Welch. Due to enormous memory consumption
        # it cannot be done on whole training dataset, so it is done in cycle (dataset's part length = 'step' on each iteration)
        # and then average is taken. (ensemble)
        if case == 'BW':
            step = 2000
            bw_range = len(unsupervised_training_data) - len(unsupervised_training_data) % step
            count = 1
            # New structures for ensemble
            self.ensemble_tag_transition = np.zeros((self.tagset_size, self.tagset_size, self.tagset_size), dtype=float)
            self.ensemble_word_emission = np.zeros((self.tagset_size, len(self.known_words)), dtype=float)

            # Copy of initial transition and emission models
            cp_tag_transition = np.copy(self.tag_transition)
            cp_word_emission = np.copy(self.word_emission)
            print(f'Baum-Welch {count}')
            self.baum_welch(unsupervised_training_data[:step])
            self.ensemble_tag_transition += self.tag_transition
            self.ensemble_word_emission += self.word_emission

            self.tag_transition = np.copy(cp_tag_transition)
            self.word_emission = np.copy(cp_word_emission)
            i = step
            j = step*2
            while i <= bw_range - step and j <= bw_range:
                count += 1
                print(f'Baum-Welch {count}')
                self.baum_welch(unsupervised_training_data[i:j])
                self.ensemble_tag_transition += self.tag_transition
                self.ensemble_word_emission += self.word_emission
                self.tag_transition = np.copy(cp_tag_transition)
                self.word_emission = np.copy(cp_word_emission)
                i = j
                j += step

            # Taking average of updated transition and emission models
            self.tag_transition = self.ensemble_tag_transition / (bw_range/step)
            self.word_emission = self.ensemble_word_emission / (bw_range/step)

            # Smoothing
            self.smoothed_transition_probs(self.tag_bigram_probs, self.tag_unigram_probs, self.t_uniform_probability, self.lambdas_transition)
            self.smoothed_emission_probs(self.word_unigram_probs, self.w_uniform_probability, self.lambdas_emission)


    def baum_welch(self, train):
            # Create first element (matrix) in alpha
            alpha_1 = np.zeros((self.tagset_size, self.tagset_size), dtype=float)
            alpha_1 += 1/((self.tagset_size))

            # Create last element (matrix) in beta
            beta_last = np.ones((self.tagset_size, self.tagset_size), dtype=float)

            word_uniform = np.ones((self.tagset_size), dtype=float)/self.tagset_size

            # Create word dictionary to access words faster
            known_words_dict = {}
            for i in range(0,len(self.word_list)):  
                known_words_dict[self.word_list[i]]=i  

            # Create word dictionary for word emission updates
            words_training_dict ={}
            for i in range(0,len(train)):
                if train[i] in words_training_dict:
                    words_training_dict[train[i]].append(i)
                else:
                    words_training_dict[train[i]] = [i]

            transition_conv = 100
            emission_conv = 100

            while transition_conv > 15 and emission_conv > 0.0005:  # Convergence condition
                alpha = [alpha_1]
                beta = [None] * len(train)
                beta[len(train)-1] = beta_last

                #Alpha computation
                for t in range(1,len(train)):
                    a = np.zeros((self.tagset_size, self.tagset_size), dtype=float)
                    b = []
                    if train[t] in known_words_dict:
                        b = self.word_emission[:,known_words_dict[train[t]]]
                    else:
                        b = word_uniform

                    for k in range(0,self.tagset_size):
                        a[:,k] = np.sum(np.multiply(self.tag_transition[:,:,k],alpha[t-1]),0) * b[k]
                    a = a/np.sum(a, 0)
                    alpha.append(a)

                #Beta computation
                for t in range(1,len(train)):
                    b_back = []
                    bet = np.zeros((self.tagset_size, self.tagset_size), dtype=float)
                    if train[len(train)-t-1] in known_words_dict:
                        b_back = self.word_emission[:,known_words_dict[train[len(train)-t-1]]]
                    else:
                        b_back = word_uniform

                    for i in range(0,self.tagset_size):
                            bet[i,:] = np.sum(np.multiply(np.multiply(beta[len(train)-t],b_back), self.tag_transition[i,:,:]),1)
                    bet = bet/np.sum(bet, 0)
                    beta[len(train)-t-1] = bet

                #Compute gamma
                gamma = np.multiply(alpha,beta)
                sum_list = np.sum(np.sum(gamma,1),1)
                for i in range(0,len(alpha)):
                    gamma[i] = gamma[i]/sum_list[i]

                #Compute xi
                xis = [] 
                for t in range(0,len(alpha)-1):
                    xi = np.zeros((self.tagset_size, self.tagset_size, self.tagset_size), dtype=float)
                    #numerator
                    word_output = []
                    if train[t+1] in known_words_dict:
                        word_output = self.word_emission[:,known_words_dict[train[t+1]]]
                    else:
                        word_output = word_uniform
                
                    step_1 = np.multiply(self.tag_transition, alpha[t]) # transition * alpha
                    step_2 = np.multiply(beta[t+1],word_output) # beta * emission
                    for i in range(0,self.tagset_size):
                        xi[i,:,:] = np.multiply(step_1[i,:,],step_2)

                    denominator = np.sum(xi)

                    xi = xi /denominator

                    xis.append(xi)

                #Update transition probabilities
                cube_sum = np.sum(xis,0)
                k_sum = np.sum(cube_sum, 2)
                update_transition = np.zeros((self.tagset_size, self.tagset_size, self.tagset_size), dtype=float)

                for k in range(0,self.tagset_size):
                    update_transition[:,:,k] = cube_sum[:,:,k]/k_sum

                #Update emission probabilities
                update_emission = np.zeros((self.tagset_size, len(self.word_list)), dtype=float)
                denominator_gamma = np.sum(np.sum(gamma,0),1)
                for i in range(0,len(self.word_list)):
                    if self.word_list[i] in words_training_dict:
                        update_emission[:,i] = np.divide(np.sum(np.sum(gamma[words_training_dict[self.word_list[i]]],0),1),denominator_gamma)
                    else:
                        update_emission[:,i] = np.ones((self.tagset_size),dtype=float)/len(self.word_list)
        

                transition_conv = np.sum(np.abs(self.tag_transition - update_transition))  #15
                self.tag_transition = update_transition
                emission_conv = np.sum(np.abs(self.word_emission - update_emission))   #0.0005
                self.word_emission = update_emission
                print("transition convergence:" + str(transition_conv))
                print("emission convergence:" + str(emission_conv))

    # Full Viterbi (not pruned), it is not used, takes too much time, especially for Czech
    def viterbi(self, test):
        test_sequence = np.asarray(test)
        test_size = test_sequence.shape[0]

        viterbi = np.full((test_size, self.tagset_size, self.tagset_size), np.NINF, dtype=float)
        if test_sequence[0] in self.word_list:
            index_0 = self.word_list.index(test_sequence[0])
            log_emission = self.word_emission[:, index_0]
        else:
            log_emission = self.w_uniform_probability
        viterbi[0, :, :] = np.log(self.initial_probabilities * log_emission)
        backtrack = np.zeros((test_size, self.tagset_size, self.tagset_size))

        for t in range(1, test_size):   
            for j in range(self.tagset_size):
                    if test_sequence[t] in self.word_list:
                        index_t = self.word_list.index(test_sequence[t])
                        log_emission = np.log(self.word_emission[j, index_t])
                    else:
                        log_emission = np.log(self.w_uniform_probability)
                    probability =  viterbi[t - 1, :, :] + np.log(self.tag_transition[:, :, j]) + log_emission
                    index = np.unravel_index(probability.argmax(), probability.shape)
                    tag1 = index[0]
                    tag2 = index[1]
                    backtrack[t, tag2, j] = tag1

                    viterbi[t, tag2, j] = np.max(probability)
 
        path = np.zeros(test_size, dtype=np.int)

        index = np.unravel_index(np.argmax(viterbi[test_size - 1, :, :]), viterbi[test_size - 1, :, :].shape)
        l = index[0]
        j = index[1]
        last_state = j
        before_last_state = l 
 
        path[0] = last_state
        path[1] = before_last_state

        backtrack_index = 2
        for i in range(test_size - 2, -1, -1):
            if backtrack_index == test_size:
                break
            path[backtrack_index] = backtrack[i+1, int(before_last_state), int(last_state)]
            last_state = before_last_state
            before_last_state = path[backtrack_index]
            backtrack_index += 1

        path = np.flip(path, axis=0)
 
        result = []
        for step in path:
            result.append(self.tag_list[step])
 
        return result

    def get_most_probable_states(self, viterbi, N):
        root = math.sqrt(N)
        new_omega = np.full((int(root), int(root)), np.NINF, dtype=float)
        tag_indices = list()
        u = 0
        v = 0
        count = 0
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
        test_size = test_sequence.shape[0]
        N = 9
        #Structures to compute viterbi value for each step from previous one
        viterbi_prev = np.full((self.tagset_size, self.tagset_size), np.NINF, dtype=float)
        viterbi_next = np.full((self.tagset_size, self.tagset_size), np.NINF, dtype=float)

        #Compute the first step
        if test_sequence[0] in self.word_list:
            index_0 = self.word_list.index(test_sequence[0])
            log_emission = self.word_emission[:, index_0]
        else:
            log_emission = self.w_uniform_probability
        viterbi_prev[:,:] = np.log(self.initial_probabilities * log_emission)

        #Create structure for backtrack
        backtrack = np.zeros((test_size, self.tagset_size, self.tagset_size))

        root = math.sqrt(N)
        new_trans = np.full((int(root), int(root), self.tagset_size), np.NINF, dtype=float)
        count_word = 0

        for t in range(1, test_size):
            count_word += 1
            # Print when went through every 100 words to track progress
            if count_word % 100 == 0:
                print(count_word)
            
            #For each next step take N best states from previous step (prunning)
            if t > 1:
                viterbi_prev = np.copy(viterbi_next)
                viterbi_next[:,:] = np.NINF
                new_omega, tag_indices = self.get_most_probable_states(viterbi_prev, N)
            for j in range(self.tagset_size):
                    #For each next step take transitions according to N best states from previous step (prunning)
                    if t > 1:
                        new_trans = self.get_new_transitions(new_trans, tag_indices, j, N)
                    
                    #Compute viterbi values for tag j
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
                        index = np.unravel_index(probability.argmax(), probability.shape)
                        tag1 = index[0]
                        tag2 = index[1]

                    #Save tags for backtrack
                    backtrack[t, tag2, j] = tag1

                    #Save Viterbi values
                    viterbi_next[tag2, j] = np.max(probability)

        # Create structure for best path
        path = np.zeros(test_size, dtype=np.int)

        # Get the most probable last state
        index = np.unravel_index(np.argmax(viterbi_next[:, :]), viterbi_next[:, :].shape)
        l = index[0]
        j = index[1]
        last_state = j
        before_last_state = l 
 
        path[0] = last_state
        path[1] = before_last_state

        # Track the best path 
        backtrack_index = 2
        for i in range(test_size - 2, -1, -1):
            if backtrack_index == test_size:
                break
            path[backtrack_index] = backtrack[i+1, int(before_last_state), int(last_state)]
            last_state = before_last_state
            before_last_state = path[backtrack_index]
            backtrack_index += 1

        # Reverse the path
        path = np.flip(path, axis=0)
 
        # Convert numbers to tags
        result = []
        for step in path:
            result.append(self.tag_list[step])
 
        return result

    def remove_rare_tags(self, training_data, training_data_tags, tagset, case):
        if case == 'BW':
            n = 20
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

    def smoothed_transition_probs(self, bigram_probs, unigram_probs, uniform_prob, lambdas):
        for j in range(self.tagset_size):
            self.tag_transition[:, j, :] = lambdas[3] * self.tag_transition[:, j, :] + lambdas[2] * bigram_probs[j, :] + lambdas[1] * unigram_probs[:] + lambdas[0]*uniform_prob 

    def smoothed_emission_probs(self, unigram_probs, uniform_prob, lambdas):
        self.word_emission[:,:] = lambdas[2] * self.word_emission[:,:] + lambdas[1] * unigram_probs[:] + lambdas[0] * uniform_prob