#!/usr/bin/env python

import sys
import time
import pomdp_generator
import pomdp_parser
import policy_parser
import readline
# import speech_recognizer
import numpy
import random
from scipy import stats
#sys.path.append('/home/ludc/software/python_progress/progress-1.2')
#sys.path.append('/home/szhang/software/python_progress/progress-1.2')
from progress.bar import Bar
import subprocess
import gen_dis_plog
import conf
import re

import os
import string

import ast

numpy.set_printoptions(suppress=True)

class Simulator(object):

    def __init__(self, 
        auto_observations=True, 
        auto_state = False, 
        uniform_init_belief =True,
        print_flag=True,
        use_plog = False,
        policy_file='policy/default.policy', 
        pomdp_file='models/default.pomdp', 
        trials_num=1000,
        num_task=1, 
        num_patient=1,
        num_recipient=1):

        # print(pomdp_file)
        # print(policy_file)

        self.auto_observations = auto_observations
        self.auto_state = auto_state
        self.uniform_init_belief = uniform_init_belief
        self.print_flag = print_flag
        self.use_plog = use_plog
        self.trials_num = trials_num

        self.num_task = num_task
        self.num_patient = num_patient
        self.num_recipient = num_recipient
        self.tablelist = conf.tablelist

        # to read the pomdp model
        model = pomdp_parser.Pomdp(filename=pomdp_file, parsing_print_flag=False)
        self.states = model.states
        self.states_plus= None
        self.actions = model.actions
        self.observations = model.observations
        # print self.observations
        self.trans_mat = model.trans_mat
        self.obs_mat = model.obs_mat
        self.reward_mat = model.reward_mat

        # to read the learned policy
        self.policy = policy_parser.Policy(len(self.states), len(self.actions), 
            filename=policy_file)

        self.b = None   
        self.b_plus = None   
        self.a = None
        self.a_plus=None
        self.o = None
        self.o_plus= None
        self.md = 'happy'
        self.fl = True
        self.trigger= 1 ###triggle of event in which dialog turn
        # self.dialog_turn = 0

        self.plog = gen_dis_plog.DistrGen()

        # for semantic parser
        self.path_to_main = os.path.dirname(os.path.abspath(__file__))
        self.log_filename = os.path.join(self.path_to_main,'data','log','log.txt')

        #path to SPF jar
        self.path_to_spf = os.path.join(self.path_to_main,'spf','dist','spf-1.5.5.jar')
        #path to write-able experiment directory
        self.path_to_experiment = os.path.join(self.path_to_main,'spf','geoquery','experiments','template','dialog_writeable')
        # known words and
        given_words,word_to_ontology_map = self.get_known_words_from_seed_files()
        self.known_words = given_words

        # full request string
        self.full_request = ''
        
        self.known_words_to_number = {}
        self.get_known_words_to_number()
        # to make the screen print simple 
        numpy.set_printoptions(precision=2)

    #######################################################################
    def init_belief(self):

        if self.uniform_init_belief:
            self.b = numpy.ones(len(self.states)) / float(len(self.states))
                # print '\n',self.s, self.ct, self.b
        else:
            # here initial belief is sampled from a Dirichlet distribution
            self.b = numpy.random.dirichlet( numpy.ones(len(self.states)) )

        self.b = self.b.T

    ######################################################################
    #######################################################################
    def init_belief_plus(self):

        if self.uniform_init_belief:
            self.b_plus = numpy.ones(len(self.states_plus)) / float(len(self.states_plus))
                # print '\n',self.s, self.ct, self.b
        else:
            # here initial belief is sampled from a Dirichlet distribution
            self.b_plus = numpy.random.dirichlet( numpy.ones(len(self.states_plus)) )

        self.b_plus = self.b_plus.T

    ######################################################################


    def get_known_words_to_number(self):
        "DEBUG: getting known words to observation map"
        file_known = open(os.path.join(self.path_to_main,'data','known_words_to_obs.txt'), 'r')
        s = file_known.read()
        self.known_words_to_number = ast.literal_eval(s)
        print(str(self.known_words_to_number))
        file_known.close()

    def write_known_words_to_number(self):
        "DEBUG: saving known words to observations to file"
        file_known = open(os.path.join(self.path_to_main,'data','known_words_to_obs.txt'), 'w+')
        file_known.write(str(self.known_words_to_number))
        file_known.close()

    #######################################################################
    def get_known_words_from_seed_files(self):

    	seed_words = {}
    	word_to_ontology_map = {}
    	for filename in ['np-list.lex','seed.lex']:
    		f = open(os.path.join(self.path_to_experiment,'resources',filename))
    		for line in f:
    			if (len(line) <= 1 or line[:2] == "//"):
    				continue
    			[token_sequence,tag_and_grounding] = line.split(" :- ")
    			to_add = tag_and_grounding.strip().split(" : ")
    			if (filename == 'np-list.lex'): #only add to seed words those words which are already grounded (ie, no CCG tags)
    				seed_words[token_sequence] = to_add
    			word_to_ontology_map[to_add[1].split(":")[0].strip()] =to_add[1].split(":")[1].strip()
    	return seed_words,word_to_ontology_map
	###########################################################################

    #! this is an experimental method !#
    #invoke SPF parser and get the semantic parse(s) of the sentence, as well as any new unmapped words in the utterance
    def parse_utterance(self, user_utterance_text):

        f = open(os.path.join(self.path_to_experiment,'data','test.ccg'),'w')
        f.write(user_utterance_text+"\n(lambda $0:e $0)\n")
        f.close()

        #run parser and read output
        os.system('java -jar '+self.path_to_spf+' '+os.path.join(self.path_to_experiment,'test.exp'))
        f = open(os.path.join(self.path_to_experiment,'logs','load_and_test.log'),'r')
        lines = f.read().split("\n")
        parses = []
        current_unmapped_sequence = None #[sequence, last_index]
        unmapped_words_in_utterance = {}
        for i in range(0,len(lines)):
            if (' WRONG: ' in lines[i] or 'too many parses' in lines[i]): #found parses
                if (' WRONG: ' in lines[i] and len(lines[i].split('WRONG: ')[1]) > 0 and 'parses' not in lines[i].split('WRONG: ')[1]): #one parse
                    parses.append((lines[i].split('WRONG: ')[1],0))
                else: #multiple parses
                    j = 1 if ' WRONG: ' in lines[i] else 2
                    while (' Had correct parses: ' not in lines[i+j]):
                        if ('[S' not in lines[i+j]):
                            p = lines[i+j][lines[i+j].index('[')+2:]
                        else:
                            p = lines[i+j].split(']')[2][1:]
                        s = float(lines[i+j+1].split()[3])
                        print s #DEBUG
                        parses.append((p,s))
                        j += 3
            elif ('EMPTY' in lines[i] and len(lines[i].split()) >= 4 and lines[i].split()[3] == "EMPTY"): #found unmapped word
                empty_token = lines[i].split()[1]
                if (current_unmapped_sequence == None):
                    current_unmapped_sequence = [empty_token,i]
                elif (i-1 == current_unmapped_sequence[1]):
                    current_unmapped_sequence[0] += " "+empty_token
                    current_unmapped_sequence[1] = i
                else:
                    if (current_unmapped_sequence[0] not in self.known_words):
                        unmapped_words_in_utterance[current_unmapped_sequence[0]] = {}
                    current_unmapped_sequence = [empty_token,i]
        if (current_unmapped_sequence != None and current_unmapped_sequence[0] not in self.known_words):
            unmapped_words_in_utterance[current_unmapped_sequence[0]] = {}
        f.close()

        return parses,unmapped_words_in_utterance

    ######################################################################
    # EXPERIMENTAL: Generate model
    # Saeid: this method is working fine now, only name of pomdp and policy files needs to be updated in future to avoid conflicts
    def generate_new_model(self):
        r_max = 20.0
        r_min = -20.0

        wh_cost = -1.5
        yesno_cost = -1.0

        num_task = self.num_task
        num_patient = self.num_patient
        num_recipient = self.num_recipient

        strategy = str(num_task) + str(num_patient) + str(num_recipient)
      # two lines below commented temporarily
      #  pg = pomdp_generator.PomdpGenerator(num_task, num_patient + 1, num_recipient+ 1, r_max, r_min, strategy, \
      #      wh_cost, yesno_cost,timeout=2,pomdpfilename = '333_new_plus.pomdp')

        # once its generated:
        # to read the pomdp model
        model = pomdp_parser.Pomdp(filename='444_new.pomdp', parsing_print_flag=False)             # probably filename needs to be changed to a better one avoiding conflicts
        self.states_plus = model.states
        self.actions_plus = model.actions
        self.observations_plus = model.observations
        # print self.observations
        self.trans_mat_plus = model.trans_mat
        self.obs_mat_plus = model.obs_mat
        self.reward_mat_plus = model.reward_mat

        # to read the learned policy
        ##############################Saeid commented lines below ###################################
        #self.policy = policy_parser.Policy(len(self.states), len(self.actions), 
        #    filename=strategy+'_new.policy')
        # self.reinit_belief()

    ######################################################################
    # EXPERIMENTAL: Retrain parser:
    def retrain_parser(self):
        print "PARSER: retraining parser..."
        os.system('java -jar '+self.path_to_spf+' '+os.path.join(self.path_to_experiment,'init_train.exp'))

    #######################################################################
    def get_user_input(self, useFile=False):
        if useFile:
            user_input = "Test string"
        else:
            user_input = raw_input("Enter text: ")

        user_input = user_input.strip().lower()
        user_input = user_input.replace("'s"," s")
        user_input = user_input.translate(string.maketrans("",""), string.punctuation)

        self.full_request = user_input
        #log
        f = open(self.log_filename,'a')
        f.write("\t".join(["USER",user_input])+"\n")
        f.close()

        return [[user_input,0]] #full confidence value (log-probability) returned with text

    ######################################################################
    def get_observation_from_name(self, string):
        if string in self.known_words_to_number.keys():
            return self.known_words_to_number[string]
        else:
            return None

    def get_name_from_observation(self, string):
        for key,value in self.known_words_to_number.items():
            if value == string:
                return key

    #####################################################################
    def get_action(self, string):
        i = 0
        for action in self.actions:
            if action == string:
                return i
            i += 1
    def get_action_plus(self, string):
        i = 0
        for action in self.actions_plus:
            if action == string:
                return i
            i += 1

    def action_to_text(self, string):
        if string == 'ask_p':
            return "What item should I bring?"
        elif string == 'ask_r':
            return "Who should I bring the item to?"
        
        match = None
        match = re.search('(?<=confirm_)\w*', string)
        if match:
            obsname = match.group(0)
            return "confirm " + self.get_name_from_observation(obsname)

    ######################################################################
    def get_full_request(self, cycletime):
        print self.observations # debug
        user_utterances = self.get_user_input()
        parses_list = []
        unmapped_list = []

        for utterance,score in user_utterances:
            parses,unmapped = self.parse_utterance(utterance)
            parses_list.append(parses)
            unmapped_list.append(unmapped)
        
        #print parses_list,unmapped_list
        #print "action list: ", self.actions
        #print "selected: " + self.actions[self.a]
        patient = None
        recipient = None
        print "PARSES LIST: ",parses_list
        for parses in parses_list:
            for parse,score in parses:
                for word in str(parse).split():
                    match = None
                    #print word
                    match = re.search('\w*(?=:it.*)', word)
                    if match:
                        patient = match.group(0)
                        print "Patient: " + patient
                    match = None
                    match = re.search('\w*(?=:pe.*)', word)
                    if match:
                        recipient = match.group(0)
                        print "Recipient: " + recipient

        if patient:
            # get action from key
            self.a = self.get_action('ask_p')
            self.a_plus = self.get_action_plus('ask_p')

            # get observation from patient
            ind = self.known_words_to_number[patient]
            self.o = next(i for i in range(len(self.observations)) \
                if self.observations[i] == ind)

            # update for patient observation
            self.update(cycletime)

            # for belief plus
            self.o_plus = next(i for i in range(len(self.observations_plus)) \
                if self.observations_plus[i] == ind)

            # update for patient observation
            self.update_plus(cycletime)
            

        if recipient:
            # get action from key
            self.a = self.get_action('ask_r')
            self.a_plus = self.get_action_plus('ask_r')

            # get observation from patient
            ind = self.known_words_to_number[recipient]
            self.o = next(i for i in range(len(self.observations)) \
                if self.observations[i] == ind)

            # update for recipient observation
            self.update(cycletime)

            # for belief plus
            # get observation from patient
            ind = self.known_words_to_number[recipient]
            self.o_plus = next(i for i in range(len(self.observations_plus)) \
                if self.observations_plus[i] == ind)

            # update for recipient observation
            self.update_plus(cycletime)

        print "Unmapped: ",unmapped_list
##########################################################
    
    def add_new(self, raw_str):
        print "DEBUG: adding new"
        #file_init_train = open(os.path.join(self.path_to_experiment,'data','fold0_init_train.ccg'),'a')
        file_seed = open(os.path.join(self.path_to_experiment,'resources','seed.lex'),'a')
        file_nplist = open(os.path.join(self.path_to_experiment,'resources','np-list.lex'),'a')
        file_geo_consts = open(os.path.join(self.path_to_experiment,'resources','geo.consts.ont'),'r')
        lines = file_geo_consts.readlines()
        file_geo_consts.close()
        file_geo_consts_write = open(os.path.join(self.path_to_experiment,'resources','geo.consts.ont'),'w')
        file_geo_consts_write.writelines([item for item in lines[:-1]])

        if self.actions[self.a] == 'ask_p':
            self.num_patient += 1
            self.known_words_to_number[raw_str] = 'p'+str(self.num_patient - 1)
            file_seed.write(raw_str + " :- NP : " + raw_str + ":it\n")
            file_nplist.write(raw_str + " :- NP : " + raw_str + ":it\n")            
            file_geo_consts_write.write(raw_str + ":it\n")

        elif self.actions[self.a] == 'ask_r':
            self.num_recipient += 1
            self.known_words_to_number[raw_str] = 'r'+str(self.num_recipient - 1)
            file_seed.write(raw_str + " :- NP : " + raw_str + ":pe\n")
            file_nplist.write(raw_str + " :- NP : " + raw_str + ":pe\n")
            file_geo_consts_write.write(raw_str + ":pe\n")

        file_geo_consts_write.write(")\n")
        self.write_known_words_to_number()

        file_num_config = open(os.path.join(self.path_to_main,'data','num_config.txt'), 'w+')
        file_num_config.write(str(self.num_task) + " " + str(self.num_patient) + " " + str(self.num_recipient))
        file_num_config.close()
        file_geo_consts_write.close()
        file_nplist.close()
        file_seed.close()
        self.retrain_parser()

        #self.generate_new_model()

    #######################################################################
    def observe(self, ind):
        self.o = None

        if self.auto_observations:
            # not functional right now
            sys.exit("Error: Auto observation not implemented")
        else:
            # main part
            ind = self.get_observation_from_name(ind)

            if ind == None:
                print "DEBUG: Not found in list of observations"
                rand = numpy.random.random_sample()
                acc = 0.0
                for i in range(len(self.observations)):
                    acc += self.obs_mat[self.a, self.s, i]
                    if acc > rand:
                        self.o = i
                        break
                if self.o == None:
                    sys.exit('Error: observation is not properly sampled')
            else:
                self.o = next(i for i in range(len(self.observations)) \
                    if self.observations[i] == ind)


    #######################################################################
    def update(self,cycletime):

        new_b = numpy.dot(self.b, self.trans_mat[self.a, :])

        new_b = [new_b[i] * self.obs_mat[self.a, i, self.o] for i in range(len(self.states))]

        # print 'sum of belief: ',sum(new_b)

        self.b = (new_b / sum(new_b)).T

        if cycletime == self.trigger and self.use_plog and (self.md == 'sad' or self.fl == False):

            if self.b[len(self.tablelist)] == 1:
                return
            # print '\n',self.b
            #belief = self.plog.cal_belief(mood = self.md, foll = self.fl, pdpDist = self.b, curr_table = self.ct, prev_table = self.pt).split(',')
            # belief = self.plog.cal_belief(mood = 'sad', pdpDist = self.b, curr_table = self.ct).split(',')
            for i in range(len(belief)):
                belief[i] = float(belief[i].strip())
            self.b = numpy.array(belief)
            self.b = self.b/ sum(self.b)

    #######################################################################
    #######################################################################
    def update_plus(self,cycletime):
        print self.actions_plus[self.a_plus]
        if self.actions_plus[self.a_plus] == "ask_r" or self.actions_plus[self.a_plus] == "ask_p":
            return

        new_b_plus = numpy.dot(self.b_plus, self.trans_mat_plus[self.actions_plus.index(self.actions[self.a]), :])

        new_b_plus = [new_b_plus[i] * self.obs_mat_plus[self.actions_plus.index(self.actions[self.a]), i, self.observations_plus.index(self.observations[self.o]),] for i in range(len(self.states_plus))]

        # print 'sum of belief: ',sum(new_b)

        self.b_plus = (new_b_plus / sum(new_b_plus)).T


    def run(self):
        self.retrain_parser()

        cost = 0.0
        self.init_belief()
        self.init_belief_plus()

        reward = 0.0
        overall_reward = 0.0

        cycletime = 0

        current_entropy = float("inf")
        old_entropy = float("inf")
        inc_count = 0

        while True:
            cycletime += 1

            # print self.b

            if self.print_flag:
                print('\tstate:\t' + self.states[self.s] + ' ' + str(self.s))
                print('\tcost so far:\t' + str(cost))

            # select action
            # entropy
            old_entropy = current_entropy
            current_entropy = stats.entropy(self.b)
            current_entropy_plus = stats.entropy(self.b_plus)
            print "DEBUG: Entropy = ",current_entropy
            print "DEBUG: Entropy_plus = ",current_entropy_plus
            # check if entropy increased
            if (old_entropy < current_entropy):
                inc_count += 0
                print "DEBUG: entropy increased"

            if(current_entropy > 2.3):
                self.get_full_request(cycletime)
                if self.print_flag:
                    print('\nbelief:\t' + str(self.b))
                #self.update_plus(cycletime)
                if self.print_flag:
                    print('\nbelief_plus:\t' + str(self.b_plus))
            else:
                done = False
                self.a = int(self.policy.select_action(self.b))
                self.a_plus = self.actions_plus.index(self.actions[self.a])
            
                if self.print_flag:
                    print('\taction:\t' + self.actions[self.a] + ' ' + str(self.a))
                    
                    question = self.action_to_text(self.actions[self.a])
                    if question:
                        print('QUESTION: ' + question)
                    elif ('go' in self.actions[self.a]):
                        print('EXECUTE: ' + self.actions[self.a])
                        done = True


                if done == True:
                    break

                raw_str = raw_input("Input observation: ")

                # check entropy increases arbitrary no of times for now
                if (inc_count > 2):
                    print "--- new item/person ---"
                    self.add_new(raw_str)

                self.observe(raw_str)
                if self.print_flag:
                    print('\tobserve:\t'+self.observations[self.o]+' '+str(self.o))

                self.update(cycletime)
                if self.print_flag:
                    print('\nbelief:\t' + str(self.b))
                self.update_plus(cycletime)
                if self.print_flag:
                    print('\nbelief_plus:\t' + str(self.b_plus))


            overall_reward += self.reward_mat[self.a, self.s]
            # print('current cost: ' + str(self.reward_mat[self.a, self.s]))
            # print('overall cost: ' + str(overall_reward))
            # print self.actions[self.a]

            if 'go' in self.actions[self.a]:
                # print '--------------------',
                if self.print_flag is True:
                    print('\treward: ' + str(self.reward_mat[self.a, self.s]))
                reward += self.reward_mat[self.a, self.s]
                break
            else:
                cost += self.reward_mat[self.a, self.s]

            if cycletime == 20:
                cost += self.reward_mat[self.a, self.s]
                break

        return reward, cost, overall_reward

    #######################################################################
    def run_numbers_of_trials(self):

        cost_list = []
        success_list = []
        reward_list = []
        overall_reward_list = []

        string_i = ''
        string_p = ''
        string_r = ''
        
        bar = Bar('Processing', max=self.trials_num)

        for i in range(self.trials_num):

            # get a sample as the current state, terminal state exclusive
            if self.auto_state:
                self.s = numpy.random.randint(low=0, high=len(self.states)-1,
                    size=(1))[0]
                tuples = self.states[self.s].split('_')
                ids = [int(tuples[0][1]),int(tuples[1][1]),int(tuples[2][1])]
                self.ct = numpy.random.randint(low=0, high=len(self.tablelist),size=(1))[0] ###curr table
                self.pt = self.ct - 1 if self.ct != 0 else len(self.tablelist)-1
                self.md = 'happy'
                self.fl = True
                # print self.tablelist[self.ct], ids
                if self.tablelist[self.ct][0] != ids[0] and self.tablelist[self.ct][1] != ids[1] and self.tablelist[self.ct][2] != ids[2]:
                     self.md = 'sad'
                if self.tablelist[self.pt][0] == ids[0] and self.tablelist[self.pt][1] == ids[1] and self.tablelist[self.pt][2] == ids[2]:
                     self.fl = False
            else:
                self.s = int(input("Please specify the index of state: "))

            # run this episode and save the reward
            reward, cost, overall_reward = self.run()
            reward_list.append(reward)
            cost_list.append(cost)
            overall_reward_list.append(overall_reward)

            guide_index = int(self.a - (3 + self.num_task + self.num_patient \
                + self.num_recipient))

            if guide_index == int(self.s):
                success_list.append(1.0)
            else:
                success_list.append(0.0)

            bar.next()

        bar.finish()

        cost_arr = numpy.array(cost_list)
        success_arr = numpy.array(success_list)
        reward_arr = numpy.array(reward_list)
        overall_reward_arr = numpy.array(overall_reward_list)

        print('average cost: ' + str(numpy.mean(cost_arr))[1:] + \
            ' with std ' + str(numpy.std(cost_arr)))
        print('average success: ' + str(numpy.mean(success_arr)) + \
            ' with std ' + str(numpy.std(success_arr)))
        print('average reward: ' + str(numpy.mean(reward_arr)) + \
            ' with std ' + str(numpy.std(reward_arr)))
        print('average overall reward: ' + str(numpy.mean(overall_reward_arr)) + \
            ' with std ' + str(numpy.std(overall_reward_arr)))

        return (numpy.mean(cost_arr), numpy.mean(success_arr), \
            numpy.mean(reward_arr))

def main():
    # the number of variables are stored in this file for now
    f = open("./data/num_config.txt")
    num = f.readline().split()
    print num
    s = Simulator(uniform_init_belief = True, 
        auto_state = True, 
        auto_observations = False, # was true
        print_flag = True, 
        use_plog = False,
        policy_file = '333_new.policy', 
        pomdp_file =  '333_new.pomdp',
        trials_num = 10,
        num_task = int(num[0]), 
        num_patient = int(num[1]), 
        num_recipient = int(num[2]))
 
    if not s.uniform_init_belief:   
        print('note that initial belief is not uniform\n')
    s.generate_new_model()
    s.run_numbers_of_trials()
    #s.generate_new_model()
if __name__ == '__main__':
    main()

