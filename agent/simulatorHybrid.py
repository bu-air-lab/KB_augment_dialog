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
#sys..append('/home/ludc/software/python_progress/progress-1.2')
#sys.path.append('/home/szhang/software/python_progress/progress-1.2')
#from progress.bar import Bar
import subprocess
import re
import os
import string
import ast
import datetime

from allennlp.predictors.predictor import Predictor
from clyngor import ASP, solve

numpy.set_printoptions(suppress=True)

class Simulator(object):

    def __init__(self, 
        auto_observations=True, 
        auto_state = False, 
        uniform_init_belief =True,
        print_flag=True,
        policy_file='policy/default.policy', 
        pomdp_file='models/default.pomdp',
        pomdp_file_plus=None,
        policy_file_plus=None,
        trials_num=1,
        num_task=1, 
        num_patient=1,
        num_recipient=1,
        belief_threshold=0.4,
        ent_threshold=2):
        devnull = open(os.devnull, 'wb')
        # print(pomdp_file)
        # print(policy_file)
        # generate main model
        self.generate_model(num_task, num_patient, num_recipient, pomdp_file,policy_file, False)
        now = datetime.datetime.now().strftime("%I_%M%p_%B_%d_%Y")+"/"
        subprocess.Popen(['cp','-r', 'spf/',now],stdout=devnull, stderr=devnull)
        self.pomdp_file_plus=pomdp_file_plus
        self.known_words_path = '/home/cihangir/KB_augment_dialog/agent/data'
        self.policy_file_plus=policy_file_plus
        self.auto_observations = auto_observations
        self.auto_state = auto_state
        self.uniform_init_belief = uniform_init_belief
        self.print_flag = print_flag
        self.trials_num = trials_num
        self.belief_threshold = belief_threshold
        self.ent_threshold = ent_threshold
        self.num_task = num_task
        self.num_patient = num_patient
        self.num_recipient = num_recipient
        

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
        self.spffolder=now
        self.b = None   
        self.b_plus = None   
        self.a = None
        self.a_plus=None
        self.o = None
        self.o_plus= None
        # self.dialog_turn = 0

        # Generating the plus model that considers one unknown item and one unknown person
        self.generate_model(num_task, num_patient+1, num_recipient+1, pomdp_file_plus,policy_file_plus, False)
        self.read_plus_model()

        # for semantic parser
        self.path_to_main = os.path.dirname(os.path.abspath(__file__))
        self.log_filename = os.path.join(self.path_to_main,'data','log','log.txt')

        #path to SPF jar
        self.path_to_spf = os.path.join(self.path_to_main,self.spffolder,'dist','spf-1.5.5.jar')
        #path to write-able experiment directory
        self.path_to_experiment = os.path.join(self.path_to_main,self.spffolder,'geoquery','experiments','template','dialog_writeable')
        # known words and
        given_words,word_to_ontology_map = self.get_known_words_from_seed_files()
        self.known_words = given_words

        # full request string
        self.full_request = ''
        
        self.known_words_to_number = {}
        self.get_known_words_to_number()

        self.synonym = {
            'apple':['fruit', 'red apple', 'fuji', 'red delicious'],
            'coffee':['cappuccino', 'java', 'decaf', 'cup', 'mug', 'caffeine'],
            'hamburger':['burger', 'big mac', 'sandwich'],
            'phone':['cell', 'cell phone', 'iphone', 'mobile', 'mobile phone'],
            'soda':['pop','coke','can','drink','soft drink','cold drink'],
            'alice':['Alice', 'Alice Anderson', 'alice anderson'],
            'bob':['Bob', 'Bob Brown', 'bob brown'],
            'carol':['Carol', 'Carol Clark', 'carol clark'],
            'dennis':['Dennis', 'Dennis Davis', 'dennis davis'],
            'ellen':['Ellen', 'Ellen Edwards', 'ellen edwards'],
            'yes':['yup', 'yeah', 'thats right', 'correct'],
            'no':['nope', 'nah', 'thats wrong', 'incorrect']
        }

        # to plot entropy
        self.entropy_list = []
        self.added_point = None
        self.question_list = []
        self.answer_list = []

        # to make the screen print simple 
        numpy.set_printoptions(precision=2)






    #######################################################################
    def solveKB(self):
        answers = solve('rules.lp') #Return value is frozenset, thats the reason it is converted to predicate and entity lists.

        self.entityList = list()
        predicateList = list()

        for answer in answers:
            for line in answer:
                predicateList.append(''.join(line[0]))
                self.entityList.append(''.join(line[1]))


        print("Predicate list of KB: ")
        print(predicateList)
        print()
        print("Entity list of KB: ")
        print(self.entityList)
        

    #######################################################################
    def init_belief(self):

        if self.uniform_init_belief:
            self.b = numpy.ones(len(self.states)) / float(len(self.states))
                # print '\n',self.s, self.ct, self.b
        else:
            # here initial belief is sampled from a Dirichlet distribution
            self.b = numpy.random.dirichlet( numpy.ones(len(self.states)) )

        self.b = self.b.T

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
        file_known = open(os.path.join(self.path_to_main,self.known_words_path,'known_words_to_obs.txt'), 'r')
        s = file_known.read()
        self.known_words_to_number = ast.literal_eval(s)
        if self.print_flag:
            print(str(self.known_words_to_number))
        file_known.close()

    def write_known_words_to_number(self):
        "DEBUG: saving known words to observations to file"
        file_known = open(os.path.join(self.path_to_main,self.known_words_path,'known_words_to_obs.txt'), 'w+')
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
                        print (s) #DEBUG
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
    def generate_model(self, num_task, num_patient, num_recipient, file_name,policy_file, is_plus):
        r_max = 40.0
        r_min = -40.0

        wh_cost = -1.5
        yesno_cost = -1.0
        strategy=str(num_task)+str(num_patient)+str(num_recipient)
        pg = pomdp_generator.PomdpGenerator(num_task, num_patient, num_recipient, r_max, r_min, strategy, \
            wh_cost, yesno_cost,pomdpfile = file_name,policyfile=policy_file,timeout=50, is_plus=is_plus)

        # to read the learned policy
        ##############################Saeid commented lines below ###################################
        #self.policy = policy_parser.Policy(len(self.states), len(self.actions), 
        #    filename=strategy+'_new.policy')
        # self.reinit_belief()

    def read_plus_model(self):
        # to read the pomdp model
        model = pomdp_parser.Pomdp(filename=self.pomdp_file_plus, parsing_print_flag=False) 
        self.states_plus = model.states
        self.actions_plus = model.actions
        self.observations_plus = model.observations
        # print self.observations
        self.trans_mat_plus = model.trans_mat
        self.obs_mat_plus = model.obs_mat
        self.reward_mat_plus = model.reward_mat
        self.policy_plus = policy_parser.Policy(len(self.states_plus), len(self.actions_plus), 
            filename=self.policy_file_plus)

    ######################################################################
    # EXPERIMENTAL: Retrain parser:
    def retrain_parser(self):
        print ("PARSER: retraining parser...")
        os.system('java -jar '+self.path_to_spf+' '+os.path.join(self.path_to_experiment,'init_train.exp'))

    #######################################################################
    def get_string(self, question):
        # this method can be overriden when needed (eg. use a gui wrapper)
        politeness = "I know I asked it already but could you "
        tmp = question
        if(self.question_list.count(question) >= 1):
            tmp = politeness + question
        print ("QUESTION: " + tmp)
        answer = input().lower()

        # to log or entropy plot
        self.question_list.append(question)
        self.answer_list.append(answer)

        return answer


    def print_message(self, message):
        # this method can be overriden when needed
        print (message)

    #######################################################################
    def get_user_input(self, question, useFile=False):
        if useFile:
            user_input = "Test string"
        else:
            user_input = self.get_string(question)

        user_input = user_input.strip().lower()
        user_input = user_input.replace("'s"," s")
        #user_input = user_input.translate(string.maketrans("",""), string.punctuation)

        self.full_request = user_input
        #log
        f = open(self.log_filename,'a')
        f.write("\t".join(["USER",user_input])+"\n")
        f.close()

        return [[user_input,0]]

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
            if 'p' in obsname:
                return "Do you want me to deliver " + self.get_name_from_observation(obsname) + "?"
            elif 'r' in obsname:
                return "Is this delivery for " + self.get_name_from_observation(obsname) + "?"

        if 'go' in string:
            parts = string.split('_')
            return self.get_name_from_observation(parts[1]) + " " \
                 + self.get_name_from_observation(parts[2]) + " for " \
                 + self.get_name_from_observation(parts[3])


    def resolve_synonym(self, string):
        # only for experiment
        # this is a workaround for the semantic parser not returning for single phrases
        # this will be replaced/removed when we fix the parser or use a different one

        for key in self.synonym:
            synlist = self.synonym[key]
            if string in synlist:
                return key

        return string


    ######################################################################
    def get_full_request(self, cycletime): #### EDITED


        
        #start_time = time.time()
        
        #print("\n--- %s seconds ---\n	" % (time.time() - start_time))	

        KBActionList = ["task", "item", "recipient"]
        KBDict = dict.fromkeys(KBActionList) #Initialize dict.


        if self.print_flag:
            print (self.observations) # debug

        user_utterances = self.get_user_input("How can I help you?")
        parses_list = []
        unmapped_list = []

        neuralParserInput = user_utterances[0][0]
        print("DEBUG: Neural Parser Input: ")
        print(neuralParserInput)

        for utterance,score in user_utterances:
            parses,unmapped = self.parse_utterance(utterance)
            parses_list.append(parses)
            unmapped_list.append(unmapped)
        
        patient = None
        recipient = None

        if self.print_flag:
            print ("PARSES LIST: ",parses_list)

        for parses in parses_list:
            for parse,score in parses:
                for word in str(parse).split():
                    match = None
                    #print word
                    match = re.search('\w*(?=:it.*)', word)
                    if match: # Add back up as 'else'
                        patient = match.group(0)
                        if self.print_flag:
                            print ("Patient: " + patient)
                    match = None
                    match = re.search('\w*(?=:pe.*)', word)
                    if match: # Add back up as 'else'
                        recipient = match.group(0)
                        if self.print_flag:
                            print ("Recipient: " + recipient)
    


        if patient is None or recipient is None:
            predictor = Predictor.from_path("./model.gz")
            print("DEBUG: SPF can not parse utterance, thus neural parser called.\n")
            predicted = predictor.predict(sentence=neuralParserInput)
            for key, value in predicted.items(): #Preparation of data for 'POS_Word' dict.
                print(key, value)
                if(key == 'words'): #Read space seperated words
                    valueList = predicted[key]
                if(key == 'predicted_dependencies'): #Read dependency tags
                    keyList = predicted[key]
            depWordList = list(zip(keyList, valueList)) #Zip two list and make another list. (CAN NOT BE DICT BECAUSE SAME POS TAG CAN OCCUR TWICE).
            print()
            for key, value in depWordList: #Creation of 'DepTag_Word' dict. (Because of the specific domain, we know POS tags for task, item and recipient.)
                print(key, value)
                if key == "root":
                    KBDict["task"] = value
                if key == "dobj":
                    KBDict["item"] = value
                if key == "pobj" or key == "xcomp":
                    KBDict["recipient"] = value
            
            if patient is None:
                if KBDict["item"] in self.entityList:
                    patient = KBDict["item"]
                    if self.print_flag:
                        print ("\nPatient(Neural Parser): " + patient)   

            if recipient is None:
                if KBDict["recipient"] in self.entityList:
                    recipient = KBDict["recipient"]
                    if self.print_flag:
                        print ("\Recipient(Neural Parser): " + recipient)   
                        

                        


        if patient:
            # get action from key
            self.a = self.get_action('ask_p')
            self.a_plus = self.get_action_plus('ask_p')
            # get observation from patient
            self.observe(patient)
            self.update(cycletime)
            # update for b+
            self.update_plus(cycletime)
        else:
            self.a = self.get_action('ask_p')
            self.a_plus = self.get_action_plus('ask_p')
            # random/unknown observation
            self.observe(None)
            self.update(cycletime)
            self.update_plus(cycletime)
            

        if recipient:
            # get action from key
            self.a = self.get_action('ask_r')
            self.a_plus = self.get_action_plus('ask_r')
            # get observation from patient
            self.observe(recipient)
            # update for recipient observation
            self.update(cycletime)
            # update for b+
            self.update_plus(cycletime)
        else:
            self.a = self.get_action('ask_r')
            self.a_plus = self.get_action_plus('ask_r')
            self.observe(None)
            self.update(cycletime)
            self.update_plus(cycletime)

        #print "Unmapped: ",unmapped_list

    def get_partial(self, question): #### EDITED



        #predictor = Predictor.from_path("./model.gz")
        #start_time = time.time()
        
        #print("\n--- %s seconds ---\n	" % (time.time() - start_time))	

        KBActionList = ["task", "item", "recipient"]
        KBDict = dict.fromkeys(KBActionList) #Initialize dict.


        if 'confirm' in self.actions[self.a]:
            raw_str = self.get_string(question)
            raw_str = self.resolve_synonym(raw_str)
            return raw_str

        user_utterances = self.get_user_input(question)
        parses_list = []
        unmapped_list = []

        neuralParserInput = user_utterances[0][0]
        print("DEBUG: Neural Parser Input: " + neuralParserInput)

        for utterance,score in user_utterances:
            parses,unmapped = self.parse_utterance(utterance)
            parses_list.append(parses)
            unmapped_list.append(unmapped)
        
        patient = None
        recipient = None

        if self.print_flag:
            print ("PARSES LIST: ",parses_list)

        for parses in parses_list:
            for parse,score in parses:
                for word in str(parse).split():
                    match = None
                    #print word
                    match = re.search('\w*(?=:it.*)', word)
                    if match: # Add back up as 'else'
                        patient = match.group(0)
                        if self.print_flag:
                            print ("Patient: " + patient)

                    match = None
                    match = re.search('\w*(?=:pe.*)', word)
                    if match: # Add back up as 'else'
                        recipient = match.group(0)
                        if self.print_flag:
                            print ("Recipient: " + recipient)



        if patient is None or recipient is None:
            predictor = Predictor.from_path("./model.gz")
            print("DEBUG: SPF can not parse utterance, thus neural parser called.\n")
            predicted = predictor.predict(sentence=neuralParserInput)
            for key, value in predicted.items(): #Preparation of data for 'POS_Word' dict.
                print(key, value)
                if(key == 'words'): #Read space seperated words
                    valueList = predicted[key]
                if(key == 'predicted_dependencies'): #Read dependency tags
                    keyList = predicted[key]
            depWordList = list(zip(keyList, valueList)) #Zip two list and make another list. (CAN NOT BE DICT BECAUSE SAME POS TAG CAN OCCUR TWICE).
            print()
            for key, value in depWordList: #Creation of 'DepTag_Word' dict. (Because of the specific domain, we know POS tags for task, item and recipient.)
                print(key, value)
                if key == "root": # if input is one work, it parsed as root and patient or recipient values assigned to None, thus overall algorithm stay unchanged.
                    KBDict["task"] = value
                if key == "dobj":
                    KBDict["item"] = value
                if key == "pobj" or key == "xcomp":
                    KBDict["recipient"] = value
            
            if patient is None:
                if KBDict["item"] in self.entityList:
                    patient = KBDict["item"]
                    if self.print_flag:
                        print ("\nPatient(Neural Parser): " + patient)   

            if recipient is None:
                if KBDict["recipient"] in self.entityList:
                    recipient = KBDict["recipient"]
                    if self.print_flag:
                        print ("\nRecipient(Neural Parser): " + recipient)   
                        
              



        # workaround for experiment only
        utterance = self.resolve_synonym(utterance)

        if self.actions[self.a] == 'ask_r':
            if recipient:
                return recipient
            else:
                print (utterance)
                return utterance
        elif self.actions[self.a] == 'ask_p':
            if patient:
                return patient
            else:
                print (utterance)
                return utterance



##########################################################
    def reinit_belief(self, added_type):
        m = self.num_patient + 1
        n = self.num_recipient + 1
        if added_type == 'p':
            for i in range(m):
                self.b_plus[(n * i + (n - 1))] = 0
        elif added_type == 'r':
            for i in range(n):
                self.b_plus[(n * (m - 1) + i)] = 0

        b_sum = sum(self.b_plus)

        # renormalize
        for i in range(len(self.b_plus)):
            self.b_plus[i] = self.b_plus[i]/b_sum

    
    def add_new(self):
        print ("DEBUG: adding new..")
        #file_init_train = open(os.path.join(self.path_to_experiment,'data','fold0_init_train.ccg'),'a')
        file_seed = open(os.path.join(self.path_to_experiment,'resources','seed.lex'),'a')
        file_nplist = open(os.path.join(self.path_to_experiment,'resources','np-list.lex'),'a')
        file_geo_consts = open(os.path.join(self.path_to_experiment,'resources','geo.consts.ont'),'r')
        lines = file_geo_consts.readlines()
        file_geo_consts.close()
        file_geo_consts_write = open(os.path.join(self.path_to_experiment,'resources','geo.consts.ont'),'w')
        file_geo_consts_write.writelines([item for item in lines[:-1]])

        #if self.actions[self.a] == 'ask_p':
        belief_rn, belief_pm = self.get_marginal_edges(self.b_plus, self.num_recipient+1, self.num_patient+1)
        print ("marginal rn", belief_rn)
        print ("marginal pm", belief_pm)
        if belief_pm > belief_rn:
            raw_str = self.get_string("It seems I do not know the item you are talking about.  Please write the name of the item so I can learn it.")
            first = raw_str.strip().split()[0]
            with open("rules.lp", "a") as KBFile:
                KBFile.write("item(%s).\n" % first)
            self.reinit_belief('p')
            self.num_patient += 1
            self.known_words_to_number[first] = 'p'+str(self.num_patient - 1)
            file_seed.write(raw_str + " :- NP : " + first + ":it\n")
            file_nplist.write(raw_str + " :- NP : " + first + ":it\n")            
            file_geo_consts_write.write(first + ":it\n")
            self.synonym[first] = raw_str
            
        #elif self.actions[self.a] == 'ask_r':
        else:
            raw_str = self.get_string("It seems I do not know the person you are talking about.  Please write their name so I can learn it.")
            first = raw_str.strip().split()[0]
            with open("rules.lp", "a") as KBFile:
                KBFile.write("recipient(%s).\n" % first)
            self.reinit_belief('r')
            self.num_recipient += 1
            self.known_words_to_number[first] = 'r'+str(self.num_recipient - 1)
            file_seed.write(raw_str + " :- NP : " + first + ":pe\n")
            file_nplist.write(raw_str + " :- NP : " + first + ":pe\n")
            file_geo_consts_write.write(first + ":pe\n")
            self.synonym[first] = raw_str
            

        file_geo_consts_write.write(")\n")
        self.write_known_words_to_number()

        file_num_config = open(os.path.join(self.path_to_main,'data','num_config.txt'), 'w+')
        file_num_config.write(str(self.num_task) + " " + str(self.num_patient) + " " + str(self.num_recipient))
        file_num_config.close()
        file_geo_consts_write.close()
        file_nplist.close()
        file_seed.close()
        self.retrain_parser()

        #self.num_patient += 1
        #self.num_recipient += 1
        self.b = self.b_plus
        self.states = self.states_plus
        self.actions = self.actions_plus
        self.observations = self.observations_plus
        self.trans_mat = self.trans_mat_plus
        self.obs_mat = self.obs_mat_plus
        self.reward_mat = self.reward_mat_plus
        self.policy = self.policy_plus

        # generate new plus model
        #self.generate_model(self.num_task, self.num_patient+1, self.num_recipient+1, self.pomdp_file_plus, True)
        #self.read_plus_model()

    #######################################################################
    def observe(self, raw_str):
        self.o = None

        ind = self.get_observation_from_name(raw_str)

        if ind == None:
            if self.print_flag:
                print ("DEBUG: Not found in list of observations")

            if 'confirm' in self.actions[self.a]:
                self.o = self.observations.index(numpy.random.choice(['yes', 'no']))
                return

            q_type = str(self.actions[self.a][-1])
            domain = [self.observations.index(o) for o in self.observations if q_type in o]
            #print domain
            self.o = numpy.random.choice(domain)
        else:
            for i in range(len(self.observations)):
                if self.observations[i] == ind:
                    self.o = i


    #######################################################################
    def update(self,cycletime):
        new_b = numpy.dot(self.b, self.trans_mat[self.a, :])
        new_b = [new_b[i] * self.obs_mat[self.a, i, self.o] for i in range(len(self.states))]
        self.b = (new_b / sum(new_b)).T


    #######################################################################
    #######################################################################
    def update_plus(self,cycletime):
        #print self.actions_plus[self.a_plus]
        if self.actions_plus[self.a_plus] == "ask_r" or self.actions_plus[self.a_plus] == "ask_p":
            return

        new_b_plus = numpy.dot(self.b_plus, self.trans_mat_plus[self.actions_plus.index(self.actions[self.a]), :])
        new_b_plus = [new_b_plus[i] * self.obs_mat_plus[self.actions_plus.index(self.actions[self.a]), i, self.observations_plus.index(self.observations[self.o]),] for i in range(len(self.states_plus))]

        # print 'sum of belief: ',sum(new_b)
        self.b_plus = (new_b_plus / sum(new_b_plus)).T


    def entropy_check(self, entropy):
        x = self.num_recipient * self.num_patient
        if entropy > ((-0.00104921 * (x ** 2)) + (0.0916123 * x) + 1.21017):
            return True

        return False


    # for this domain n = num_patients, m = num_patients in the belief distribution
    def get_marginal_edges(self, b, n, m):
        belief_rn = 0
        for i in range(m):
            belief_rn += b[n * i + n - 1]

        belief_pm = 0
        for i in range(n):
            belief_pm += b[n * (m - 1) + i]

        return belief_rn, belief_pm


    def belief_check(self):
        n = self.num_recipient + 1
        m = self.num_patient + 1

        belief_rn, belief_pm = self.get_marginal_edges(self.b_plus, n, m)

        if self.print_flag:
            print ("DEBUG: Marginal rn = ",belief_rn)
            print ("DEBUG: Marginal pm = ",belief_pm)

        if belief_rn > self.belief_threshold or belief_pm > self.belief_threshold:
            return True

        return False


    def run(self):
        # seed random for experiments
        numpy.random.seed()
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
        added = False
        self.solveKB()

        while True:
            cycletime += 1

            # print self.b

            ##if self.print_flag:
                ##print('\tstate (plus):\t' + self.states_plus[self.s_plus] + ' ' + str(self.s_plus))
                ##print('\tcost so far:\t' + str(cost))

            # select action
            # entropy

            old_sign = self.sign(current_entropy - old_entropy)
            old_entropy = current_entropy
            current_entropy = stats.entropy(self.b)
            current_entropy_plus = stats.entropy(self.b_plus)
            
            if self.print_flag:
                print ("DEBUG: Entropy = ",current_entropy)
                print ("DEBUG: Entropy_plus = ",current_entropy_plus)

            self.entropy_list.append(current_entropy)
            
            # check if entropy increased
            if (self.sign(current_entropy - old_entropy) != old_sign):
                inc_count += 1
                if self.print_flag:
                    print ("DEBUG: entropy fluctuated")

            if (self.entropy_check(current_entropy)):
                self.get_full_request(cycletime)
                if self.print_flag:
                    print('\nbelief:\t' + str(self.b))
                    print('\nbelief+:\t' + str(self.b_plus))
            else:
                done = False
                self.a = int(self.policy.select_action(self.b))
                self.a_plus = self.actions_plus.index(self.actions[self.a])
            
                if self.print_flag:
                    print('\taction:\t' + self.actions[self.a] + ' ' + str(self.a))
                    
                    print ('num_recipients', self.num_recipient)
                    print ('num_patients', self.num_patient)

                # check entropy increases arbitrary no of times for now
                if (added == False):
                    if(inc_count > self.ent_threshold or self.belief_check()):
                        print ("--- new item/person ---")
                        added = True
                        self.added_point = (cycletime-1, current_entropy)
                        self.add_new()

                if ('go' in self.actions[self.a]):
                    self.print_message('EXECUTE: ' + self.action_to_text(self.actions[self.a]))
                    done = True
                else:
                    #raw_str = self.get_string(self.action_to_text(self.actions[self.a]))
                    raw_str = self.get_partial(self.action_to_text(self.actions[self.a]))

                if done == True:
                    break

                self.observe(raw_str)
                if self.print_flag:
                    print('\tobserve:\t'+self.observations[self.o]+' '+str(self.o))

                self.update(cycletime)
                if self.print_flag:
                    print('\n\tbelief: ' + str(self.b))

                self.update_plus(cycletime)
                if self.print_flag:
                    print('\n\tbelief+: ' + str(self.b_plus))


            ## overall_reward += self.reward_mat_plus[self.a_plus, self.s_plus]
            # print('current cost: ' + str(self.reward_mat[self.a, self.s]))
            # print('overall cost: ' + str(overall_reward))
            # print self.actions[self.a]

            if self.a_plus == None:
                continue

            if 'go' in self.actions_plus[self.a_plus]:
                # print '--------------------',
                ##if self.print_flag is True:
                    ##print('\treward: ' + str(self.reward_mat_plus[self.a_plus, self.s_plus]))
                ##reward += self.reward_mat_plus[self.a_plus, self.s_plus]
                break
            ##else:
                ##cost += self.reward_mat_plus[self.a_plus, self.s_plus]

            if cycletime == 20:
                ##cost += self.reward_mat_plus[self.a_plus, self.s_plus]
                break

        ##return reward, cost, overall_reward, added
        # entropy data to plot
        '''
        f = open('entropy_plot/entropy.txt', 'w')
        for item in self.entropy_list:
            f.write("%s\n" % item)
        d.close()

        f = open('entropy_plot/question.txt', 'w')
        for item in self.question_list:
            f.write("%s\n" % item)
        d.close()

        f = open('entropy_plot/answer.txt', 'w')
        for item in self.answer_list:
            f.write("%s\n" % item)
        d.close()

        f = open('entropy_plot/added.txt', 'w')
        f.write("%s\n" % self.added_point)
        d.close()
        '''

        return

    def sign(self,x):

        if x>0:
            return 1
        elif x<0:
            return -1
        else:
            return 0

    #######################################################################
    
    def run_numbers_of_trials(self):

        cost_list = []
        success_list = []
        reward_list = []
        overall_reward_list = []
        
        # for new item or person
        true_positives = 0.0
        false_positives = 0.0
        true_negatives = 0.0
        false_negatives = 0.0

        string_i = ''
        string_p = ''
        string_r = ''

        # save initial values to reset before next run
        initial_num_recipient = self.num_recipient
        initial_num_patient = self.num_patient
        initial_states = self.states
        initial_actions = self.actions
        initial_observations = self.observations
        initial_trans_mat = self.trans_mat
        initial_obs_mat = self.obs_mat
        initial_reward_mat = self.reward_mat
        initial_policy = self.policy

        
        bar = Bar('Processing', max=self.trials_num)

        for i in range(self.trials_num):

            # seed random for experiments
            numpy.random.seed(i+9309)

            # get a sample as the current state, terminal state exclusive
            if self.auto_state:
                # 50% chance fixed to select unknown state
                unknown_state = numpy.random.choice([True, False])

                if unknown_state == False:
                    self.s = numpy.random.randint(low=0, high=len(self.states)-1, size=(1))[0]
                    tuples = self.states[self.s].split('_')
                    ids = [int(tuples[0][1]),int(tuples[1][1]),int(tuples[2][1])]
                    self.s_plus = self.states_plus.index(self.states[self.s])
                else:
                    unknown_set = set(self.states_plus) - set(self.states)
                    unknown_set = list(unknown_set)
                    selected = numpy.random.choice(unknown_set)
                    self.s_plus = self.states_plus.index(selected)

            else:
                self.s_plus = int(input("Please specify the index of state: "))

            #!!! important note: State self.s not used as goal anymore, since we need new items to be possible as well,
            #instead self.s_plus is used to compare 

            #self.s_plus = self.states_plus.index(self.states[self.s])
            print (self.states_plus[self.s_plus])
            print (self.states)
            if str(self.states_plus[self.s_plus]) in self.states:
                is_new = False
            else:
                is_new = True

            # run this episode and save the reward
            reward, cost, overall_reward, added = self.run()
            reward_list.append(reward)
            cost_list.append(cost)
            overall_reward_list.append(overall_reward)
 

            # use string based checking of success for now
            if (str(self.states_plus[self.s_plus]) in self.actions[self.a]) and (is_new == added):
                success_list.append(1.0)
            else:
                success_list.append(0.0)

            if is_new == True and added == True:
                true_positives += 1
            elif is_new == True and added == False:
                false_negatives += 1
            elif is_new == False and added == True:
                false_positives += 1
            elif is_new == False and added == False:
                true_negatives += 1

            # reset for next run

            self.num_patient = initial_num_patient
            self.num_recipient = initial_num_recipient
            self.num_recipient = initial_num_recipient
            self.num_patient = initial_num_patient
            self.states = initial_states
            self.actions = initial_actions
            self.observations = initial_observations
            self.trans_mat = initial_trans_mat
            self.obs_mat = initial_obs_mat
            self.reward_mat = initial_reward_mat
            self.policy = initial_policy

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

        print('True positives (%):' + str(true_positives))
        print('False positives (%):' + str(false_positives))
        print('True negatives (%):' + str(true_negatives))
        print('False negatives (%):' + str(false_negatives))

        precision = true_positives/(true_positives + false_positives)
        recall = true_positives/(true_positives + false_negatives)
        print('Precision:' + str(precision))
        print('Recall:' + str(recall))

        return (numpy.mean(cost_arr), numpy.mean(success_arr), \
            numpy.mean(overall_reward_arr), precision, recall)
    


def main():
    # the number of variables are stored in this file for now
    f = open("./data/num_config.txt")
    num = f.readline().split()

    s = Simulator(uniform_init_belief = True, 
        auto_state = True, 
        auto_observations = False, # was true
        print_flag = True, 
        policy_file = '155.policy', 
        pomdp_file =  '155.pomdp',
        policy_file_plus = '166.policy',
        pomdp_file_plus = '166.pomdp',
        trials_num = 1,
        num_task = int(num[0]), 
        num_patient = int(num[1]), 
        num_recipient = int(num[2]),
        belief_threshold = 0.4,
        ent_threshold = 2)
 
    if not s.uniform_init_belief:   
        print('note that initial belief is not uniform\n')

    ##s.run_numbers_of_trials()
    s.run()

if __name__ == '__main__':
    main()

