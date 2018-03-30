#!/usr/bin/env python

import sys
import time
import pomdp_generator
import pomdp_parser
import policy_parser
import readline
import numpy
import random
from scipy import stats
from progress.bar import Bar
import subprocess
import gen_dis_plog
import conf
import re
import pandas as pd
import os
import string

import ast

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
        pomdp_file_plus=None,
        policy_file_plus=None,
        trials_num=1,
        num_task=1, 
        num_patient=1,
        num_recipient=1):

        self.pomdp_file_plus=pomdp_file_plus
        self.policy_file_plus=policy_file_plus
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
        #model_plus= pomdp_parser.Pomdp(filename='444_new.pomdp', parsing_print_flag=False)
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

    def init_belief_plus(self):

        if self.uniform_init_belief:
            self.b_plus = numpy.ones(len(self.states_plus)) / float(len(self.states_plus))
                # print '\n',self.s, self.ct, self.b
        else:
            # here initial belief is sampled from a Dirichlet distribution
            self.b_plus = numpy.random.dirichlet( numpy.ones(len(self.states_plus)) )

        self.b_plus = self.b_plus.T

    ######################################################################
    def read_model_plus(self):
       
        # once its generated:
        # to read the pomdp model
        model = pomdp_parser.Pomdp(filename=self.pomdp_file_plus, parsing_print_flag=False) # probably filename needs to be changed to a better one avoiding conflicts
        self.states_plus = model.states
        self.actions_plus = model.actions
        self.observations_plus = model.observations
        # print self.observations
        self.trans_mat_plus = model.trans_mat
        self.obs_mat_plus = model.obs_mat
        self.reward_mat_plus = model.reward_mat

        self.policy_plus = policy_parser.Policy(len(self.states_plus), len(self.actions_plus), 
            filename=self.policy_file_plus)

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
            return "confirm " + obsname

    
    ######################################################################
    def get_full_request(self, cycletime):
        print self.observations # debug
        print "DEBUG: Full request here"

        # patient
        # get action from key
        self.a = self.get_action('ask_p')
        self.a_plus = self.get_action_plus('ask_p')

        # if auto observe, get initial request
        if self.auto_observations:
            rand = numpy.random.random_sample()
            acc = 0.0
            for i in range(len(self.observations_plus)):
                acc += self.obs_mat_plus[self.a_plus, self.s_plus, i]
                if acc > rand:
                    raw_str = self.observations_plus[i]
                    self.request_p = raw_str
                    break
            if raw_str == None:
                sys.exit('Error: observation no sampled properly')
        else:
            raw_str = raw_input("Input observation (patient/item): ")

        ''' Note: Should we have a random observation for unknown here too
        like we do for the later phases?  For now I have implemented it
        that way, which is different from how we do it in the standard 
        'simulator.py' file, where we just observe a known or we don't 
        observe at all '''

        self.observe(raw_str)
        # update for patient observation
        self.update(cycletime)
        # for belief plus
        self.update_plus(cycletime)

        # recipient
        # get action from key
        self.a = self.get_action('ask_r')
        self.a_plus = self.get_action_plus('ask_r')

        if self.auto_observations:
            rand = numpy.random.random_sample()
            acc = 0.0
            for i in range(len(self.observations_plus)):
                acc += self.obs_mat_plus[self.a_plus, self.s_plus, i]
                if acc > rand:
                    raw_str = self.observations_plus[i]
                    self.request_r = raw_str
                    break
            if raw_str == None:
                sys.exit('Error: observation no sampled properly')
        else:
            raw_str = raw_input("Input observation (recipient/person): ")
        
        self.observe(raw_str)
        # update for recipient observation
        self.update(cycletime)
        # for belief plus
        self.update_plus(cycletime)

##########################################################

    
    def add_new(self, raw_str):
        print "DEBUG: adding new"
        self.num_patient += 1
        self.num_recipient += 1
        self.b = self.b_plus
        self.states = self.states_plus
        self.actions = self.actions_plus
        self.s = self.s_plus
        self.observations = self.observations_plus
        self.trans_mat = self.trans_mat_plus
        self.obs_mat = self.obs_mat_plus
        self.reward_mat = self.reward_mat_plus
        self.policy = self.policy_plus

    #######################################################################
    def auto_observe(self):
        rand = numpy.random.random_sample()
        acc = 0.0
        for i in range(len(self.observations_plus)):
            acc += self.obs_mat_plus[self.a_plus, self.s_plus, i]
            if acc > rand:
                raw_str = self.observations_plus[i]
                self.request_p = raw_str
                break
        
        if raw_str == None:
            sys.exit('Error: observation no sampled properly')

        return raw_str

    def observe(self, ind):
        self.o = None
        print "OBSERVE:",ind

        for i in range(len(self.observations)):
            if self.observations[i] == ind:
                self.o = i

        if self.o == None:
            q_type = str(self.actions[self.a][-1])

            domain = [self.observations.index(o) for o in self.observations if q_type in o]
            print domain
            self.o = random.choice(domain)
            print self.o


    #######################################################################
    def update(self,cycletime):
        new_b = numpy.dot(self.b, self.trans_mat[self.a, :])
        new_b = [new_b[i] * self.obs_mat[self.a, i, self.o] for i in range(len(self.states))]

        # print 'sum of belief: ',sum(new_b)
        self.b = (new_b / sum(new_b)).T


    def update_plus(self,cycletime):
        #print self.actions_plus[self.a_plus]
        if self.actions_plus[self.a_plus] == "ask_r" or self.actions_plus[self.a_plus] == "ask_p":
            return

        new_b_plus = numpy.dot(self.b_plus, self.trans_mat_plus[self.actions_plus.index(self.actions[self.a]), :])
        new_b_plus = [new_b_plus[i] * self.obs_mat_plus[self.actions_plus.index(self.actions[self.a]), i, self.observations_plus.index(self.observations[self.o]),] for i in range(len(self.states_plus))]

        # print 'sum of belief: ',sum(new_b)
        self.b_plus = (new_b_plus / sum(new_b_plus)).T


    def run(self):
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

        print 'num_recipients', self.num_recipient
        print 'num_patients', self.num_patient

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
                inc_count += 1
                print "DEBUG: entropy increased"

            if(current_entropy > 2.3):
                self.get_full_request(cycletime)
                if self.print_flag:
                    print('\n\tbelief:\t\t' + str(self.b))

                if self.print_flag:
                    print('\n\tbelief_plus:\t' + str(self.b_plus))
            else:
                done = False
                self.a = int(self.policy.select_action(self.b))
                self.a_plus = self.actions_plus.index(self.actions[self.a])
            
                if self.print_flag:
                    print('\taction:\t' + self.actions[self.a] + ' ' + str(self.a))

                    print 'num_recipients', self.num_recipient
                    print 'num_patients', self.num_patient

                    question = self.action_to_text(self.actions[self.a])
                    if question:
                        print('QUESTION: ' + question)
                    elif ('go' in self.actions[self.a]):
                        print('EXECUTE: ' + self.actions[self.a])
                        done = True


                if done == True:
                    break

                if self.auto_observations:
                    raw_str = self.auto_observe()
                else:
                    raw_str = raw_input("Input observation: ")

                # check entropy increases arbitrary no of times for now
                if (inc_count > 2  and added == False):
                    if (self.actions[self.a] == "ask_p" or self.actions[self.a] == "ask_r"):
                        print "--- new item/person ---"
                        added = True
                        self.add_new(raw_str)

                self.observe(raw_str)
                if self.print_flag:
                    print('\tobserve:\t'+self.observations[self.o]+' '+str(self.o))

                self.update(cycletime)
                if self.print_flag:
                    print('\n\tbelief:\t\t' + str(self.b))
                self.update_plus(cycletime)
                if self.print_flag:
                    print('\n\tbelief_plus:\t' + str(self.b_plus))


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

            self.s_plus = self.states_plus.index(self.states[self.s])

            # run this episode and save the reward
            reward, cost, overall_reward = self.run()
            reward_list.append(reward)
            cost_list.append(cost)
            overall_reward_list.append(overall_reward)
 
            guide_index = int(self.a - (3 + self.num_task + self.num_patient \
               + self.num_recipient))

            print 'self.a', self.a
            print 'self.num_task', self.num_task
            print 'num_recipient', self.num_recipient
            print 'num_patient', self.num_patient
            print 'guide index: ' , guide_index
            print 'state index: ', self.s
            print self.states
            print self.actions

            if guide_index == int(self.s):
                success_list.append(1.0)
            else:
                success_list.append(0.0)

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

        return (numpy.mean(cost_arr), numpy.mean(success_arr), \
            numpy.mean(overall_reward_arr))

def main():
    # the number of variables are stored in this file for now
    #f = open("./data/num_config.txt")
    #num = f.readline().split()
    #print num
    num=300
    df=pd.DataFrame() 
    for name in ['133', '144','155','166','177','188']:
   # for name in ['133']:

        s = Simulator(uniform_init_belief = True, 
            auto_state = True, 
            auto_observations = True, # was true
            print_flag = True, 
            use_plog = False,
            policy_file = name+'_new.policy',
            pomdp_file =  name +'_new.pomdp',
                pomdp_file_plus=list(name)[0]+str(int(list(name)[1])+1)+str(int(list(name)[2])+1)+'_new.pomdp',
                policy_file_plus=list(name)[0]+str(int(list(name)[1])+1)+str(int(list(name)[2])+1)+'_new.policy',
            trials_num = num,
            num_task = int(name[0]), 
            num_patient = int(name[1]), 
            num_recipient = int(name[2]))
     
        if not s.uniform_init_belief:   
            print('note that initial belief is not uniform\n')
        s.read_model_plus()
        a,b,c=s.run_numbers_of_trials()
        df.at[name,'Overall Cost']= a
        df.at[name,'Overall Success']= b
        df.at[name,'Overall Reward']= c
    print df
    
    fig=plt.figure(figsize=(8,3))
    for count,metric in enumerate(list(df)):
        ax=plt.subplot(1,len(list(df)),count+1)

        l1 = plt.plot([3,4,5,6,7,8],df.loc['133':'188',metric],marker='*',linestyle='-',label='Average of '+str(num)+ ' trials')
        plt.ylabel(metric)
        plt.xlim(2.5,8.5)
        xleft , xright =ax.get_xlim()
        ybottom , ytop = ax.get_ylim()
        ax.set_aspect(aspect=abs((xright-xleft)/(ybottom-ytop)), adjustable=None, anchor=None)


        plt.xlabel('Recipients/Patients')


    ax.legend(loc='upper left', bbox_to_anchor=(-2.10, 1.35),  shadow=True, ncol=5)
    fig.tight_layout()
    plt.show()
    fig.savefig('Results_'+str(num)+'_trials')
    
if __name__ == '__main__':
    main()
