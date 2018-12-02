from simulator_noparser import Simulator
import sys
import time
import pomdp_generator
import pomdp_parser
import policy_parser
import readline
import numpy
import random
from scipy import stats
#from progress.bar import Bar
import subprocess
#import conf
import re
import os
import string
import time
import ast

class Baseline(Simulator):



    def run_numbers_of_trials(self, num):

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

        
        #bar = Bar('Processing', max=self.trials_num)

        print "\n\nTrials num is: " + str(self.trials_num)
        for i in range(self.trials_num):

            # seed random for experiments
            #numpy.random.seed(i+20)

            # get a sample as the current state, terminal state exclusive
            if self.auto_state:
                # 50% chance fixed to select unknown state
                unknown_state = numpy.random.choice([True, False])
                # 100% chance to select unknown
                #unknown_state = True

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

            '''!!! important note: State self.s not used as goal anymore, since we need new items to be possible as well,
            instead self.s_plus is used to compare''' 

            #self.s_plus = self.states_plus.index(self.states[self.s])
            print self.states_plus[self.s_plus]
            print self.states
            if str(self.states_plus[self.s_plus]) in self.states:
                is_new = False
            else:
                is_new = True

            # run this episode and save the reward
            reward, cost, added = self.run(num)

            reward_list.append(reward)
            cost_list.append(cost)

            # use string based checking of success for now
            if (str(self.states_plus[self.s_plus]) in self.actions[self.a]) and (is_new == added):
                success_list.append(1.0)
            else:
                success_list.append(0.0)                

            overall_reward = reward
            overall_reward_list.append(overall_reward)

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

            #bar.next()

        #bar.finish()

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

        print('True positives:' + str(true_positives))
        print('False positives:' + str(false_positives))
        print('True negatives:' + str(true_negatives))
        print('False negatives:' + str(false_negatives))

        if true_positives + false_positives == 0:
            precision = 0
        else:
            precision = true_positives/(true_positives + false_positives)
        
        if true_positives + false_negatives == 0:
            recall = 0
        else:        
            recall = true_positives/(true_positives + false_negatives)
	try:
            f1_score = 2 * precision * recall/(precision + recall)
	except:
	    f1_score = 0
        
        print('Precision:' + str(precision))
        print('Recall:' + str(recall))
        print('F1:' + str(f1_score))

        return (numpy.mean(cost_arr), numpy.mean(success_arr), \
            numpy.mean(overall_reward_arr), precision, recall, f1_score)


    def run(self,  N=5):
        cost = 0.0
        self.init_belief()
        self.init_belief_plus()

        reward = 0.0

        cycletime = 0

        current_entropy = float("inf")
        old_entropy = float("inf")
        inc_count = 0
        added = False

        while True:
            cycletime += 1

            # print self.b

            if self.print_flag:
                print('\tstate:\t' + self.states_plus[self.s_plus] + ' ' + str(self.s_plus))
                print('\tcost so far:\t' + str(cost))

            # select action
            # entropy
            old_entropy = current_entropy
            current_entropy = stats.entropy(self.b)
            current_entropy_plus = stats.entropy(self.b_plus)
            if self.print_flag:
                print "DEBUG: Entropy = ",current_entropy
                print "DEBUG: Entropy_plus = ",current_entropy_plus
            # check if entropy increased
            if (old_entropy < current_entropy):
                inc_count += 1

            if(self.entropy_check(current_entropy)):
                self.get_full_request(cycletime)
                if self.print_flag:
                    print('\n\tbelief:\t\t' + str(self.b))
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
                    if self.print_flag is True:
                        print('\treward: ' + str(self.reward_mat_plus[self.a_plus, self.s_plus]))
                    reward = cost + self.reward_mat_plus[self.a_plus, self.s_plus]
                    done = True


                if done == True:
                    break

                # check entropy increases arbitrary no of times for now
                if (added == False):
                    print cycletime
                    if(cycletime > N): # N equal 1
                        print "--- new item/person ---"
                        added = True
                        self.add_new()

                if self.auto_observations:
                    raw_str = self.auto_observe()
                else:
                    raw_str = raw_input("Input observation: ")

                self.observe(raw_str)
                self.update(cycletime)
                self.update_plus(cycletime)

                if self.print_flag:
                    print('\tobserve:\t'+self.observations[self.o]+' '+str(self.o))
                    print('\n\tbelief:\t\t' + str(self.b))
                    print('\n\tbelief_plus:\t' + str(self.b_plus))


            # print('current cost: ' + str(self.reward_mat[self.a, self.s]))
            # print('overall cost: ' + str(overall_reward))

            if 'go' not in self.actions_plus[self.a_plus]:
                cost += self.reward_mat_plus[self.a_plus, self.s_plus]

            if cycletime == 50:
                print "BASELINE: REACHED CYCLE TIME 50"
            #    sys.exit(1)
                reward = cost + self.reward_mat_plus[self.a_plus, self.s_plus]
                break

        return reward, abs(cost), added
