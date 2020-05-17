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

    def run(self,counteR):
        cost = 0.0
        self.init_belief()
        self.init_belief_plus()

        reward = 0.0

        cycletime = 0
        added_point = 0

        current_entropy = float("inf")
        old_entropy = float("inf")
        old_sign = 1
        inc_count = 0
        added = False

        while True:
            print("\n\n\n")
            cycletime += 1
            politeness = "I know I asked it already but could you "
            # print self.b

            if self.print_flag:
                print('\tstate:\t' + self.states_plus[self.s_plus] + ' ' + str(self.s_plus))
                print('\tcost so far:\t' + str(cost))

            # select action
            # entropy
            old_sign = self.sign(current_entropy - old_entropy)
            old_entropy = current_entropy
            current_entropy = stats.entropy(self.b)
            current_entropy_plus = stats.entropy(self.b_plus)
            
            if self.print_flag:
                print ("DEBUG: Entropy = ",current_entropy)
                print ("DEBUG: Entropy_plus = ",current_entropy_plus)
            # check if entropy increased

            self.entropy_list.append(current_entropy)

            # if(old_entropy < current_entropy):
            if (self.sign(current_entropy - old_entropy) != old_sign):
                if old_entropy != float("inf"):
                    inc_count += 1
                    print ("DEBUG: entropy fluctuated")

            if(self.entropy_check(current_entropy)):
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

                    print ('num_recipients', self.num_recipient)
                    print ('num_patients', self.num_patient)
        # Count questio occurence, if it is > 1, add politeness word to head
                question = self.action_to_text(self.actions[self.a])
                self.question_list.append(self.actions[self.a])
                tmp = question
                if question:
                    if(self.question_list.count(question) >= 1):
                        tmp = politeness + question
                        print('     QUESTION: ' + tmp)
                        self.question_list.append(question)
                elif ('go' in self.actions[self.a]):
                    print('EXECUTE: ' + self.actions[self.a])
                    if self.print_flag:
                        print('\treward: ' + str(self.reward_mat_plus[self.a_plus, self.s_plus]))
                    
                    
                    reward = cost + self.reward_mat_plus[self.a_plus, self.s_plus]     
                    done = True


                if done == True:
                    break

                # check entropy increases arbitrary no of times for now
                if (added == False):
                    if(self.belief_check()):
                        print ("--- new item/person ---")
                        added = True
                        self.added_point = (cycletime-1, current_entropy, len(self.states))
                        f = open('entropy_plot/addedOur.txt', 'a')
                        f.write("%s\n" % str(self.added_point))
                        f.close()
                        added_point = cycletime-1
                        self.add_new()

                if self.auto_observations:
                    raw_str = self.auto_observe(counteR)
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
                print ("REACHED CYCLE TIME 50")
                reward = cost + self.reward_mat_plus[self.a_plus, self.s_plus]
        #        sys.exit(1)
                break

        
        return cycletime-1,added_point, reward, abs(cost), added
