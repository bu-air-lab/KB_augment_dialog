#!/usr/bin/env python

import sys
import time
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

class Simulator(object):

    def __init__(self, 
        auto_observations=True,
        auto_state = False, 
        uniform_init_belief =True,
        print_flag=True,
        use_plog = True,
        policy_file='policy/default.policy', 
        pomdp_file='models/default.pomdp', 
        trials_num=1000,
        num_item=1, 
        num_person=1,
        num_room=1):

        # print(pomdp_file)
        # print(policy_file)

        self.auto_observations = auto_observations
        self.auto_state = auto_state
        self.uniform_init_belief = uniform_init_belief
        self.print_flag = print_flag
        self.use_plog = use_plog
        self.trials_num = trials_num

        self.num_item = num_item
        self.num_person = num_person
        self.num_room = num_room
        self.tablelist = conf.tablelist

        # to read the pomdp model
        model = pomdp_parser.Pomdp(filename=pomdp_file, parsing_print_flag=False)
        self.states = model.states
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
        self.a = None
        self.o = None

        self.md = 'happy'
        self.fl = True
        self.trigger= 1 ###triggle of event in which dialog turn
        # self.dialog_turn = 0

        self.plog = gen_dis_plog.DistrGen()

        # to make the screen print simple 
        numpy.set_printoptions(precision=2)
        if self.use_plog:
            print '------------ context-aware-icorpp -----------------'
        else:
            print '------------ previous icorpp -----------------'

    #######################################################################
    def init_belief(self):

        if self.uniform_init_belief:
            self.b = numpy.ones(len(self.states)) / float(len(self.states))

            if self.use_plog and (self.md == 'sad' or self.fl == False):
                if self.b[len(self.tablelist)] == 1:
                    return
                # print '\n',self.b
                belief = self.plog.cal_belief(mood = self.md, foll = self.fl, pdpDist = self.b, curr_table = self.ct, prev_table = self.pt).split(',')
                for i in range(len(belief)):
                    belief[i] = float(belief[i].strip())
                self.b = numpy.array(belief)
                self.b = self.b/ sum(self.b)
                # print '\n',self.s, self.ct, self.b
        else:
            # here initial belief is sampled from a Dirichlet distribution
            self.b = numpy.random.dirichlet( numpy.ones(len(self.states)) )

        self.b = self.b.T

    #######################################################################
    def observe(self):
        self.o = None
        if self.auto_observations:
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
            ind = input("Please input the name of observation: ")
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
            belief = self.plog.cal_belief(mood = self.md, foll = self.fl, pdpDist = self.b, curr_table = self.ct, prev_table = self.pt).split(',')
            # belief = self.plog.cal_belief(mood = 'sad', pdpDist = self.b, curr_table = self.ct).split(',')
            for i in range(len(belief)):
                belief[i] = float(belief[i].strip())
            self.b = numpy.array(belief)
            self.b = self.b/ sum(self.b)

    #######################################################################
    def run(self):

        cost = 0.0
        self.init_belief()

        reward = 0.0
        overall_reward = 0.0

        cycletime = 0

        while True:
            cycletime += 1

            # print self.b

            if self.print_flag:
                print('\tstate:\t' + self.states[self.s] + ' ' + str(self.s))
                print('\tcost so far:\t' + str(cost))

            self.a = int(self.policy.select_action(self.b))
            
            if self.print_flag:
                print('\taction:\t' + self.actions[self.a] + ' ' + str(self.a))

            self.observe()
            if self.print_flag:
                print('\tobserve:\t'+self.observations[self.o]+' '+str(self.o))

            self.update(cycletime)
            if self.print_flag:
                print('\nbelief:\t' + str(self.b))


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

            guide_index = int(self.a - (3 + self.num_item + self.num_person \
                + self.num_room))

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

    s = Simulator(uniform_init_belief = True, 
        auto_state = True, 
        auto_observations = True, 
        print_flag = False, 
        use_plog = True,
        policy_file = '333_new.policy', 
        pomdp_file =  '333_new.pomdp',
        trials_num = 1000,
        num_item = 3, 
        num_person = 3, 
        num_room = 3)
 
    if not s.uniform_init_belief:   
        print('note that initial belief is not uniform\n')

    s.run_numbers_of_trials()

if __name__ == '__main__':
    main()

