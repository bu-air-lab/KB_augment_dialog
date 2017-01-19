#!/usr/bin/env python

import numpy as np
import time
import subprocess
import os.path

class State(object):

    def __init__(self, num_identity, num_college, num_year):
        
        self.num_identity = num_identity
        self.num_college = num_college
        self.num_year = num_year

class StateRegular(State):

    def __init__(self, identity, college, year, num_identity, num_college, num_year, index):

        State.__init__(self, num_identity, num_college, num_year)

        self.identity = identity
        self.college = college
        self.year = year
        self.index = index

    def getName(self):
        return 'i' + str(self.identity) + '_c' + str(self.college) + '_y' + \
            str(self.year)

    def getIndex(self):
        return self.index

    def getidentityIndex(self):
        return self.identity

    def getcollegeIndex(self):
        return self.college

    def getyearIndex(self):
        return self.year

    def isTerminal(self):
        return False


class StateTerminal(State):

    def __init__(self, num_identity, num_college, num_year, index):

        State.__init__(self, num_identity, num_college, num_year)
        self.index = index

    def getName(self):
        return 'terminal'

    def getIndex(self):
        return self.index

    def isTerminal(self):
        return True

class Action(object):

    # qd_type: type of this action: ask or deliver? 
    def __init__(self, qd_type):
        
        assert qd_type in ['ask', 'guide']
        self.qd_type = qd_type


class ActionAsk(Action):

    def __init__(self, q_type):

        assert q_type in ['wh', 'polar']
        Action.__init__(self, 'ask')
        self.q_type = q_type


class ActionAskWh(ActionAsk):

    def __init__(self, var):

        assert var in ['identity', 'college', 'year']
        ActionAsk.__init__(self, 'wh')
        self.var = var

    def getName(self):
        if self.var == 'identity':
            return 'ask_i'
        elif self.var == 'college':
            return 'ask_c'
        elif self.var == 'year':
            return 'ask_y'

    def getIndex(self):
        
        if self.var == 'identity':
            return 0
        elif self.var == 'college':
            return 1
        elif self.var == 'year':
            return 2


class ActionAskPolar(ActionAsk):

    def __init__(self, var, num_identity, num_college, num_year):

        ActionAsk.__init__(self, 'polar')
        self.num_identity = num_identity
        self.num_college = num_college
        self.num_year = num_year
        assert var in ['identity', 'college', 'year']
        self.var = var

class ActionAskPolaridentity(ActionAskPolar):

    def __init__(self, identity, num_identity, num_college, num_year):

        assert identity < num_identity
        ActionAskPolar.__init__(self, 'identity', num_identity, num_college, num_year)
        self.identity = identity

    def getidentityIndex(self):
        return self.identity

    def getIndex(self):
        return 3 + self.identity

    def getName(self):
        return 'confirm_i' + str(self.identity)

class ActionAskPolarcollege(ActionAskPolar):

    def __init__(self, college, num_identity, num_college, num_year):
        
        assert college < num_college
        ActionAskPolar.__init__(self, 'college', num_identity, num_college, num_year)
        self.college = college

    def getcollegeIndex(self):
        return self.college

    def getIndex(self):
        return 3 + self.num_identity + self.college

    def getName(self):
        return 'confirm_c' + str(self.college)

class ActionAskPolaryear(ActionAskPolar):

    def __init__(self, year, num_identity, num_college, num_year):
        
        assert year < num_year
        ActionAskPolar.__init__(self, 'year', num_identity, num_college, num_year)
        self.year = year

    def getyearIndex(self):
        return self.year

    def getIndex(self):
        return 3 + self.num_identity + self.num_college + self.year

    def getName(self):
        return 'confirm_y' + str(self.year)

class ActionDeliver(Action):

    def __init__(self, identity, college, year, num_identity, num_college, num_year, index):

        assert identity < num_identity
        assert college < num_college
        assert year < num_year
        Action.__init__(self, 'guide')
        self.identity = identity
        self.college = college
        self.year = year
        self.num_identity = num_identity
        self.num_college = num_college
        self.num_year = num_year
        self.index = 3 + self.num_identity + self.num_college + self.num_year + index

    def getidentityIndex(self):
        return self.identity

    def getcollegeIndex(self):
        return self.college

    def getyearIndex(self):
        return self.year

    def getIndex(self):
        return self.index

    def getName(self):
        return 'go_i' + str(self.identity) + '_c' + str(self.college) + '_y' + \
            str(self.year)

class Observation(object):

    def __init__(self, qd_type):
    
        assert qd_type in ['wh', 'polar', 'none']
        self.qd_type = qd_type

class ObservationWh(Observation):

    def __init__(self, var):

        assert var in ['identity', 'college', 'year']
        Observation.__init__(self, 'wh')
        self.var = var

class ObservationWhidentity(ObservationWh):

    def __init__(self, identity, num_identity, num_college, num_year):

        assert identity < num_identity
        ObservationWh.__init__(self, 'identity')
        self.identity = identity
        self.num_identity = num_identity
        self.num_college = num_college
        self.num_year = num_year

    def getidentityIndex(self):
        return self.identity

    def getIndex(self):
        return self.identity

    def getName(self):
        return 'i' + str(self.identity)


class ObservationWhcollege(ObservationWh):

    def __init__(self, college, num_identity, num_college, num_year):

        assert college < num_college
        ObservationWh.__init__(self, 'college')
        self.college = college
        self.num_identity = num_identity
        self.num_college = num_college
        self.num_year = num_year

    def getcollegeIndex(self):
        return self.college

    def getIndex(self):
        return self.num_identity + self.college

    def getName(self):
        return 'c' + str(self.college)

class ObservationWhyear(ObservationWh):

    def __init__(self, year, num_identity, num_college, num_year):

        assert year < num_year
        ObservationWh.__init__(self, 'year')
        self.year = year
        self.num_identity = num_identity
        self.num_college = num_college
        self.num_year = num_year

    def getyearIndex(self):
        return self.year

    def getIndex(self):
        return self.num_identity + self.num_college + self.year

    def getName(self):
        return 'y' + str(self.year)

class ObservationPolar(Observation):

    def __init__(self, polar, num_identity, num_college, num_year):

        assert polar in ['yes', 'no']
        Observation.__init__(self, 'polar')
        self.polar = polar
        self.num_identity = num_identity
        self.num_college = num_college
        self.num_year = num_year

    def getName(self):
        if self.polar == 'yes':
            return 'yes'
        elif self.polar == 'no':
            return 'no'

    def getIndex(self):
        if self.polar == 'yes':
            return self.num_identity + self.num_college + self.num_year
        elif self.polar == 'no':
            return self.num_identity + self.num_college + self.num_year + 1


class ObservationNone(Observation):

    def __init__(self, num_identity, num_college, num_year):

        Observation.__init__(self, 'none')
        self.num_identity = num_identity
        self.num_college = num_college
        self.num_year = num_year

    def getName(self):
        return 'none'

    def getIndex(self):
        return self.num_identity + self.num_college + self.num_year + 2

class PomdpGenerator(object):

    def __init__(self, num_identity, num_college, num_year, r_max, r_min, strategy, \
        weight_i, weight_c, weight_y, wh_cost, yesno_cost):
        

        self.num_identity = num_identity
        self.num_college = num_college
        self.num_year = num_year

        self.r_max = r_max
        self.r_min = r_min

        self.weight_i = weight_i
        self.weight_c = weight_c
        self.weight_y = weight_y

        self.weight_i_bin = weight_i.astype(int)
        self.weight_c_bin = weight_c.astype(int)
        self.weight_y_bin = weight_y.astype(int)

        # the larger, the more unreliable for the wh-questions. 
        self.magic_number = 0.3
        self.polar_tp_rate = 0.7
        self.polar_tn_rate = 0.7

        self.tablelist = [[0, 0, 0], [0, 1, 1], [1, 1, 2], [2, 2, 2]]

        self.state_set = []
        self.action_set = []
        self.observation_set = []

        self.trans_mat = None
        self.obs_mat = None
        self.reward_mat = None

        # compute the sets of states, actions, observations
        self.state_set = self.computeStateSet(self.num_identity, self.num_college,
            self.num_year)
        self.action_set = self.computeActionSet(self.num_identity, self.num_college,
            self.num_year)
        self.observation_set = self.computeObservationSet(self.num_identity, 
            self.num_college, self.num_year)

        # compute the functions of transition, observation
        self.trans_mat = self.computeTransFunction(self.num_identity,
            self.num_college, self.num_year)

        self.obs_mat = self.computeObsFunction(self.num_identity, self.num_college,
            self.num_year, self.magic_number, self.polar_tp_rate)

        # compute two versions of reward function
        reward_mat_float = self.computeRewardFunction(self.num_identity,
            self.num_college, self.num_year, self.r_max, self.r_min, \
            self.weight_i, self.weight_c, self.weight_y, wh_cost, yesno_cost)

        reward_mat_bin = self.computeRewardFunction(self.num_identity,
            self.num_college, self.num_year, self.r_max, self.r_min, \
            self.weight_i_bin, self.weight_c_bin, self.weight_y_bin, wh_cost, yesno_cost)

        # the idea is to keep the reward of fully correct deliveries and
        # question-asking actions, while changing the other negative values
        reward_mat_float_positive = reward_mat_float.clip(min=0)
        reward_mat_float_negative = reward_mat_float.clip(max=0)
        reward_mat_float_negative_deliveries = reward_mat_float_negative -\
            reward_mat_float_negative.clip(min=min(wh_cost, yesno_cost))
        reward_mat_float_negative_questions = reward_mat_float_negative - \
            reward_mat_float_negative_deliveries

        reward_mat_bin_positive = reward_mat_bin.clip(min=0)
        reward_mat_bin_negative = reward_mat_bin.clip(max=0)
        reward_mat_bin_negative_deliveries = reward_mat_bin_negative -\
            reward_mat_bin_negative.clip(min=min(wh_cost, yesno_cost))
        reward_mat_bin_negative_questions = reward_mat_bin_negative -\
            reward_mat_bin_negative_deliveries

        sum_float_negative_deliveries = np.sum(reward_mat_float_negative_deliveries)
        sum_bin_negative_deliveries = np.sum(reward_mat_bin_negative_deliveries)
        reweight_factor = sum_bin_negative_deliveries / sum_float_negative_deliveries

        reward_mat_float = reward_mat_float_positive + \
            reward_mat_float_negative_questions + \
            reward_mat_float_negative_deliveries * reweight_factor

        # writing to files
        self.filename = strategy + '_old.pomdp'
        self.reward_mat = reward_mat_bin
        self.writeToFile()

        self.filename = strategy +'_new.pomdp'
        self.reward_mat = reward_mat_float
        #self.reward_mat = reward_mat_float_negative_deliveries
        self.writeToFile()

        print 'training for 60 seconds'

        pomdpsol_lu = '/home/ludc/workspace/context_aware_icorpp/appl-0.96/src/pomdpsol'
        pomdpsol_zhang = '/home/szhang/software/appl/appl-0.95/src/pomdpsol'

        if os.path.isfile(pomdpsol_lu):
            pomdpsol = pomdpsol_lu
        elif os.path.isfile(pomdpsol_zhang):
            pomdpsol = pomdpsol_zhang
        else:
            print "pomdpsol not installed..."
            exit(1)

        subprocess.check_output(pomdpsol + ' --timeout 60 --output ' \
                                    + strategy + '_new.policy ' + strategy + '_new.pomdp', shell = True)
        print 'finish training'

    def computeTransFunction(self, num_identity, num_college, num_year):

        num_state = len(self.state_set)

        # as this problem is specific, this can be manually done
        trans_mat = np.ones((len(self.action_set), len(self.state_set), 
            len(self.state_set)))

        for action in self.action_set:

            if action.qd_type == 'ask':
                trans_mat[action.getIndex()] = np.eye(num_state, dtype=float)
            elif action.qd_type == 'guide':
                trans_mat[action.getIndex()] = np.zeros((num_state, num_state))
                trans_mat[action.getIndex()][:, num_state-1] = 1.0
                
        return trans_mat

    # HERE WE INTRODUCE A NOVEL OBSERVATION MODEL
    # true-positive rate = 1/(n^0.1), where n is the variable's range
    def computeObsFunction(self, num_identity, num_college, num_year, magic_number,
        polar_tp_rate):
        
        num_action = len(self.action_set)
        num_state = len(self.state_set)
        num_obs = len(self.observation_set)

        obs_mat = np.zeros((num_action, num_state, num_obs))

        for action in self.action_set:

            # no observation given 'terminal' state, no matter of the action
            for state in self.state_set:
                if state.isTerminal() == True:
                    obs_mat[action.getIndex()] = np.zeros((num_state, num_obs))
                    obs_mat[action.getIndex()][state.getIndex(), num_obs-1]=1.0

            if action.qd_type == 'guide':
                obs_mat[action.getIndex()] = np.zeros((num_state, num_obs))
                obs_mat[action.getIndex()][:, num_obs-1] = 1.0
                
            elif action.qd_type == 'ask':
                if action.q_type == 'wh':
                    if action.var == 'identity':
                        tp_rate = 1.0 / pow(num_identity, magic_number)
                        for state in self.state_set:
                            if state.isTerminal() == True:
                                continue
                            for observation in self.observation_set:
                                if observation.qd_type == 'wh':
                                    if observation.var == 'identity':
                                        if observation.getidentityIndex() == \
                                            state.getidentityIndex():
                                            
                                            obs_mat[action.getIndex()]\
                                                [state.getIndex()]\
                                                [observation.getIndex()] = \
                                                tp_rate
                                        else:
                                            obs_mat[action.getIndex()]\
                                                [state.getIndex()]\
                                                [observation.getIndex()] = \
                                                (1.0-tp_rate) / (num_identity-1)
                                    
                    elif action.var == 'college':
                        tp_rate = 1.0 / pow(num_college, magic_number)
                        for state in self.state_set:
                            if state.isTerminal() == True:
                                continue
                            for observation in self.observation_set:
                                if observation.qd_type == 'wh':
                                    if observation.var == 'college':
                                        if observation.getcollegeIndex() == \
                                            state.getcollegeIndex():
                                            
                                            obs_mat[action.getIndex()]\
                                                [state.getIndex()]\
                                                [observation.getIndex()]\
                                                = tp_rate
                                        else:
                                            obs_mat[action.getIndex()]\
                                                [state.getIndex()]\
                                                [observation.getIndex()]\
                                                = (1.0-tp_rate) / (num_college-1)

                    elif action.var == 'year':
                        tp_rate = 1.0 / pow(num_year, magic_number)
                        for state in self.state_set:
                            if state.isTerminal() == True:
                                continue
                            for observation in self.observation_set:
                                if observation.qd_type == 'wh':
                                    if observation.var == 'year':
                                        if observation.getyearIndex() ==\
                                            state.getyearIndex():
                                            
                                            obs_mat[action.getIndex()]\
                                                [state.getIndex()]\
                                                [observation.getIndex()]\
                                                = tp_rate
                                        else:
                                            obs_mat[action.getIndex()]\
                                                [state.getIndex()]\
                                                [observation.getIndex()]\
                                                = (1.0-tp_rate) / (num_year-1)
                    
                elif action.q_type == 'polar':
                    if action.var == 'identity':
                        for state in self.state_set:
                            if state.isTerminal() == True:
                                continue
                            for observation in self.observation_set:
                                if observation.getName() == 'yes':
                                    if state.getidentityIndex() ==\
                                        action.getidentityIndex():
                                        obs_mat[action.getIndex()]\
                                            [state.getIndex()]\
                                            [observation.getIndex()]\
                                            = self.polar_tp_rate
                                    else:
                                        obs_mat[action.getIndex()]\
                                            [state.getIndex()]\
                                            [observation.getIndex()] = \
                                            1.0 - self.polar_tp_rate
                                elif observation.getName() == 'no':
                                    if state.getidentityIndex() ==\
                                        action.getidentityIndex():
                                        obs_mat[action.getIndex()]\
                                            [state.getIndex()]\
                                            [observation.getIndex()]= \
                                            1.0 - self.polar_tn_rate
                                    else:
                                        obs_mat[action.getIndex()]\
                                            [state.getIndex()]\
                                            [observation.getIndex()] \
                                            = self.polar_tn_rate
                    elif action.var == 'college':
                        for state in self.state_set:
                            if state.isTerminal() == True:
                                continue
                            for observation in self.observation_set:
                                if observation.getName() == 'yes':
                                    if state.getcollegeIndex() == \
                                        action.getcollegeIndex():
                                        obs_mat[action.getIndex()]\
                                            [state.getIndex()]\
                                            [observation.getIndex()]\
                                            = self.polar_tp_rate
                                    else:
                                        obs_mat[action.getIndex()]\
                                            [state.getIndex()]\
                                            [observation.getIndex()]=\
                                            1.0 - self.polar_tp_rate
                                elif observation.getName() == 'no':
                                    if state.getcollegeIndex() == \
                                        action.getcollegeIndex():
                                        obs_mat[action.getIndex()]\
                                            [state.getIndex()]\
                                            [observation.getIndex()]= \
                                            1.0 - self.polar_tn_rate
                                    else:
                                        obs_mat[action.getIndex()]\
                                            [state.getIndex()]\
                                            [observation.getIndex()]=\
                                            self.polar_tn_rate
                    elif action.var == 'year':
                        for state in self.state_set:
                            if state.isTerminal() == True:
                                continue
                            for observation in self.observation_set:
                                if observation.getName() == 'yes':
                                    if state.getyearIndex() ==\
                                        action.getyearIndex():
                                        obs_mat[action.getIndex()]\
                                            [state.getIndex()]\
                                            [observation.getIndex()]\
                                            = self.polar_tp_rate
                                    else:
                                        obs_mat[action.getIndex()]\
                                            [state.getIndex()]\
                                            [observation.getIndex()]=\
                                            1.0 - self.polar_tp_rate
                                elif observation.getName() == 'no':
                                    if state.getyearIndex() ==\
                                        action.getyearIndex():
                                        obs_mat[action.getIndex()]\
                                            [state.getIndex()]\
                                            [observation.getIndex()]= \
                                            1.0 - self.polar_tn_rate
                                    else:
                                        obs_mat[action.getIndex()]\
                                            [state.getIndex()]\
                                            [observation.getIndex()]\
                                            = self.polar_tn_rate

        return obs_mat

    def computeRewardFunction(self, num_identity, num_college, num_year,
        r_max, r_min, weight_i, weight_c, weight_y, wh_cost, yesno_cost):

        reward_mat = np.zeros((len(self.action_set), len(self.state_set), ))

        for action in self.action_set:

            if action.qd_type == 'ask':

                if action.q_type == 'wh':
                    reward_mat[action.getIndex()] = wh_cost
                elif action.q_type == 'polar':
                    reward_mat[action.getIndex()] = yesno_cost

            elif action.qd_type == 'guide':
 
                for state in self.state_set:
                    
                    if state.isTerminal() == False:

                        reward_mat[action.getIndex()][state.getIndex()] = \
                            self.guideReward(r_max, r_min, 
                            weight_i, weight_c, weight_y, action, state)

            num_state = len(self.tablelist) + 1
            reward_mat[action.getIndex()][num_state - 1] = 0.0

        return reward_mat

    def guideReward(self, r_max, r_min, weight_i, weight_c, weight_y,
        action, state):

        if weight_i[action.getidentityIndex()][state.getidentityIndex()] == 1.0 and  \
                weight_c[action.getcollegeIndex()][state.getcollegeIndex()] == 1.0 and  \
                weight_y[action.getyearIndex()][state.getyearIndex()] == 1.0:
            return r_max
        else:
            ret = r_min * (1.0 - \
                weight_i[action.getidentityIndex()][state.getidentityIndex()] * \
                weight_c[action.getcollegeIndex()][state.getcollegeIndex()] * \
                weight_y[action.getyearIndex()][state.getyearIndex()])
            return ret

    def computeStateSet(self, num_identity, num_college, num_year):

        ret = []
        for i in range(num_identity):
            for c in range(num_college):
                for y in range(num_year):
                    for table in self.tablelist:
                        if i == table[0] and c == table[1] and y == table[2]:
                            ret.append(StateRegular(i, c, y, num_identity, num_college, num_year,self.tablelist.index(table)))

        ret.append(StateTerminal(num_identity, num_college, num_year, len(self.tablelist)))

        return ret

    def computeActionSet(self, num_identity, num_college, num_year):

        ret = []
        ret.append(ActionAskWh('identity'))
        ret.append(ActionAskWh('college'))
        ret.append(ActionAskWh('year'))
        
        for i in range(num_identity):
            ret.append(ActionAskPolaridentity(i, num_identity, num_college, num_year))

        for c in range(num_college):
            ret.append(ActionAskPolarcollege(c, num_identity, num_college, num_year))

        for y in range(num_year):
            ret.append(ActionAskPolaryear(y, num_identity, num_college, num_year))

        for i in range(num_identity):
            for c in range(num_college):
                for y in range(num_year):
                    for table in self.tablelist:
                        if i == table[0] and c == table[1] and y == table[2]:
                            ret.append(ActionDeliver(i, c, y, num_identity, num_college,
                                num_year, self.tablelist.index(table)))

        return ret

    def computeObservationSet(self, num_identity, num_college, num_year):

        ret = []
        for i in range(num_identity):
            ret.append(ObservationWhidentity(i, num_identity, num_college, num_year))

        for p in range(num_college):
            ret.append(ObservationWhcollege(p, num_identity, num_college, num_year))

        for r in range(num_year):
            ret.append(ObservationWhyear(r, num_identity, num_college, num_year))

        ret.append(ObservationPolar('yes', num_identity, num_college, num_year))
        ret.append(ObservationPolar('no', num_identity, num_college, num_year))
        ret.append(ObservationNone(num_identity, num_college, num_year))

        return ret

    def writeToFile(self):

        f = open(self.filename, 'w')

        # first few lines
        s = ''
        s += 'discount : 0.999999\n\nvalues: reward\n\nstates: '

        # section of states
        for state in self.state_set:
            s += state.getName() + ' '

        # section of actions
        s += '\n\nactions: '

        for action in self.action_set:
            s += action.getName() + ' '

        # section of observations
        s += '\n\nobservations: '

        for observation in self.observation_set:
            s += observation.getName() + ' ' 

        # section of transition matrix
        for action in self.action_set:
            s += '\n\nT: ' + action.getName() + '\n'
            for state_from in self.state_set:
                for state_to in self.state_set:

                    prob = self.trans_mat[action.getIndex(), \
                        state_from.getIndex(), state_to.getIndex()]
                    s += str(prob) + ' '

                s += '\n'

        # section of observation matrix
        for action in self.action_set:
            s += '\nO: ' + action.getName() + '\n'
            for state in self.state_set:
                for observation in self.observation_set:
                    s += str(self.obs_mat[action.getIndex()]\
                        [state.getIndex()]\
                        [observation.getIndex()]) + ' '
                s += '\n'

        s += '\n'

        # sectoin of reward matrix
        for action in self.action_set:
            for state in self.state_set:
                s += 'R: ' + action.getName() + '\t\t: ' + state.getName() + \
                    '\t: *\t\t: * ' + \
                    str(self.reward_mat[action.getIndex()][state.getIndex()]) + '\n'

        f.write(s)
        f.close()

def main():

    r_max = 20.0
    r_min = -20.0

    wh_cost = -1.5
    yesno_cost = -1.0

    num_identity = 3
    num_college = 3
    num_year = 3

    # row corresponds to action, column to underlying state
    # all
    weight_i = np.array([[1.00, 0.00, 0.00], 
                         [0.00, 1.00, 0.00], 
                         [0.00, 0.00, 1.00]])

    weight_c = np.array([[1.0, 0.0, 0.0], 
                         [0.0, 1.0, 0.0], 
                         [0.0, 0.0, 1.0]])

    weight_y = np.array([[1.0, 0.0, 0.0], 
                         [0.0, 1.0, 0.0], 
                         [0.0, 0.0, 1.0]])
    
    # strategy = str(num_identity) + str(num_college) + str(num_year) 
    # strategy = str(num_identity) + str(num_college) + str(num_year) + '_' + str(entry)
    # strategy = str(num_identity) + str(num_college) + str(num_year) + '_' + str(entry1) + str(entry2)
    strategy = str(num_identity) + str(num_college) + str(num_year)

    pg = PomdpGenerator(num_identity, num_college, num_year, r_max, r_min, strategy, \
        weight_i, weight_c, weight_y, wh_cost, yesno_cost)

if __name__ == '__main__':

    main()


