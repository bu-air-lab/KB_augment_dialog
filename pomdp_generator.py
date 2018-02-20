#!/usr/bin/env python

import numpy as np
import time
import conf
import subprocess
import os.path

class State(object):

    def __init__(self, num_task, num_patient, num_recipient):
        
        self.num_task = num_task
        self.num_patient = num_patient
        self.num_recipient = num_recipient

class StateRegular(State):

    def __init__(self, task, patient, recipient, num_task, num_patient, num_recipient, index):

        State.__init__(self, num_task, num_patient, num_recipient)

        self.task = task
        self.patient = patient
        self.recipient = recipient
        self.index = index

    def getName(self):
        return 't' + str(self.task) + '_p' + str(self.patient) + '_r' + \
            str(self.recipient)

    def getIndex(self):
        return self.index

    def gettaskIndex(self):
        return self.task

    def getpatientIndex(self):
        return self.patient

    def getrecipientIndex(self):
        return self.recipient

    def isTerminal(self):
        return False


class StateTerminal(State):

    def __init__(self, num_task, num_patient, num_recipient, index):

        State.__init__(self, num_task, num_patient, num_recipient)
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
        
        assert qd_type in ['ask', 'executeTask']
        self.qd_type = qd_type


class ActionAsk(Action):

    def __init__(self, q_type):

        assert q_type in ['wh', 'polar']
        Action.__init__(self, 'ask')
        self.q_type = q_type


class ActionAskWh(ActionAsk):

    def __init__(self, var):

        assert var in ['task', 'patient', 'recipient']
        ActionAsk.__init__(self, 'wh')
        self.var = var

    def getName(self):
        if self.var == 'task':
            return 'ask_t'
        elif self.var == 'patient':
            return 'ask_p'
        elif self.var == 'recipient':
            return 'ask_r'

    def getIndex(self):
        
        if self.var == 'task':
            return 0
        elif self.var == 'patient':
            return 1
        elif self.var == 'recipient':
            return 2


class ActionAskPolar(ActionAsk):

    def __init__(self, var, num_task, num_patient, num_recipient):

        ActionAsk.__init__(self, 'polar')
        self.num_task = num_task
        self.num_patient = num_patient
        self.num_recipient = num_recipient
        assert var in ['task', 'patient', 'recipient']
        self.var = var

class ActionAskPolartask(ActionAskPolar):

    def __init__(self, task, num_task, num_patient, num_recipient):

        assert task < num_task
        ActionAskPolar.__init__(self, 'task', num_task, num_patient, num_recipient)
        self.task = task

    def gettaskIndex(self):
        return self.task

    def getIndex(self):
        return 3 + self.task

    def getName(self):
        return 'confirm_t' + str(self.task)

class ActionAskPolarpatient(ActionAskPolar):

    def __init__(self, patient, num_task, num_patient, num_recipient):
        
        assert patient < num_patient
        ActionAskPolar.__init__(self, 'patient', num_task, num_patient, num_recipient)
        self.patient = patient

    def getpatientIndex(self):
        return self.patient

    def getIndex(self):
        return 3 + self.num_task + self.patient

    def getName(self):
        return 'confirm_p' + str(self.patient)

class ActionAskPolarrecipient(ActionAskPolar):

    def __init__(self, recipient, num_task, num_patient, num_recipient):
        
        assert recipient < num_recipient
        ActionAskPolar.__init__(self, 'recipient', num_task, num_patient, num_recipient)
        self.recipient = recipient

    def getrecipientIndex(self):
        return self.recipient

    def getIndex(self):
        return 3 + self.num_task + self.num_patient + self.recipient

    def getName(self):
        return 'confirm_r' + str(self.recipient)

class ActionDeliver(Action):

    def __init__(self, task, patient, recipient, num_task, num_patient, num_recipient, index):

        assert task < num_task
        assert patient < num_patient
        assert recipient < num_recipient
        Action.__init__(self, 'executeTask')
        self.task = task
        self.patient = patient
        self.recipient = recipient
        self.num_task = num_task
        self.num_patient = num_patient
        self.num_recipient = num_recipient
        self.index = 3 + self.num_task + self.num_patient + self.num_recipient + index

    def gettaskIndex(self):
        return self.task

    def getpatientIndex(self):
        return self.patient

    def getrecipientIndex(self):
        return self.recipient

    def getIndex(self):
        return self.index

    def getName(self):
        return 'go_t' + str(self.task) + '_p' + str(self.patient) + '_r' + \
            str(self.recipient)

class Observation(object):

    def __init__(self, qd_type):
    
        assert qd_type in ['wh', 'polar', 'none']
        self.qd_type = qd_type

class ObservationWh(Observation):

    def __init__(self, var):

        assert var in ['task', 'patient', 'recipient']
        Observation.__init__(self, 'wh')
        self.var = var

class ObservationWhtask(ObservationWh):

    def __init__(self, task, num_task, num_patient, num_recipient):

        assert task < num_task
        ObservationWh.__init__(self, 'task')
        self.task = task
        self.num_task = num_task
        self.num_patient = num_patient
        self.num_recipient = num_recipient

    def gettaskIndex(self):
        return self.task

    def getIndex(self):
        return self.task

    def getName(self):
        return 't' + str(self.task)


class ObservationWhpatient(ObservationWh):

    def __init__(self, patient, num_task, num_patient, num_recipient):

        assert patient < num_patient
        ObservationWh.__init__(self, 'patient')
        self.patient = patient
        self.num_task = num_task
        self.num_patient = num_patient
        self.num_recipient = num_recipient

    def getpatientIndex(self):
        return self.patient

    def getIndex(self):
        return self.num_task + self.patient

    def getName(self):
        return 'p' + str(self.patient)

class ObservationWhrecipient(ObservationWh):

    def __init__(self, recipient, num_task, num_patient, num_recipient):

        assert recipient < num_recipient
        ObservationWh.__init__(self, 'recipient')
        self.recipient = recipient
        self.num_task = num_task
        self.num_patient = num_patient
        self.num_recipient = num_recipient

    def getrecipientIndex(self):
        return self.recipient

    def getIndex(self):
        return self.num_task + self.num_patient + self.recipient

    def getName(self):
        return 'r' + str(self.recipient)

class ObservationPolar(Observation):

    def __init__(self, polar, num_task, num_patient, num_recipient):

        assert polar in ['yes', 'no']
        Observation.__init__(self, 'polar')
        self.polar = polar
        self.num_task = num_task
        self.num_patient = num_patient
        self.num_recipient = num_recipient

    def getName(self):
        if self.polar == 'yes':
            return 'yes'
        elif self.polar == 'no':
            return 'no'

    def getIndex(self):
        if self.polar == 'yes':
            return self.num_task + self.num_patient + self.num_recipient
        elif self.polar == 'no':
            return self.num_task + self.num_patient + self.num_recipient + 1


class ObservationNone(Observation):

    def __init__(self, num_task, num_patient, num_recipient):

        Observation.__init__(self, 'none')
        self.num_task = num_task
        self.num_patient = num_patient
        self.num_recipient = num_recipient

    def getName(self):
        return 'none'

    def getIndex(self):
        return self.num_task + self.num_patient + self.num_recipient + 2

class PomdpGenerator(object):

    def __init__(self, num_task, num_patient, num_recipient, r_max, r_min, strategy, \
        wh_cost, yesno_cost,timeout=5):
        

        self.num_task = num_task
        self.num_patient = num_patient
        self.num_recipient = num_recipient

        self.r_max = r_max
        self.r_min = r_min

        self.weight_t=np.eye(num_task, dtype=float) 
        self.weight_p=np.eye(num_patient, dtype=float) 
        self.weight_r=np.eye(num_recipient, dtype=float)     

        self.weight_t_bin = self.weight_t.astype(int)
        self.weight_p_bin = self.weight_p.astype(int)
        self.weight_r_bin = self.weight_r.astype(int)

        # the larger, the more unreliable for the wh-questions. 
        self.magic_number = 0.3
        self.polar_tp_rate = 0.8
        self.polar_tn_rate = 0.8

        self.tablelist = conf.tablelist

        self.state_set = []
        self.action_set = []
        self.observation_set = []

        self.trans_mat = None
        self.obs_mat = None
        self.reward_mat = None

        # compute the sets of states, actions, observations
        self.state_set = self.computeStateSet(self.num_task, self.num_patient,
            self.num_recipient)
        self.action_set = self.computeActionSet(self.num_task, self.num_patient,
            self.num_recipient)
        self.observation_set = self.computeObservationSet(self.num_task, 
            self.num_patient, self.num_recipient)

        # compute the functions of transition, observation
        self.trans_mat = self.computeTransFunction(self.num_task,
            self.num_patient, self.num_recipient)

        self.obs_mat = self.computeObsFunction(self.num_task, self.num_patient,
            self.num_recipient, self.magic_number, self.polar_tp_rate)

        # compute two versions of reward function
        reward_mat_float = self.computeRewardFunction(self.num_task,
            self.num_patient, self.num_recipient, self.r_max, self.r_min, \
            self.weight_t, self.weight_p, self.weight_r, wh_cost, yesno_cost)

        reward_mat_bin = self.computeRewardFunction(self.num_task,
            self.num_patient, self.num_recipient, self.r_max, self.r_min, \
            self.weight_t_bin, self.weight_p_bin, self.weight_r_bin, wh_cost, yesno_cost)

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

        print 'Training for '+ str(timeout)+' seconds'

        pomdpsol_lu = '/home/ludc/workspace/context_aware_icorpp/appl-0.96/src/pomdpsol'
        pomdpsol_zhang = '/home/szhang/software/appl/appl-0.95/src/pomdpsol'
        pomdpsol_sujay = '/home/sujay/context_aware_icorpp/sarsop/src/pomdpsol'
        pomdpsol_saeid = '/home/saeid/software/sarsop/src/pomdpsol'

        if os.path.isfile(pomdpsol_lu):
            pomdpsol = pomdpsol_lu
        elif os.path.isfile(pomdpsol_zhang):
            pomdpsol = pomdpsol_zhang
        elif os.path.isfile(pomdpsol_sujay):
        	pomdpsol = pomdpsol_sujay
        elif os.path.isfile(pomdpsol_saeid):
                pomdpsol = pomdpsol_saeid
        else:
            print "pomdpsol not installed..."
            exit(1)

        subprocess.check_output(pomdpsol + ' --timeout '+str(timeout)+' --output ' \
                                    + strategy + '_new.policy ' + strategy + '_new.pomdp', shell = True)
        print 'Finished training'

    def computeTransFunction(self, num_task, num_patient, num_recipient):

        num_state = len(self.state_set)

        # as this problem is specific, this can be manually done
        trans_mat = np.ones((len(self.action_set), len(self.state_set), 
            len(self.state_set)))

        for action in self.action_set:

            if action.qd_type == 'ask':
                trans_mat[action.getIndex()] = np.eye(num_state, dtype=float)
            elif action.qd_type == 'executeTask':
                trans_mat[action.getIndex()] = np.zeros((num_state, num_state))
                trans_mat[action.getIndex()][:, num_state-1] = 1.0
                
        return trans_mat

    # HERE WE INTRODUCE A NOVEL OBSERVATION MODEL
    # true-positive rate = 1/(n^0.1), where n is the variable's range
    def computeObsFunction(self, num_task, num_patient, num_recipient, magic_number,
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

            if action.qd_type == 'executeTask':
                obs_mat[action.getIndex()] = np.zeros((num_state, num_obs))
                obs_mat[action.getIndex()][:, num_obs-1] = 1.0
                
            elif action.qd_type == 'ask':
                if action.q_type == 'wh':
                    if action.var == 'task':
                        tp_rate = 1.0 / pow(num_task, magic_number)
                        for state in self.state_set:
                            if state.isTerminal() == True:
                                continue
                            for observation in self.observation_set:
                                if observation.qd_type == 'wh':
                                    if observation.var == 'task':
                                        if observation.gettaskIndex() == \
                                            state.gettaskIndex():
                                            
                                            obs_mat[action.getIndex()]\
                                                [state.getIndex()]\
                                                [observation.getIndex()] = \
                                                tp_rate
                                        else:
                                            obs_mat[action.getIndex()]\
                                                [state.getIndex()]\
                                                [observation.getIndex()] = \
                                                (1.0-tp_rate) / (num_task-1)
                                    
                    elif action.var == 'patient':
                        tp_rate = 1.0 / pow(num_patient, magic_number)
                        for state in self.state_set:
                            if state.isTerminal() == True:
                                continue
                            for observation in self.observation_set:
                                if observation.qd_type == 'wh':
                                    if observation.var == 'patient':
                                        if observation.getpatientIndex() == \
                                            state.getpatientIndex():
                                            
                                            obs_mat[action.getIndex()]\
                                                [state.getIndex()]\
                                                [observation.getIndex()]\
                                                = tp_rate
                                        else:
                                            obs_mat[action.getIndex()]\
                                                [state.getIndex()]\
                                                [observation.getIndex()]\
                                                = (1.0-tp_rate) / (num_patient-1)

                    elif action.var == 'recipient':
                        tp_rate = 1.0 / pow(num_recipient, magic_number)
                        for state in self.state_set:
                            if state.isTerminal() == True:
                                continue
                            for observation in self.observation_set:
                                if observation.qd_type == 'wh':
                                    if observation.var == 'recipient':
                                        if observation.getrecipientIndex() ==\
                                            state.getrecipientIndex():
                                            
                                            obs_mat[action.getIndex()]\
                                                [state.getIndex()]\
                                                [observation.getIndex()]\
                                                = tp_rate
                                        else:
                                            obs_mat[action.getIndex()]\
                                                [state.getIndex()]\
                                                [observation.getIndex()]\
                                                = (1.0-tp_rate) / (num_recipient-1)
                    
                elif action.q_type == 'polar':
                    if action.var == 'task':
                        for state in self.state_set:
                            if state.isTerminal() == True:
                                continue
                            for observation in self.observation_set:
                                if observation.getName() == 'yes':
                                    if state.gettaskIndex() ==\
                                        action.gettaskIndex():
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
                                    if state.gettaskIndex() ==\
                                        action.gettaskIndex():
                                        obs_mat[action.getIndex()]\
                                            [state.getIndex()]\
                                            [observation.getIndex()]= \
                                            1.0 - self.polar_tn_rate
                                    else:
                                        obs_mat[action.getIndex()]\
                                            [state.getIndex()]\
                                            [observation.getIndex()] \
                                            = self.polar_tn_rate
                    elif action.var == 'patient':
                        for state in self.state_set:
                            if state.isTerminal() == True:
                                continue
                            for observation in self.observation_set:
                                if observation.getName() == 'yes':
                                    if state.getpatientIndex() == \
                                        action.getpatientIndex():
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
                                    if state.getpatientIndex() == \
                                        action.getpatientIndex():
                                        obs_mat[action.getIndex()]\
                                            [state.getIndex()]\
                                            [observation.getIndex()]= \
                                            1.0 - self.polar_tn_rate
                                    else:
                                        obs_mat[action.getIndex()]\
                                            [state.getIndex()]\
                                            [observation.getIndex()]=\
                                            self.polar_tn_rate
                    elif action.var == 'recipient':
                        for state in self.state_set:
                            if state.isTerminal() == True:
                                continue
                            for observation in self.observation_set:
                                if observation.getName() == 'yes':
                                    if state.getrecipientIndex() ==\
                                        action.getrecipientIndex():
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
                                    if state.getrecipientIndex() ==\
                                        action.getrecipientIndex():
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

    def computeRewardFunction(self, num_task, num_patient, num_recipient,
        r_max, r_min, weight_t, weight_p, weight_r, wh_cost, yesno_cost):

        reward_mat = np.zeros((len(self.action_set), len(self.state_set), ))

        for action in self.action_set:

            if action.qd_type == 'ask':

                if action.q_type == 'wh':
                    reward_mat[action.getIndex()] = wh_cost
                elif action.q_type == 'polar':
                    reward_mat[action.getIndex()] = yesno_cost

            elif action.qd_type == 'executeTask':
 
                for state in self.state_set:
                    
                    if state.isTerminal() == False:

                        reward_mat[action.getIndex()][state.getIndex()] = \
                            self.executeTaskReward(r_max, r_min, 
                            weight_t, weight_p, weight_r, action, state)

            num_state = len(self.tablelist) + 1
            reward_mat[action.getIndex()][num_state - 1] = 0.0

        return reward_mat

    def executeTaskReward(self, r_max, r_min, weight_t, weight_p, weight_r,
        action, state):

        if weight_t[action.gettaskIndex()][state.gettaskIndex()] == 1.0 and  \
                weight_p[action.getpatientIndex()][state.getpatientIndex()] == 1.0 and  \
                weight_r[action.getrecipientIndex()][state.getrecipientIndex()] == 1.0:
            return r_max
        else:
            ret = r_min * (1.0 - \
                weight_t[action.gettaskIndex()][state.gettaskIndex()] * \
                weight_p[action.getpatientIndex()][state.getpatientIndex()] * \
                weight_r[action.getrecipientIndex()][state.getrecipientIndex()])
            return ret

    def computeStateSet(self, num_task, num_patient, num_recipient):

        ret = []
        for t in range(num_task):
            for p in range(num_patient):
                for r in range(num_recipient):
                    for table in self.tablelist:
                        if t == table[0] and p == table[1] and r == table[2]:
                            ret.append(StateRegular(t, p, r, num_task, num_patient, num_recipient,self.tablelist.index(table)))

        ret.append(StateTerminal(num_task, num_patient, num_recipient, len(self.tablelist)))

        return ret

    def computeActionSet(self, num_task, num_patient, num_recipient):

        ret = []
        ret.append(ActionAskWh('task'))
        ret.append(ActionAskWh('patient'))
        ret.append(ActionAskWh('recipient'))
        
        for i in range(num_task):
            ret.append(ActionAskPolartask(i, num_task, num_patient, num_recipient))

        for c in range(num_patient):
            ret.append(ActionAskPolarpatient(c, num_task, num_patient, num_recipient))

        for y in range(num_recipient):
            ret.append(ActionAskPolarrecipient(y, num_task, num_patient, num_recipient))

        for i in range(num_task):
            for c in range(num_patient):
                for y in range(num_recipient):
                    for table in self.tablelist:
                        if i == table[0] and c == table[1] and y == table[2]:
                            ret.append(ActionDeliver(i, c, y, num_task, num_patient,
                                num_recipient, self.tablelist.index(table)))

        return ret

    def computeObservationSet(self, num_task, num_patient, num_recipient):

        ret = []
        for i in range(num_task):
            ret.append(ObservationWhtask(i, num_task, num_patient, num_recipient))

        for p in range(num_patient):
            ret.append(ObservationWhpatient(p, num_task, num_patient, num_recipient))

        for r in range(num_recipient):
            ret.append(ObservationWhrecipient(r, num_task, num_patient, num_recipient))

        ret.append(ObservationPolar('yes', num_task, num_patient, num_recipient))
        ret.append(ObservationPolar('no', num_task, num_patient, num_recipient))
        ret.append(ObservationNone(num_task, num_patient, num_recipient))

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

    num_task = 3
    num_patient = 3
    num_recipient = 3

    # row corresponds to action, column to underlying state
    # all
 
    # strategy = str(num_task) + str(num_patient) + str(num_recipient) 
    # strategy = str(num_task) + str(num_patient) + str(num_recipient) + '_' + str(entry)
    # strategy = str(num_task) + str(num_patient) + str(num_recipient) + '_' + str(entry1) + str(entry2)
    strategy = str(num_task) + str(num_patient) + str(num_recipient)

    pg = PomdpGenerator(num_task, num_patient, num_recipient, r_max, r_min, strategy, \
        wh_cost, yesno_cost,timeout=5)

if __name__ == '__main__':

    main()


