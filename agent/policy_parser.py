#!/usr/bin/env python

import sys
import numpy
from numpy import matrix

class Policy(object):
  actions = None
  policy = None

  def __init__(self, num_states, num_actions, filename='policy/default.policy'):
    try:
      f = open(filename, 'r')
    except:
      print('\nError: unable to open file: ' + filename)

    lines = f.readlines()

    # the first three and the last lines are not related to the actual policy
    lines = lines[3:]

    self.actions = -1 * numpy.ones((len(lines)-1, 1, ))
    self.policy = numpy.zeros((len(lines)-1, num_states, ))

    for i in range(len(lines)-1):
      # print("this line:\n\n" + lines[i])
      if lines[i].find('/AlphaVector') >= 0:
        break
      l = lines[i].find('"')
      r = lines[i].find('"', l + 1)
      self.actions[i] = int(lines[i][l + 1 : r])

      ll = lines[i].find('>')
      rr = lines[i].find(' <')
      # print(str(i))
      self.policy[i] = numpy.matrix(lines[i][ll + 1 : rr])

    f.close()
    
  def select_action(self, b):
    
    # sanity check if probabilities sum up to 1
    if sum(b) - 1.0 > 0.00001:
      print('Error: belief does not sum to 1, diff: ', sum(b)[0] - 1.0)
      sys.exit()

    return self.actions[numpy.argmax(numpy.dot(self.policy, b.T)), 0]
    # return numpy.argmax(b) + 12
    # return numpy.random.randint(24, size=1)[0]


