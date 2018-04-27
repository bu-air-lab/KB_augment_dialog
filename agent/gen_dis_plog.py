#!/usr/bin/env python

import sys
import shutil
import os
import subprocess
import conf

class DistrGen(object):

  def __init__(self):
    self.tablelist = conf.tablelist
    self.identity=['alice','bob','carol'] 
    self.college=['engineering','education','bussiness']
    self.year=['seventies','eighties','nineties']

  def __initpast__(self, plog_file = 'program_0.plog'):
    if os.path.isfile(plog_file) is False:
      print('p-log file does not exist: ' + plog_file)
      sys.exit()

    shutil.copy(plog_file, '/tmp')
    f = open('/tmp/' + plog_file, 'r')
    s = f.read()

    identity_list = ['alice', 'bob', 'carol']
    college_list=['engineering','education','bussiness']
    years_list=['seventies','eighties','nineties']
    mood_list = ['happy', 'sad']
    following_list = ['yes', 'no']

    output = ''

    for mood in mood_list:
      # print mood+'......'
      for fl in following_list:
        print( mood+'......' + fl + '......')
        output = ''
        for ident in identity_list:
          for coll in college_list:
            for year in years_list:
              query = '?{best_t(' + ident + ',' + coll + ',' + year + ')}'
              if fl is '':
                query += '.'
              else:
                query += '|obs(interest_m=' + mood + '),obs(interest_f=' + fl + '),do(last_table(bob,education,nineties)).'
              tmp_name = '/tmp/' + plog_file + '.tmp'
              with open(tmp_name, 'w') as ff:
                ff.write(s)
              with open(tmp_name, 'a') as ff:
                ff.write(query)

              out = subprocess.check_output('/home/sujay/context_aware_icorpp/plog/src/plog -t ' + tmp_name, shell = True)
              # print out
              out = out.split('\n')
              out = out[3]
              out = out.split(' ')
              out = out[3:]

              
              output += out[0] + ', '
              print(ident + '-' + coll + '-' + year + ': ' + out[0])
        print(output)

    f.close()

  def cal_belief(self, plog_file = 'guide.plog',\
                  mood= 'sad', \
                  foll= True, \
                  curr_table=-1, \
                  prev_table=-1, \
                  pdpDist = []):
    if os.path.isfile(plog_file) is False:
      print('p-log file does not exist: ' + plog_file)
      sys.exit()

    shutil.copy(plog_file, '/tmp')
    f = open('/tmp/' + plog_file, 'r')
    s = f.read()

    identity_list = ['alice', 'bob', 'carol']
    college_list=['engineering','education','bussiness']
    years_list=['seventies','eighties','nineties']

    output = ''
    # print( mood+'......' + fl + '......')
    for i in range(len(self.tablelist)):
      query = ''
      for j in range(len(self.tablelist)):
        query += '[rt] pr(interest_t=t' +  str(j) + ') = '+ str(int(pdpDist[j]*100))+'/100.\n'
      query += '?{interest_t=t' + str(i) + '}'
      if mood is 'sad':
        if curr_table != -1:
          cur_str = '|do(curr_table('+self.identity[self.tablelist[curr_table][0]] + ',' \
          			+ self.college[self.tablelist[curr_table][1]] + ',' + self.year[self.tablelist[curr_table][2]]+'))'
          query += cur_str
        if prev_table != -1:
          cur_str = ',do(prev_table('+self.identity[self.tablelist[prev_table][0]] + ',' \
                + self.college[self.tablelist[prev_table][1]] + ',' + self.year[self.tablelist[prev_table][2]]+'))'
          query += cur_str
        if not foll:
          query += ', obs(obs_f=false)'
          # print '\n',cur_str
          # if last_table is not '':
          #   query += ',do(' + last_table + ')'
        # else:
        #   if last_table is not '':
        #     query += '|do(' + last_table + ')'
        query += ', obs(valid), obs(obs_s=true).'
      else:
        if curr_table != -1:
          query += '|do(curr_table('+self.identity[self.tablelist[curr_table][0]] + ',' \
          			+ self.college[self.tablelist[curr_table][1]] + ',' + self.year[self.tablelist[curr_table][2]]+'))'
        if prev_table != -1:
          cur_str = ',do(prev_table('+self.identity[self.tablelist[prev_table][0]] + ',' \
                + self.college[self.tablelist[prev_table][1]] + ',' + self.year[self.tablelist[prev_table][2]]+'))'
          query += cur_str
        if not foll:
          query += ', obs(obs_f=false)'
          
        query += ',obs(valid).'
      # print query
      tmp_name = '/tmp/' + plog_file + '.tmp'
      with open(tmp_name, 'w') as ff:
        ff.write(s)
      with open(tmp_name, 'a') as ff:
        ff.write(query)



      plog_lu = '/home/ludc/workspace/context_aware_icorpp/plog/src/plog'
      plog_zhang = '/home/szhang/software/plog/plog/src/plog'
      plog_sujay = '/home/sujay/context_aware_icorpp/plog/src/plog'

      if os.path.isfile(plog_lu):
        plog = plog_lu
      elif os.path.isfile(plog_zhang):
        plog = plog_zhang
      elif os.path.isfile(plog_sujay):
        plog = plog_sujay
      else:
        print "plog not installed on this machine"
        exit(1)

      out = subprocess.check_output(plog + ' -t ' + tmp_name, shell = True)
      # print query, out
      out = out.split('\n')
      out = out[3]
      out = out.split(' ')
      out = out[3:]
      
      output += out[0] + ', '
          # print(ident + '-' + coll + '-' + year + ': ' + out[0])
    output += '0'
    # print(output)

    f.close()
    return output

def main():
  d = DistrGen(plog_file='guide.plog')
  subprocess.check_output('rm out.txt result.txt', shell = True)

if __name__ == '__main__':
  d = DistrGen()
  d.cal_belief(mood = 'sad', curr_table = 4, pdpDist = [0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11])
  # main()
