#!/usr/bin/env python

import sys
import shutil
import os
import subprocess

class DistrGen(object):

  def __init__(self, plog_file = 'program_0.plog'):
    if os.path.isfile(plog_file) is False:
      print('p-log file does not exist: ' + plog_file)
      sys.exit()

    shutil.copy(plog_file, '/tmp')
    f = open('/tmp/' + plog_file, 'r')
    s = f.read()

    item_list = ['sandwich', 'coffee']
    person_list = ['alice', 'bob', 'carol']
    room_list = ['alice_office', 'bob_office', 'lab', 'conf']
    time_list = ['morning', 'noon', 'afternoon', '']

    output = ''

    for time in time_list:
      print(time + '......')
      output = ''
      for item in item_list:
        for person in person_list:
          for room in room_list:
            query = '?{task(' + item + ',' + person + ',' + room + ')}'
            if time is '':
              query += '.'
            else:
              query += '|obs(shoppingtime=' + time + ').'
            
            tmp_name = '/tmp/' + plog_file + '.tmp'
            with open(tmp_name, 'w') as ff:
              ff.write(s)
            with open(tmp_name, 'a') as ff:
              ff.write(query)

            out = subprocess.check_output('/home/szhang/software/p-log/plog/src/plog -t ' + tmp_name, shell = True)
            out = out.split('\n')
            out = out[3]
            out = out.split(' ')
            out = out[3:]
            
            output += out[0] + ', '
            print(item + '-' + person + '-' + room + ': ' + out[0])

      print
      print('Time: ' + time)
      print(output)
      print

    f.close()

def main():
  d = DistrGen(plog_file='program_3.plog')
  subprocess.check_output('rm out.txt result.txt', shell = True)

if __name__ == '__main__':
  main()
