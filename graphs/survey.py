import matplotlib.pyplot as plt
from matplotlib import mlab
import numpy as np
import pandas as pd

trial1 = [[2,4,4,4,4,3,2,2,4,4,4,4], 
          [0,4,1,4,1,4,0,1,3,4,4,4],
          [1,0,2,1,3,1,4,1,0,2,0,3],
          [1,4,2,3,2,3,0,1,2,4,4,4]]

trial2 = [[3,4,3,4,4,3,2,3,4,2,4,4],
          [1,4,2,1,3,0,3,2,1,3,1,1],
          [2,0,1,3,2,3,3,1,3,2,4,2],
          [1,3,3,0,2,1,0,2,1,3,2,3]]

# likert data for trial 1 and 2
d1 = []
d2 = []
n = 12.0

for i in range(len(trial1)):
    t1 = []
    t2 = []
    # count each value (0-4)
    for j in range(5):
        t1.append(trial1[i].count(j)/n)
        t2.append(trial2[i].count(j)/n)

    d1.append(t1)
    d2.append(t2)

colors = ['firebrick','lightcoral','gainsboro','cornflowerblue', 'darkblue']
cols = ['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree']
questions = ['The tasks were easy to understand',
        'The robot understood me',
        'The robot frustrated me',
        'I would use the robot to get items for myself or other']
keys = ['Q1','Q2','Q3','Q4']

df1 = pd.DataFrame(data=d1, columns=cols, index=keys)
df2 = pd.DataFrame(data=d2, columns=cols, index=keys)

fig, ax = plt.subplots()
df1.plot.barh(width=0.25, position=0, stacked=True, color=colors, ax=ax, edgecolor='black')
df2.plot.barh(width=0.25, position=1, stacked=True, color=colors, ax=ax, edgecolor='black', legend=False)

#plt.margins(0.5)
plt.show()
