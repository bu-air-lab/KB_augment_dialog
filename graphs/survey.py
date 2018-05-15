import matplotlib.pyplot as plt
from matplotlib import mlab
from matplotlib.patches import Patch
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
        t1.append((trial1[i].count(j)/n) * 100)
        t2.append((trial2[i].count(j)/n) * 100)

    d1.append(t1)
    d2.append(t2)

colors = ['firebrick','indianred','lightcoral','mistyrose','gainsboro']
cols = ['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree']
questions = ['The tasks were easy to understand',
        'The robot understood me',
        'The robot frustrated me',
        'I would use the robot to\n get items for myself or other']
keys = ['Q1','Q2','Q3','Q4']
legend_elements = [Patch(facecolor='firebrick', label='Strongly Disagree'),
                   Patch(facecolor='indianred', label='Disagree'),
                   Patch(facecolor='lightcoral', label='Neutral'),
                   Patch(facecolor='mistyrose', label='Agree'),
                   Patch(facecolor='gainsboro', label='Strongly Agree'),
                   Patch(hatch='...', label='Our Method'),
                   Patch(hatch='\\\\', label='Baseline')]

df1 = pd.DataFrame(data=d1, columns=cols, index=questions)
df2 = pd.DataFrame(data=d2, columns=cols, index=questions)

fig, ax = plt.subplots(figsize=(9,6))
df1.plot.barh(hatch='...', width=0.25, position=0, stacked=True, color=colors, ax=ax, edgecolor='black',  legend=False)
df2.plot.barh(hatch='\\\\', width=0.25, position=1, stacked=True, color=colors, ax=ax, edgecolor='black', legend=False)

ax.legend(handles=legend_elements, loc='2', bbox_to_anchor=(1.05,1), borderaxespad=0.)
plt.yticks(clip_on=False)
plt.xlabel('Participants (%)')
plt.subplots_adjust(left=0.33, right=0.75)

#plt.margins(0.5)
plt.show()
fig.savefig('survey.svg')
