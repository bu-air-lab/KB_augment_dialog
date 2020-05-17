from __future__ import division
from simulator_noparser import Simulator
from simulator_baseline_belief import Baseline
from simulator_baseline_entropy import Baseline2
import pandas as pd
import ast
import matplotlib
import math
import matplotlib.pyplot as plt
import csv
from mpl_toolkits.mplot3d import Axes3D



def hypo2(df1,df2,df3,filelist,num):

        g=plt.figure(figsize=(10,5))
        
	#plt.subplot(132)
	plt.plot([17,26,37],df1.loc[filelist[0]:filelist[-1],'F1 Score'],marker='*',linestyle='-',label='Dual-track POMDP Manager')
	plt.plot([17,26,37],df2.loc[filelist[0]:filelist[-1],'F1 Score'],marker='o',linestyle='--',label='Baseline Belief')
	plt.plot([17,26,37],df3.loc[filelist[0]:filelist[-1],'F1 Score'],marker='v',linestyle='-.',label='Baseline EF')
	plt.ylim(0.2,0.8)
	plt.ylabel('F1 Score')
	plt.xlabel('KB Size')
	axes = plt.gca()
	xleft , xright =axes.get_xlim()
	ybottom , ytop = axes.get_ylim()
	axes.set_aspect(aspect=abs((xright-xleft)/(ybottom-ytop)), adjustable=None, anchor=None)  

	#plt.subplot(132)
	#plt.plot([10,17,26,37],df1.loc[filelist[0]:filelist[-1],'Precision'],marker='*',linestyle='-',label='Dual-track POMDP Manager')
	#plt.plot([10,17,26,37],df2.loc[filelist[0]:filelist[-1],'Precision'],marker='o',linestyle='--',label='Baseline Belief')
	#plt.plot([10,17,26,37],df3.loc[filelist[0]:filelist[-1],'Precision'],marker='v',linestyle='-.',label='Baseline EF')
	#plt.ylim(0,1.2)
	#plt.ylabel('Precision')
	#plt.xlabel('KB Size')
	#axes = plt.gca()
	#xleft , xright =axes.get_xlim()
	#ybottom , ytop = axes.get_ylim()
	#axes.set_aspect(aspect=abs((xright-xleft)/(ybottom-ytop)), adjustable=None, anchor=None)  


	#plt.subplot(131)
	#plt.plot([10,17,26,37],df1.loc[filelist[0]:filelist[-1],'Recall'],marker='*',linestyle='-',label='Dual-track POMDP Manager')
	#plt.plot([10,17,26,37],df2.loc[filelist[0]:filelist[-1],'Recall'],marker='o',linestyle='--',label='Baseline Belief')
	#plt.plot([10,17,26,37],df3.loc[filelist[0]:filelist[-1],'Recall'],marker='v',linestyle='-.',label='Baseline EF')
	#plt.ylim(0,1.5)
	#plt.ylabel('Recall')
	#plt.xlabel('KB Size')

	axes = plt.gca()
	xleft , xright =axes.get_xlim()
	ybottom , ytop = axes.get_ylim()
	axes.set_aspect(aspect=abs((xright-xleft)/(ybottom-ytop)), adjustable=None, anchor=None)  
	
	
	plt.legend(loc='upper center', fancybox=True, framealpha=0.5)

	
	plt.show()


def hypo1(df1, df2, df3, filelist, num):

        
        plt.scatter(df1.loc[filelist[0]:filelist[-1],'Efficiency'], df1.loc[filelist[0]:filelist[-1],'Accuracy'], marker= 'o', label='Dual-track POMDP Manager')
        plt.scatter(df2.loc[filelist[0]:filelist[-1],'Efficiency'], df2.loc[filelist[0]:filelist[-1],'Accuracy'], marker = '^', label='Baseline Belief')
        plt.scatter(df3.loc[filelist[0]:filelist[-1],'Efficiency'], df3.loc[filelist[0]:filelist[-1],'Accuracy'], marker = 'v', label='Baseline EF')

        for index,row in df1.iterrows():
            plt.annotate(index, (row['Efficiency'], row['Accuracy']))
 
        for index,row in df2.iterrows():
            plt.annotate(index, (row['Efficiency'], row['Accuracy']))


        for index,row in df3.iterrows():
            plt.annotate(index, (row['Efficiency'], row['Accuracy']))

        
        plt.xlabel('Dialog Turn')
        plt.ylabel('Accuracy')


        plt.legend(loc='upper center', fancybox=True, framealpha=0.5)

        #plt.legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=3, 
            #borderaxespad=0, frameon=False)
        
        plt.show()
       


def hypo3(df1,df2,df3,filelist,num):

	g=plt.figure(figsize=(15,5))
	
	font_size = 20

	plt.subplot(131)
	plt.plot([17,26,37],df1.loc[filelist[0]:filelist[-1],'QA Cost'],marker='*',linestyle='-',label='Dual-track POMDP Manager')
	plt.plot([17,26,37],df2.loc[filelist[0]:filelist[-1],'QA Cost'],marker='o',linestyle='--',label='Baseline Belief')
	plt.plot([17,26,37],df3.loc[filelist[0]:filelist[-1],'QA Cost'],marker='v',linestyle='-.',label='Baseline EF')
	#plt.xlim(8,40)
	#matplotlib.pyplot.xticks([10,17,26,37], fontsize = font_size)
	#plt.tick_params(labelsize=font_size)
	plt.ylabel('QA Cost')
	plt.xlabel('KB Size')
	axes = plt.gca()
	xleft , xright =axes.get_xlim()
	ybottom , ytop = axes.get_ylim()
	axes.set_aspect(aspect=abs((xright-xleft)/(ybottom-ytop)), adjustable=None, anchor=None)  

	plt.subplot(132)
	plt.plot([17,26,37],df1.loc[filelist[0]:filelist[-1],'Dialog Reward'],marker='*',linestyle='-',label='Dual-track POMDP Manager')
	plt.plot([17,26,37],df2.loc[filelist[0]:filelist[-1],'Dialog Reward'],marker='o',linestyle='--',label='Baseline Belief')
	plt.plot([17,26,37],df3.loc[filelist[0]:filelist[-1],'Dialog Reward'],marker='v',linestyle='-.',label='Baseline EF')
	#plt.xlim(8,40)
	#matplotlib.pyplot.xticks([17,26,37], fontsize = font_size)	
	#plt.tick_params(labelsize=font_size)
	plt.ylabel('Dialog Reward')
	plt.xlabel('KB Size')
	plt.ylim(-30, -12)
	axes = plt.gca()
	xleft , xright =axes.get_xlim()
	ybottom , ytop = axes.get_ylim()
	axes.set_aspect(aspect=abs((xright-xleft)/(ybottom-ytop)), adjustable=None, anchor=None)  


	plt.subplot(133)
	plt.plot([17,26,37],df1.loc[filelist[0]:filelist[-1],'Overall Success'],marker='*',linestyle='-',label='Dual-track POMDP Manager')
	plt.plot([17,26,37],df2.loc[filelist[0]:filelist[-1],'Overall Success'],marker='o',linestyle='--',label='Baseline Belief')
	plt.plot([17,26,37],df3.loc[filelist[0]:filelist[-1],'Overall Success'],marker='v',linestyle='-.',label='Baseline EF')
	plt.ylim(0.24, 0.5)
	#matplotlib.pyplot.xticks([10,17,26,37], fontsize = font_size)
	#plt.tick_params(labelsize=font_size)
	plt.ylabel('Overall Success')
	plt.xlabel('KB Size')
	axes = plt.gca()
	xleft , xright =axes.get_xlim()
	ybottom , ytop = axes.get_ylim()
	axes.set_aspect(aspect=abs((xright-xleft)/(ybottom-ytop)), adjustable=None, anchor=None)  


	plt.legend(loc='upper center', fancybox=True, framealpha=0.5)


	plt.show()
	g.savefig('Plots/all_'+str(num)+'_trials_domain_experiment_entropy_5_belief_0_45_N_1')




def main():

	num = 5000                    #number of trials
	filelist=['144','155','166']

	df1 = pd.read_csv("Plots_data/all_5000_pomdpdual.csv", encoding='utf-8')

	df2 = pd.read_csv("Plots_data/all_5000_baseline1.csv", encoding='utf-8')

	df3 = pd.read_csv("Plots_data/all_5000_baseline2.csv", encoding='utf-8')


	print ('Dual Track POMDP results')	
	print(df1)

	print()

	print ('Baseline Belief')
	print(df2)

	print()
	
	print ('Baseline EF')
	print(df3)

	df1.sort_values(by=['QA Cost'])
	df2.sort_values(by=['QA Cost'])
	df3.sort_values(by=['QA Cost'])

	hypo1(df1,df2, df3,filelist,num)
	hypo2(df1,df2, df3,filelist,num)
	hypo3(df1,df2, df3,filelist,num)





if __name__ == '__main__':
	
	main()






