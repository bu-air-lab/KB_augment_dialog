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

# abs(cost) makes cost values positive


def hypo2(df1,df2,df3,filelist,num):
        
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

        
        plt.xlabel('Efficiency')
        plt.ylabel('Accuracy')

        plt.legend(loc='upper left', bbox_to_anchor=(0.5, -0.1),  shadow=True, ncol=3)
        
        plt.show()
       


def hypo3(df1,df2,df3,filelist,num):

	g=plt.figure(figsize=(15,5))
	
	font_size = 20

	plt.subplot(131)
	plt.errorbar([17,26,37],df1.loc[filelist[0]:filelist[-1],'QA Cost'], yerr=df1['QACostStd'].tolist(), color='y', capsize=5, marker='*',linestyle='-',label='Dual-track POMDP Manager')
	plt.errorbar([17,26,37],df2.loc[filelist[0]:filelist[-1],'QA Cost'], yerr=df2['QACostStd'].tolist(),  marker='o', color='k', capsize=5, linestyle='--',label='Baseline Belief')
	plt.errorbar([17,26,37],df3.loc[filelist[0]:filelist[-1],'QA Cost'], yerr=df3['QACostStd'].tolist(),  marker='v', color = 'r', capsize=5, linestyle='-.',label='Baseline EF')
	plt.xlim(8,40)
	matplotlib.pyplot.xticks([10,17,26,37], fontsize = font_size)
	plt.tick_params(labelsize=font_size)
	plt.ylabel('QA Cost', fontsize = font_size)
	plt.xlabel('KB Size', fontsize = font_size)
	axes = plt.gca()
	xleft , xright =axes.get_xlim()
	ybottom , ytop = axes.get_ylim()
	axes.set_aspect(aspect=abs((xright-xleft)/(ybottom-ytop)), adjustable=None, anchor=None)  

	plt.subplot(132)
	plt.errorbar([17,26,37],df1.loc[filelist[0]:filelist[-1],'Dialog Reward'], yerr=df1['RewardStd'].tolist(), color='y', capsize=5, marker='*',linestyle='-',label='Dual-track POMDP Manager')
	plt.errorbar([17,26,37],df2.loc[filelist[0]:filelist[-1],'Dialog Reward'], yerr=df2['RewardStd'].tolist(), color='k', capsize=5, marker='o',linestyle='--',label='Baseline Belief')
	plt.errorbar([17,26,37],df3.loc[filelist[0]:filelist[-1],'Dialog Reward'], yerr=df3['RewardStd'].tolist(), color='r', capsize=5, marker='v',linestyle='-.',label='Baseline EF')
	plt.xlim(8,40)
	matplotlib.pyplot.xticks([17,26,37], fontsize = font_size)	
	plt.tick_params(labelsize=font_size)
	plt.ylabel('Dialog Reward', fontsize = font_size)
	plt.xlabel('KB Size', fontsize = font_size)
	#plt.ylim(-30, -8)
	axes = plt.gca()
	xleft , xright =axes.get_xlim()
	ybottom , ytop = axes.get_ylim()
	axes.set_aspect(aspect=abs((xright-xleft)/(ybottom-ytop)), adjustable=None, anchor=None)  


	plt.subplot(133)
	plt.plot([17,26,37],df1.loc[filelist[0]:filelist[-1],'Overall Success'],marker='*', color='y',  linestyle='-',label='Dual-track POMDP Manager')
	plt.plot([17,26,37],df2.loc[filelist[0]:filelist[-1],'Overall Success'],marker='o',color='k',   linestyle='--',label='Baseline Belief')
	plt.plot([17,26,37],df3.loc[filelist[0]:filelist[-1],'Overall Success'],marker='v',color='r',  linestyle='-.',label='Baseline EF')
	plt.xlim(8,40)
	plt.ylim(0.2, 0.5)
	matplotlib.pyplot.xticks([10,17,26,37], fontsize = font_size)
	plt.tick_params(labelsize=font_size)
	plt.ylabel('Overall Success', fontsize = font_size)
	plt.xlabel('KB Size', fontsize = font_size)
	axes = plt.gca()
	xleft , xright =axes.get_xlim()
	ybottom , ytop = axes.get_ylim()
	axes.set_aspect(aspect=abs((xright-xleft)/(ybottom-ytop)), adjustable=None, anchor=None)  


	plt.legend(loc='upper center', bbox_to_anchor=(-1, 1.4),  shadow=True, ncol=2, fontsize=font_size)
	plt.show()
	g.savefig('Plots/all_'+str(num)+'_trials_domain_experiment_entropy_5_belief_0_45_N_1')

def sigmoid(x,i):
  consList = [0.182, 0.1733, 0.1475]
  return (1 / (1 + math.exp(-x))) + 1/-x



def main():


	num = 3000                 #number of trials
	filelist=['144','155','166']                     #list of pomdp files

	belieflist=[0.35,0.20,0.18,0.15]
	addList = [0.1, 0.01, -0.06]
	i=0
	
	df1=pd.DataFrame()
	df2=pd.DataFrame() 
	df3=pd.DataFrame()

	# just use for sth in somelist, not for sth in range(len(ssomelist))
	for iterator in filelist:
		name = iterator  # or name = iterator

		s = Simulator(uniform_init_belief = True, 
			auto_state = True, 
			auto_observations = True, # was true
			print_flag = False,
			policy_file = name+'.policy',
			pomdp_file =  name +'.pomdp',
			pomdp_file_plus=list(name)[0]+str(int(list(name)[1])+1)+str(int(list(name)[2])+1)+'.pomdp',
			policy_file_plus=list(name)[0]+str(int(list(name)[1])+1)+str(int(list(name)[2])+1)+'.policy',
			trials_num = num,
			num_task = int(name[0]), 
			num_patient = int(name[1]), 
			num_recipient = int(name[2]),
			belief_threshold = sigmoid(-1 * int(name[2]), i), #Sigmoid
			ent_threshold =  max(7,int(name[2])) ) #ReLU
			


		# abs(math.log(int(name[2])/10 + addList[i] , (int(name[2])**2)))
		#belief = ( 0.8 / ( (int(name[2]) ** 2 ) + 1 ) ) * (((int(name[2]) ** 2 ) + 1)/2) - (1.0/((int(name[2])**2)+1))
		#ent = int(round(((int(name[2]) * int(name[2])) + 1) * int(name[2])/85) + int(name[2])
		#int(round(((int(name[2]) * int(name[2])) + 1) * 0.15) + int(name[2])
		#0.33
	 
		if not s.uniform_init_belief:   
			print('note that initial belief is not uniform\n')
		s.read_model_plus()

		base = Baseline(uniform_init_belief = True,  #Baseline just uses Belif Threshold to augment its KB.
			auto_state = True, 
			auto_observations = True, # was true
			print_flag = False,
			policy_file = name+'.policy',
			pomdp_file =  name +'.pomdp',
		        pomdp_file_plus=list(name)[0]+str(int(list(name)[1])+1)+str(int(list(name)[2])+1)+'.pomdp',
			policy_file_plus=list(name)[0]+str(int(list(name)[1])+1)+str(int(list(name)[2])+1)+'.policy',
			trials_num = num,
			num_task = int(name[0]), 
			num_patient = int(name[1]), 
			num_recipient = int(name[2]),
			belief_threshold = sigmoid(-1 * int(name[2]), i), # ((0.8 / KB size) * (KB size/2)) - (1/Recipient*5)
			ent_threshold =999 )
	 
		if not base.uniform_init_belief:   
			print('note that initial belief is not uniform\n')
		base.read_model_plus()



		base2 = Baseline2(uniform_init_belief = True,  #Baseline just uses EF to augment its KB.
			auto_state = True, 
			auto_observations = True, # was true
			print_flag = False,
			policy_file = name+'.policy',
			pomdp_file =  name +'.pomdp',
		    pomdp_file_plus=list(name)[0]+str(int(list(name)[1])+1)+str(int(list(name)[2])+1)+'.pomdp',
			policy_file_plus=list(name)[0]+str(int(list(name)[1])+1)+str(int(list(name)[2])+1)+'.policy',
			trials_num = num,
			num_task = int(name[0]), 
			num_patient = int(name[1]), 
			num_recipient = int(name[2]),
			belief_threshold = 999,
			ent_threshold = max(7,int(name[2]))) #ReLU
	 
		if not base2.uniform_init_belief:   
			print('note that initial belief is not uniform\n')
		base2.read_model_plus()

		# int((0.4 -  1/((int(name[2])**2)+1)) / (1/((int(name[2])**2))))





		###Saving results in a dataframe and passing data frame to plot generate_function

		#Put i or name or whatever the name of the iterator is, below in df.at[i, e.g. "Overall Cost"]
		
		a,b,c,p,r,f,ap,sh1, stdCost1, stdRew1=s.run_numbers_of_trials() 	#	f to solve too many values to unpack err. (Best N from experiments)
		ab,bb,cb,pb,rb,fb,apb,bh1, stdCost2, stdRew2= base.run_numbers_of_trials()
		ab2,bb2,cb2,pb2,rb2,fb2,apb2,bh1_2, stdCost3, stdRew3= base2.run_numbers_of_trials()
		
		df1.at[iterator,'QA Cost']= a
		df1.at[iterator,'Overall Success']= b
		df1.at[iterator,'Dialog Reward']= c
		df1.at[iterator,'Precision']= p
		df1.at[iterator,'Recall']= r
		df1.at[iterator,'F1 Score'] = f
		df1.at[iterator, 'QACostStd'] = stdCost1
                df1.at[iterator, 'RewardStd'] = stdRew1

		if(math.isnan(ap)): # For last KB, this value becomes NaN, because agent did not augment KB at all, with this assignment, figure will make sense
			ap = 50.0
		df1.at[iterator,'Efficiency']= ap
		df1.at[iterator,'Accuracy']= sh1
                

		df2.at[iterator,'QA Cost']= ab
		df2.at[iterator,'Overall Success']= bb
		df2.at[iterator,'Dialog Reward']= cb
		df2.at[iterator,'Precision']= pb
		df2.at[iterator,'Recall']= rb
		df2.at[iterator,'F1 Score'] = fb
                df2.at[iterator, 'QACostStd'] = stdCost2
                df2.at[iterator, 'RewardStd'] = stdRew2

		if(math.isnan(apb)): # For last KB, this value becomes NaN, because agent did not augment KB at all, with this assignment, figure will make sense
			apb = 50.0
		df2.at[iterator,'Efficiency']= apb
		df2.at[iterator,'Accuracy']= bh1



		df3.at[iterator,'QA Cost']= ab2
		df3.at[iterator,'Overall Success']= bb2
		df3.at[iterator,'Dialog Reward']= cb2
		df3.at[iterator,'Precision']= pb2
		df3.at[iterator,'Recall']= rb2
		df3.at[iterator,'F1 Score'] = fb2
                df3.at[iterator, 'QACostStd'] = stdCost3
                df3.at[iterator, 'RewardStd'] = stdRew3
		
		if(math.isnan(apb2)): # For last KB, this value becomes NaN, because agent did not augment KB at all, with this assignment, figure will make sense
			apb2 = 50.0
		df3.at[iterator,'Efficiency']= apb2
		df3.at[iterator,'Accuracy']= bh1_2

		i += 1


	df1.to_csv("Plots_data/all_"+str(num)+"_trials_domain_experiment_entropy_5_belief_0.35_entropyheuristic_pomdpdual.csv", encoding='utf-8', index=True)
	df2.to_csv("Plots_data/all_"+str(num)+"_trials_domain_experiment_entropy_5_justbelief_0.35_baseline1.csv", encoding='utf-8', index=True)
	df3.to_csv("Plots_data/all_"+str(num)+"_trials_domain_experiment_entropy_5_justentropyheuristic_baseline2.csv", encoding='utf-8', index=True)
	
	df1.sort_values(by=['QA Cost'])
	df2.sort_values(by=['QA Cost'])
	df3.sort_values(by=['QA Cost'])
	
	print ('Dual Track POMDP results')
	print (df1)
	print ('Baseline Belief')
	print (df2)
	print ('Baseline EF')
	print (df3)
	
	hypo1(df1,df2, df3,filelist,num)
	hypo2(df1,df2, df3,filelist,num)
	hypo3(df1,df2, df3,filelist,num)
	
	
	i+=1



if __name__ == '__main__':
	
	main()
