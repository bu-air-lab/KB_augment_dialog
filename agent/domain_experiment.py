from __future__ import division
from simulator_noparser import Simulator
from simulator_baseline import Baseline
import pandas as pd
import ast
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

# abs(cost) makes cost values positive



def hypo2(df1, df2,filelist,num):
        
        
        


        plt.subplot(133)
	plt.plot([10,17,26,37],df1.loc[filelist[0]:filelist[-1],'F1 Score'],marker='*',linestyle='-',label='Dual-track POMDP Manager')
	plt.plot([10,17,26,37],df2.loc[filelist[0]:filelist[-1],'F1 Score'],marker='o',linestyle='--',label='Baseline Learning Agent')
        plt.ylim(0,1.5)
        plt.ylabel('F1 Score')
	plt.xlabel('KB Size')


        plt.subplot(132)
	plt.plot([10,17,26,37],df1.loc[filelist[0]:filelist[-1],'Precision'],marker='*',linestyle='-',label='Dual-track POMDP Manager')
	plt.plot([10,17,26,37],df2.loc[filelist[0]:filelist[-1],'Precision'],marker='o',linestyle='--',label='Baseline Learning Agent')
        plt.ylim(0,1.2)
        plt.ylabel('Precision')
	plt.xlabel('KB Size')


        plt.subplot(131)
	plt.plot([10,17,26,37],df1.loc[filelist[0]:filelist[-1],'Recall'],marker='*',linestyle='-',label='Dual-track POMDP Manager')
	plt.plot([10,17,26,37],df2.loc[filelist[0]:filelist[-1],'Recall'],marker='o',linestyle='--',label='Baseline Learning Agent')
        plt.ylim(0,1.5)
        plt.ylabel('Recall')
	plt.xlabel('KB Size')


        plt.legend(loc='upper center')

        plt.show()



def plotgenerateF1Our(df1,filelist,num):
        
        
        plt.plot([10,17,26,37],df1.loc[filelist[0]:filelist[-1],'F1 Score'],marker='o',linestyle='-',label='F1 Score')
        plt.plot([10,17,26,37],df1.loc[filelist[0]:filelist[-1],'Precision'],marker='^',linestyle='--',label='Precision')
        plt.plot([10,17,26,37],df1.loc[filelist[0]:filelist[-1],'Recall'],marker='+',linestyle=':',label='Recall')

        #plt.xlim(1.5,max(belieflist)+1)
        plt.ylim(0,1.5)
        #plt.ylabel('Baseline Learning Agent')
        #plt.xlabel('KB Size')
        plt.legend(loc=0)
        #plt.xlim(1.5,max(belieflist)+1)
        #plt.ylim(0,1)
        plt.ylabel('Dual-track POMDP Manager')
        plt.xlabel('KB Size')
        plt.legend(loc=0)
        plt.show()	 	



def plotgenerate(df1,df2,filelist,num):
	######################################### Uncomment: Plot all 5 in one figure ###############################################################33
	'''
	fig=plt.figure(figsize=(3*len(list(df)),5))
	plt.suptitle('Increasing domain size', fontsize=14); 
	for count,metric in enumerate(list(df)):
		ax=plt.subplot(1,len(list(df)),count+1)
	  
		l1 = plt.plot(range(3,3+len(filelist)),df.loc[filelist[0]:filelist[-1],metric],marker='*',linestyle='-',label='Average of '+str(num)+ ' trials')
		#l1 = plt.plot(range(filelist[0],filelist[0]+len(filelist)),df.loc[filelist[0]:filelist[-1],metric],marker='*',linestyle='-',label='Average of '+str(num)+ ' trials')
		plt.ylabel(metric)
		plt.xlim(2.5,6.5)
		xleft , xright =ax.get_xlim()
		ybottom , ytop = ax.get_ylim()
		ax.set_aspect(aspect=abs((xright-xleft)/(ybottom-ytop)), adjustable=None, anchor=None)


		plt.xlabel('Knowledge size')


	#ax.legend(loc='upper left', bbox_to_anchor=(-2.10, 1.35),  shadow=True, ncol=5)
	fig.tight_layout()
	plt.show()
	fig.savefig('Results_'+str(num)+'_trials')
	
	#################################################### Split 5 plots to 2 figures: (cost,success, reward) and (precision, recall) ##################################
	#######################    Figure 1      ########################
	f=plt.figure(1, figsize=(3*len(list(df)),6))
	plt.suptitle('Increasing domain size, belief_threshold=0.7, entropy threshold=2', fontsize=14); 
	for count,metric in enumerate(list(df)):
		if count<3: 
			ax=plt.subplot(1,len(list(df))-2,count+1)

			l1 = plt.plot(range(3,3+len(filelist)),df.loc[filelist[0]:filelist[-1],metric],marker='*',linestyle='-',label='Average of '+str(num)+ ' trials')
			#l1 = plt.plot(range(filelist[0],filelist[0]+len(filelist)),df.loc[filelist[0]:filelist[-1],metric],marker='*',linestyle='-',label='Average of '+str(num)+ ' trials')
			plt.ylabel(metric)
			if metric=='Overall Success':
				plt.ylim(0,1)
			elif metric=='Overall Reward':
				plt.ylim(0,30)
			elif metric=='Overall Cost':
				plt.ylim(-30,0) 
			plt.xlim(2.5,6.5)
			xleft , xright =ax.get_xlim()
			ybottom , ytop = ax.get_ylim()
			ax.set_aspect(aspect=abs((xright-xleft)/(ybottom-ytop)), adjustable=None, anchor=None)


			plt.xlabel('Knowledge size')


	#ax.legend(loc='upper left', bbox_to_anchor=(-2.10, 1.35),  shadow=True, ncol=5)
	f.tight_layout()
	plt.show()
	f.savefig('Plots/cost_reward_success_'+str(num)+'_trials_domain_experiment_entropy_2_belief_07')

	#######################    Figure 2      ########################
	g=plt.figure(2, figsize=(3*len(list(df)),6))
	plt.suptitle('Increasing domain size, belief_threshold=0.7, entropy threshold=2', fontsize=14); 
	for count,metric in enumerate(['Precision','Recall']): 
		ax=plt.subplot(1,2,count+1)

		l1 = plt.plot(range(3,3+len(filelist)),df.loc[filelist[0]:filelist[-1],metric],marker='*',linestyle='-',label='Average of '+str(num)+ ' trials')
		#l1 = plt.plot(range(filelist[0],filelist[0]+len(filelist)),df.loc[filelist[0]:filelist[-1],metric],marker='*',linestyle='-',label='Average of '+str(num)+ ' trials')
		plt.ylabel(metric)
		plt.xlim(2.5,6.5)
		plt.ylim(0,1)
		xleft , xright =ax.get_xlim()
		ybottom , ytop = ax.get_ylim()
		ax.set_aspect(aspect=abs((xright-xleft)/(ybottom-ytop)), adjustable=None, anchor=None)


		plt.xlabel('Knowledge size')
	'''
	g=plt.figure(figsize=(15,5))
	#plt.suptitle('Increasing domain size, belief_threshold=0.4, entropy threshold=5, '+str(num)+ ' trials', fontsize=18);
	'''font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

	matplotlib.rc('font', **font)'''
	font_size = 20

	plt.subplot(131)
	plt.plot([10,17,26,37],df1.loc[filelist[0]:filelist[-1],'QA Cost'],marker='*',linestyle='-',label='Dual-track POMDP Manager')
	plt.plot([10,17,26,37],df2.loc[filelist[0]:filelist[-1],'QA Cost'],marker='o',linestyle='--',label='Baseline Learning Agent')
	plt.xlim(8,40)
	matplotlib.pyplot.xticks([10,17,26,37], fontsize = font_size)
	plt.tick_params(labelsize=font_size)
	plt.ylabel('QA Cost', fontsize = font_size)
	plt.xlabel('KB Size', fontsize = font_size)

	plt.subplot(132)
	plt.plot([10,17,26,37],df1.loc[filelist[0]:filelist[-1],'Dialog Reward'],marker='*',linestyle='-',label='Dual-track POMDP Manager')
	plt.plot([10,17,26,37],df2.loc[filelist[0]:filelist[-1],'Dialog Reward'],marker='o',linestyle='--',label='Baseline Learning Agent')
	plt.xlim(8,40)
	matplotlib.pyplot.xticks([10,17,26,37], fontsize = font_size)	
	plt.tick_params(labelsize=font_size)
	plt.ylabel('Dialog Reward', fontsize = font_size)
	plt.xlabel('KB Size', fontsize = font_size)
	plt.ylim(-30, 30)
	



        #plt.subplot(133)
	#plt.plot([10,17,26,37],df1.loc[filelist[0]:filelist[-1],'Overall Success'],marker='*',linestyle='-',label='Dual-track POMDP Manager')
	#plt.plot([10,17,26,37],df2.loc[filelist[0]:filelist[-1],'Overall Success'],marker='o',linestyle='--',label='Baseline Learning Agent')
	#plt.xlim(8,40)
	#matplotlib.pyplot.xticks([10,17,26,37], fontsize = font_size)
	#plt.tick_params(labelsize=font_size)
	#plt.ylabel('Overall Success', fontsize = font_size)
        #plt.xlabel('KB Size', fontsize = font_size)



	plt.subplot(133)
	plt.plot([10,17,26,37],df1.loc[filelist[0]:filelist[-1],'KB Augmentation Point'],marker='*',linestyle='-',label='Dual-track POMDP Manager')
	plt.plot([10,17,26,37],df2.loc[filelist[0]:filelist[-1],'KB Augmentation Point'],marker='o',linestyle='--',label='Baseline Learning Agent')
	plt.xlim(8,40)
	matplotlib.pyplot.xticks([10,17,26,37], fontsize = font_size)
	plt.tick_params(labelsize=font_size)
	plt.ylabel('Augmentation Point', fontsize = font_size)
	plt.xlabel('KB Size', fontsize = font_size)

	'''
	plt.subplot(234)
	plt.plot(range(3,3+len(filelist)),df1.loc[filelist[0]:filelist[-1],'Precision'],marker='*',linestyle='-',label='Dual-track POMDPs')
	plt.plot(range(3,3+len(filelist)),df2.loc[filelist[0]:filelist[-1],'Precision'],marker='o',linestyle='--',label='Baseline')
	plt.xlim(2.5,6.5)
	matplotlib.pyplot.xticks([3,4,5,6])
	plt.ylim(0,1)
	plt.ylabel('Precision')
	plt.xlabel('Knowledge size')

	plt.subplot(235)
	plt.plot(range(3,3+len(filelist)),df1.loc[filelist[0]:filelist[-1],'Recall'],marker='*',linestyle='-',label='Dual-track POMDPs')
	plt.plot(range(3,3+len(filelist)),df2.loc[filelist[0]:filelist[-1],'Recall'],marker='o',linestyle='--',label='Baseline')
	matplotlib.pyplot.xticks([3,4,5,6])
	plt.xlim(2.5,6.5)
	plt.ylim(0,1)
	plt.ylabel('Recall')
	plt.xlabel('Knowledge size')
	'''
	plt.legend(loc='upper center', bbox_to_anchor=(-1, 1.2),  shadow=True, ncol=2, fontsize=font_size)
	#g.tight_layout()
	plt.show()
	g.savefig('Plots/all_'+str(num)+'_trials_domain_experiment_entropy_5_belief_0_45_N_1')


def main():


	num=500                                  #number of trials
	filelist=['133','144','155','166']                     #list of pomdp files
	#filelist=['144']
	#entlist=[5,5,5,5]
	belieflist=[0.35,0.20,0.18,0.15]
	i=0
	#filelist = ['133', '144']
	df1=pd.DataFrame()
	df2=pd.DataFrame() 
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
			belief_threshold = 0.45, 
			ent_threshold = int(round(((int(name[2]) * int(name[2])) + 1) * 0.1) + 3)         ) # (KB size * 0.1) + 3
	 
		if not s.uniform_init_belief:   
			print('note that initial belief is not uniform\n')
		s.read_model_plus()

		base = Baseline(uniform_init_belief = True, 
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
			belief_threshold = 0.45,
			ent_threshold = 999)
	 
		if not base.uniform_init_belief:   
			print('note that initial belief is not uniform\n')
		base.read_model_plus()
		###Saving results in a dataframe and passing data frame to plot generate_function

		#Put i or name or whatever the name of the iterator is, below in df.at[i, e.g. "Overall Cost"]
		a,b,c,p,r,f, ap=s.run_numbers_of_trials() #f to solve too many values to unpack err. (Best N from experiments)
		ab,bb,cb,pb,rb,fb, apb= base.run_numbers_of_trials()
		
		df1.at[iterator,'QA Cost']= a
		df1.at[iterator,'Overall Success']= b
		df1.at[iterator,'Dialog Reward']= c
		df1.at[iterator,'Precision']= p
		df1.at[iterator,'Recall']= r
                df1.at[iterator,'F1 Score'] = f
                df1.at[iterator,'KB Augmentation Point']= ap
                

		df2.at[iterator,'QA Cost']= ab
		df2.at[iterator,'Overall Success']= bb
		df2.at[iterator,'Dialog Reward']= cb
		df2.at[iterator,'Precision']= pb
		df2.at[iterator,'Recall']= rb
                df2.at[iterator,'F1 Score'] = fb
                df2.at[iterator,'KB Augmentation Point']= apb

	df1.to_csv("Plots_data/all_"+str(num)+"_trials_domain_experiment_entropy_5_belief_0.45_pomdpdual.csv", encoding='utf-8', index=True)
	df2.to_csv("Plots_data/all_"+str(num)+"_trials_domain_experiment_entropy_5_belief_0.45_baseline.csv", encoding='utf-8', index=True)
	df1.sort_values(by=['QA Cost'])
	df2.sort_values(by=['QA Cost'])
	print ('Dual Track POMDP results')
	print (df1)
	print ('Baseline Results')
	print (df2)
	plotgenerate(df1,df2,filelist,num)
        #hypo2(df1,df2,filelist,num)
        #plotgenerateF1Base(df2,filelist,num)
	i+=1

if __name__ == '__main__':
	'''
	num=500
	df1 = pd.read_csv("Plots_data/all_"+str(num)+"_trials_domain_experiment_entropy_2_belief_0.7_pomdpdual.csv", index_col=0)
	print df1
	df2 = pd.read_csv("Plots_data/all_"+str(num)+"_trials_domain_experiment_entropy_2_belief_0.7_baseline.csv", index_col=0)
	print df2
	filelist=['133','144','155','166']
	plotgenerate(df1,df2,filelist,num)
	'''
	
	main()
