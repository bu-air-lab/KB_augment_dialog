from simulator_noparser import Simulator
import pandas as pd
import ast
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
def plotgenerate(df,belieflist,num):
    ######################################### Uncomment: Plot all 5 in one figure ###############################################################
    '''
    fig=plt.figure(figsize=(3*len(list(df)),5))
    
    for count, metric in enumerate(list(df)):
        ax=plt.subplot(1,len(list(df)),count+1)

        #l1 = plt.plot(range(3,3+len(filelist)),df.loc[filelist[0]:filelist[-1],metric],marker='*',linestyle='-',label='Average of '+str(num)+ ' trials')
        l1 = plt.plot(belieflist, df.loc[belieflist[0]:belieflist[-1],metric],marker='*',linestyle='-',label='Average of '+str(num)+ ' trials')
        plt.ylabel(metric)
        plt.xlim(1,9)
        xleft,xright = ax.get_xlim()
        ybottom,ytop = ax.get_ylim()
        ax.set_aspect(aspect=abs((xright-xleft)/(ybottom-ytop)), adjustable=None, anchor=None)
        plt.xlabel('Entropy Changes')


    #ax.legend(loc='upper left', bbox_to_anchor=(-2.10, 1.35),  shadow=True, ncol=5)
    fig.tight_layout()
    plt.show()
    fig.savefig('Results_'+str(num)+'_trials')
    
    #################################################### Split 5 plots to 2 figures: (cost,success, reward) and (precision, recall) ##################################
    #######################    Figure 1      ########################
    f=plt.figure(1, figsize=(3*len(list(df)),6))
    plt.suptitle('Increasing entropy changes for fixed model 133 , belief threshold 0.7', fontsize=14); 
    for count,metric in enumerate(list(df)):
        if count<3: 
            ax=plt.subplot(1,len(list(df))-2,count+1)

            l1 = plt.plot(range(2,max(belieflist)+1),df.loc[belieflist[0]:belieflist[-1],metric],marker='*',linestyle='-',label='Average of '+str(num)+ ' trials')
            #l1 = plt.plot(range(filelist[0],filelist[0]+len(filelist)),df.loc[filelist[0]:filelist[-1],metric],marker='*',linestyle='-',label='Average of '+str(num)+ ' trials')
            plt.ylabel(metric)
            if metric=='Overall Success':
                plt.ylim(0,1)
            elif metric=='Overall Reward':
                plt.ylim(0,30)
            elif metric=='Overall Cost':
                plt.ylim(-30,0)  
            plt.xlim(1.5,max(belieflist)+1)
            xleft , xright =ax.get_xlim()
            ybottom , ytop = ax.get_ylim()
            ax.set_aspect(aspect=abs((xright-xleft)/(ybottom-ytop)), adjustable=None, anchor=None)


            plt.xlabel('Number of entropy changes')
    

  
    #ax.legend(loc='upper left', bbox_to_anchor=(-2.10, 1.35),  shadow=True, ncol=5)
    #g.tight_layout()
    plt.show()
    g.savefig('Plots/precision_recall_'+str(num)+'_trials_domain_experiment_entropy_2_belief_07')

    #######################    Figure 2      ########################
    g=plt.figure(2, figsize=(3*len(list(df)),6))
    plt.suptitle('Increasing entropy changes for fixed model=133, belief threshold 0.7', fontsize=14); 
    for count,metric in enumerate(['Precision','Recall']): 
        ax=plt.subplot(1,2,count+1)

        l1 = plt.plot(range(2,max(belieflist)+1),df.loc[belieflist[0]:belieflist[-1],metric],marker='*',linestyle='-',label='Average of '+str(num)+ ' trials')
        #l1 = plt.plot(range(filelist[0],filelist[0]+len(filelist)),df.loc[filelist[0]:filelist[-1],metric],marker='*',linestyle='-',label='Average of '+str(num)+ ' trials')
        plt.ylabel(metric)
        plt.xlim(1.5,max(belieflist)+1)
        plt.ylim(0,1)
        xleft , xright =ax.get_xlim()
        ybottom , ytop = ax.get_ylim()
        ax.set_aspect(aspect=abs((xright-xleft)/(ybottom-ytop)), adjustable=None, anchor=None)


        plt.xlabel('Number of Entropy changes')
    '''
    g=plt.figure(figsize=(12,3))
    #plt.suptitle('Increasing entropy changes for fixed model 133 , belief threshold 0.7,'+str(num)+ ' trials', fontsize=18);
    plt.subplot(141)
    plt.plot(range(2,max(belieflist)+1),df.loc[belieflist[0]:belieflist[-1],'QA Cost'],marker='o',linestyle='-',label='Average of '+str(num)+ ' trials')
    plt.xlim(1.5,max(belieflist)+1)
    plt.ylim(-25,-14)
    plt.ylabel('QA Cost')
    plt.xlabel('Number of EFs')

    plt.subplot(142)
    plt.plot(range(2,max(belieflist)+1),df.loc[belieflist[0]:belieflist[-1],'Overall Success'],marker='o',linestyle='-',label='Average of '+str(num)+ ' trials')
    plt.xlim(1.5,max(belieflist)+1)
    plt.ylabel('Overall Success')
    plt.xlabel('Number of EFs')

    plt.subplot(143)
    plt.plot(range(2,max(belieflist)+1),df.loc[belieflist[0]:belieflist[-1],'Dialog Reward'],marker='o',linestyle='-',label='Average of '+str(num)+ ' trials')
    plt.xlim(1.5,max(belieflist)+1)
    #plt.ylim(0,30)
    plt.ylabel('Dialog Reward')
    plt.xlabel('Number of EFs')

    plt.subplot(144)
    plt.plot(range(2,max(belieflist)+1),df.loc[belieflist[0]:belieflist[-1],'F1 Score'],marker='o',linestyle='-',label='F1 Score')
    plt.plot(range(2,max(belieflist)+1),df.loc[belieflist[0]:belieflist[-1],'Precision'],marker='^',linestyle='--',label='Precision')
    plt.plot(range(2,max(belieflist)+1),df.loc[belieflist[0]:belieflist[-1],'Recall'],marker='+',linestyle=':',label='Recall')
    plt.xlim(1.5,max(belieflist)+1)
    plt.ylim(0,1)
    plt.ylabel('F1 Score, Precision, Recall')
    plt.xlabel('Number of EFs')
    plt.legend(loc=0)

    #ax.legend(loc='upper left', bbox_to_anchor=(-2.10, 1.35),  shadow=True, ncol=5)
    #g.tight_layout()
    plt.show()
    g.savefig('Plots/all_'+str(num)+'_trials_entropy_experiment_model_133')

def main():
    num=500                                        #number of trials
    entropylist=[2,3,4,5,6,7,8]
    #filelist = ['133', '144']
    df=pd.DataFrame() 
    # just use for sth in somelist, not for sth in range(len(ssomelist))
    for iterator in entropylist:
        name = '133'  # or name = iterator

        s = Simulator(uniform_init_belief = True, 
            auto_state = True, 
            auto_observations = True, # was true
            print_flag = False, 
            policy_file = name+'_new.policy',
            pomdp_file =  name +'_new.pomdp',
                pomdp_file_plus=list(name)[0]+str(int(list(name)[1])+1)+str(int(list(name)[2])+1)+'_new.pomdp',
                policy_file_plus=list(name)[0]+str(int(list(name)[1])+1)+str(int(list(name)[2])+1)+'_new.policy',
            trials_num = num,
            num_task = int(name[0]), 
            num_patient = int(name[1]), 
            num_recipient = int(name[2]),
            belief_threshold = 0.7,
            ent_threshold = iterator)
     
        if not s.uniform_init_belief:   
            print('note that initial belief is not uniform\n')
        s.read_model_plus()
        ###Saving results in a dataframe and passing data frame to plot generate_function

        #Put i or name or whatever the name of the iterator is, below in df.at[i, e.g. "Overall Cost"]
        a,b,c,p,r,f=s.run_numbers_of_trials()
        df.at[iterator,'QA Cost']= a
        df.at[iterator,'Overall Success']= b
        df.at[iterator,'Dialog Reward']= c
        df.at[iterator,'Precision']= p
        df.at[iterator,'Recall']= r
        df.at[iterator,'F1 Score']= f
    print df
    df.to_csv("Plots_data/all"+str(num)+"_trials_entropy_experiment_model_133.csv", encoding='utf-8', index=True)
    plotgenerate(df,entropylist,num)

if __name__ == '__main__':
    main()