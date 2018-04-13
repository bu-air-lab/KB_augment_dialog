from simulator_noparser import Simulator
import pandas as pd
import ast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plotgenerate(df,belieflist,num):
    fig=plt.figure(figsize=(3*len(list(df)),5))
    
    for count, metric in enumerate(list(df)):
        ax=plt.subplot(1,len(list(df)),count+1)

        #l1 = plt.plot(range(3,3+len(filelist)),df.loc[filelist[0]:filelist[-1],metric],marker='*',linestyle='-',label='Average of '+str(num)+ ' trials')
        l1 = plt.plot(belieflist, df.loc[belieflist[0]:belieflist[-1],metric],marker='*',linestyle='-',label='Average of '+str(num)+ ' trials')
        plt.ylabel(metric)
        plt.xlim(belieflist[0]-0.5,belieflist[0]+len(belieflist)-0.5)
        xleft,xright = ax.get_xlim()
        ybottom,ytop = ax.get_ylim()
        ax.set_aspect(aspect=abs((xright-xleft)/(ybottom-ytop)), adjustable=None, anchor=None)
        plt.xlabel('Belief Threshold')


    #ax.legend(loc='upper left', bbox_to_anchor=(-2.10, 1.35),  shadow=True, ncol=5)
    fig.tight_layout()
    plt.show()
    fig.savefig('Results_'+str(num)+'_trials')


def main():
    num=500                                        #number of trials
    belieflist=[0.3,0.4,0.5,0.6,0.7]
    #filelist = ['133', '144']
    df=pd.DataFrame() 
    # just use for sth in somelist, not for sth in range(len(ssomelist))
    for iterator in belieflist:
        name = '133'  # or name = iterator

        s = Simulator(uniform_init_belief = True, 
            auto_state = True, 
            auto_observations = True, # was true
            print_flag = True, 
            use_plog = False,
            policy_file = name+'_new.policy',
            pomdp_file =  name +'_new.pomdp',
                pomdp_file_plus=list(name)[0]+str(int(list(name)[1])+1)+str(int(list(name)[2])+1)+'_new.pomdp',
                policy_file_plus=list(name)[0]+str(int(list(name)[1])+1)+str(int(list(name)[2])+1)+'_new.policy',
            trials_num = num,
            num_task = int(name[0]), 
            num_patient = int(name[1]), 
            num_recipient = int(name[2]),
            belief_threshold = iterator,
            ent_threshold = 2)
     
        if not s.uniform_init_belief:   
            print('note that initial belief is not uniform\n')
        s.read_model_plus()
        ###Saving results in a dataframe and passing data frame to plot generate_function

        #Put i or name or whatever the name of the iterator is, below in df.at[i, e.g. "Overall Cost"]
        a,b,c,p,r=s.run_numbers_of_trials()
        df.at[iterator,'Overall Cost']= a
        df.at[iterator,'Overall Success']= b
        df.at[iterator,'Overall Reward']= c
        df.at[iterator,'Precision']= p
        df.at[iterator,'Recall']= r
    print df
    plotgenerate(df,belieflist,num)

if __name__ == '__main__':
    main()