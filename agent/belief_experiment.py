from simulator_noparser import Simulator
import pandas as pd
import ast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plotgenerate(df,belieflist,entropylist,num):
    '''
    fig=plt.figure(figsize=(3*len(list(df)),5))
    
    for count, metric in enumerate(list(df)):
        ax=plt.subplot(1,len(list(df)),count+1)

        #l1 = plt.plot(range(3,3+len(filelist)),df.loc[filelist[0]:filelist[-1],metric],marker='*',linestyle='-',label='Average of '+str(num)+ ' trials')
        l1 = plt.plot(belieflist, df.loc[belieflist[0]:belieflist[-1],metric],marker='*',linestyle='-',label='Average of '+str(num)+ ' trials')
        plt.ylabel(metric)
        plt.xlim(0,1)
        xleft,xright = ax.get_xlim()
        ybottom,ytop = ax.get_ylim()
        ax.set_aspect(aspect=abs((xright-xleft)/(ybottom-ytop)), adjustable=None, anchor=None)
        plt.xlabel('Belief Threshold')
    
    #ax.legend(loc='upper left', bbox_to_anchor=(-2.10, 1.35),  shadow=True, ncol=5)
    fig.tight_layout()
    plt.show()
    fig.savefig('Results_'+str(num)+'_trials')
    '''
    #g=plt.figure(figsize=(15,9))
    #plt.suptitle('Increasing belief for fixed model 133 , fixed entropy changes =9, '+str(num)+ ' trials', fontsize=18);
    #plt.subplot(231)
    df.plot(x='Overall Cost', y = ['Belief', 'Entropy'])
    plt.show()
'''
    #plt.xlim(0,1)
    #plt.ylim(-30,0)
    #plt.ylabel('Overall Cost')
    #plt.xlabel('Belief threshold')

    plt.subplot(232)
    plt.plot([0.3,0.4,0.5,0.6,0.7],df.loc[belieflist[0]:belieflist[-1],'Overall Success'],marker='*',linestyle='-',label='Average of '+str(num)+ ' trials')
    plt.xlim(0,1)
    plt.ylabel('Overall Success')
    plt.xlabel('Belief threshold')

    plt.subplot(233)
    plt.plot([0.3,0.4,0.5,0.6,0.7],df.loc[belieflist[0]:belieflist[-1],'Overall Reward'],marker='*',linestyle='-',label='Average of '+str(num)+ ' trials')
    plt.xlim(0,1)
    #plt.ylim(0,30)
    plt.ylabel('Overall Reward')
    plt.xlabel('Belief threshold')

    plt.subplot(234)
    plt.plot([0.3,0.4,0.5,0.6,0.7],df.loc[belieflist[0]:belieflist[-1],'Precision'],marker='*',linestyle='-',label='Average of '+str(num)+ ' trials')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.ylabel('Precision')
    plt.xlabel('Belief threshold')

    plt.subplot(235)
    plt.plot([0.3,0.4,0.5,0.6,0.7],df.loc[belieflist[0]:belieflist[-1],'Recall'],marker='*',linestyle='-',label='Average of '+str(num)+ ' trials')
    plt.xlim(0,1)
    plt.ylim(0,1.5)
    plt.ylabel('Recall')
    plt.xlabel('Belief threshold')
    #ax.legend(loc='upper left', bbox_to_anchor=(-2.10, 1.35),  shadow=True, ncol=5)
    #g.tight_layout()
    plt.show()
    g.savefig('Plots/all_'+str(num)+'_trials_belief_experiment_model_133_ent_9')
'''



def main():
    num=50                                        #number of trials
    belieflist=[0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    entropylist=[2,3,4,5,6,7,8]
    #filelist = ['133', '144']
    df=pd.DataFrame(index = range(1,50)) 
    # just use for sth in smelist, not for sth in range(len(ssomelist))
    i = 1
    for iterator in belieflist:
        for iterator2 in entropylist:
            name = '133'  # or name = iterator

            s = Simulator(uniform_init_belief = True, 
                auto_state = True, 
                auto_observations = True, # was true
                print_flag = True,
                policy_file = name+'.policy',
                pomdp_file =  name +'.pomdp',
                    pomdp_file_plus=list(name)[0]+str(int(list(name)[1])+1)+str(int(list(name)[2])+1)+'.pomdp',
                    policy_file_plus=list(name)[0]+str(int(list(name)[1])+1)+str(int(list(name)[2])+1)+'.policy',
                trials_num = num,
                num_task = int(name[0]), 
                num_patient = int(name[1]), 
                num_recipient = int(name[2]),
                belief_threshold = iterator,
                ent_threshold = iterator2)
     
            if not s.uniform_init_belief:   
                print('note that initial belief is not uniform\n')
            s.read_model_plus()
        ###Saving results in a dataframe and passing data frame to plot generate_function

        #Put i or name or whatever the name of the iterator is, below in df.at[i, e.g. "Overall Cost"]
            a,b,c,p,r,f=s.run_numbers_of_trials()
            df.at[i, 'Belief'] = iterator
            df.at[i, 'Entropy'] = iterator2
            df.at[i, 'Overall Cost']= a
            df.at[i, 'Overall Success']= b
            df.at[i, 'Overall Reward']= c
            df.at[i, 'Precision']= p
            df.at[i, 'Recall']= r
            i += 1
    df.columns = ['Belief', 'Entropy', 'Overall Cost', 'Overall Success', 'Overall Reward', 'Precision', 'Recall']
    print df
    df.to_csv("Plots_data/all"+str(num)+"_trials_belief_experiment_model_133_ent_9.csv", encoding='utf-8', index=True)
    plotgenerate(df,belieflist,entropylist,num)

if __name__ == '__main__':
    main()
