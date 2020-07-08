#!/usr/bin/env python3
import os,sys

def findBestParameters(mlc_dictionary,mlc_name,df_data,labels,textfile_name,output_parameters):
        # -*- coding: utf-8 -*-
    """
    Created on Tue Sep 12 22:11:38 2017

    @author: lindseykitchell

    Updated on Mon Jun 29 18:09:00 2020

    @updated: Brad Caron (bacaron@iu.edu)
    """
    import gc
    import numpy as np
    import pandas as pd
    import json
    from sklearn.model_selection import KFold
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler

    # print out and write name of mlc
    print('Testing '+mlc_name+'\n',file=open(textfile_name,"a"))

    # setting up leave-one-out structure
    loo = LeaveOneOut()
    kf = loo.split(df_data)

    # grab model and parameters from dictionary and print/write parameters
    svr = mlc_dictionary[mlc_name]['model']
    parameters = mlc_dictionary[mlc_name]['parameters']
    print(parameters,file=open(textfile_name,"a"))
    
    # run gridsearch to identify best parameters and print to textfile
    clf = GridSearchCV(svr, parameters, cv=kf, n_jobs=-1)
    clf.fit(df_data, labels)
    print("\nBest Parameters for %s\n" %mlc_name,file=open(textfile_name,"a"))
    print("'%s': %s" %(mlc_name,clf.best_params_),file=open(textfile_name,"a"))
    print("\nBest Score\n",file=open(textfile_name,"a"))
    print(clf.best_score_,file=open(textfile_name,"a"))

    output_parameters[mlc_name] = {'name': mlc_name, 'parameters': clf.best_params_,'score': clf.best_score_}

    return output_parameters

def gridsearch_algs(node_type,df_nodes,df_subjects,measures,measures_name,labels,mlc_dictionary,text_dir,data_dir):
    # -*- coding: utf-8 -*-
    """
    Created on Tue Sep 12 22:11:38 2017

    @author: lindseykitchell

    Updated on Mon Jun 29 18:09:00 2020

    @updated: Brad Caron (bacaron@iu.edu)
    """
    import gc
    import numpy as np
    import pandas as pd
    import json
    from sklearn.model_selection import KFold
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler

    # set up output parameters structures and datastrcture
    output_parameters = {}

    # merge data structure
    df = pd.merge(df_nodes,df_subjects,on="subjectID")
    
    # reshape data
    data = df[measures].values.reshape((df_subjects.shape[0],-1))

    # standardize data with StandardScaler()
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    
    ## open textfile titled 'gridsearchresults.txt'
    txtfile = open(text_dir+'/'+node_type+'_'+measures_name+'_gridsearchresults.txt',"w")
    txtfile.close()
    
    ## loop through mlcs
    for m in mlc_dictionary.keys():
        output_parameters = findBestParameters(mlc_dictionary,m,data,labels,txtfile.name,output_parameters)

    # output parameter and score dictionaries
    with open(data_dir+'/'+node_type+'_'+measures_name+'_parameters.json',"w") as param_f:
        json.dump(output_parameters,param_f)

    return output_parameters

def runModel(df_nodes,df_subjects,mlc,mlc_name,model_labels,model_name,measures,iters,out_dir,out_name,output):
    import gc
    import numpy as np
    import pandas as pd
    import json
    from sklearn.model_selection import KFold
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler
    from scipy.io import savemat
    from sklearn.model_selection import cross_val_score

    # dummy variables
    results = []
    perc = []

    # generate output dataframe
    output_df = pd.DataFrame([],columns={'iterations','mlc','model','percentages'})

    # merge data structure
    df = pd.merge(df_nodes,df_subjects,on="subjectID")
    
    # reshape data
    data = df[measures].values.reshape((df_subjects.shape[0],-1))

    # standardize data with StandardScaler()
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)

    for x in range(int(iters)):
        ## leave-one-out train/test split
        loo = LeaveOneOut()

        kf = loo.split(data)

        ## accuracy performance of model
        cv_results = cross_val_score(mlc, data, model_labels, cv=kf, scoring='accuracy')

        ## append cross-validated results to results array
        results.append(cv_results)

        ## generate percentage accuracy
        perc.append((np.mean(cv_results)))

    # populate output dataframe
    output_df['percentages'] = perc
    output_df['mlc'] = [ mlc_name for f in range(len(output_df['percentages'])) ]
    output_df['model'] = [ model_name for f in range(len(output_df['percentages'])) ]
    output_df['iterations'] = [ f+1 for f in range(len(output_df['percentages'])) ]
    output_df = output_df.reindex(columns=['iterations','mlc','model','percentages'])

    # tissue type output
    output = pd.concat([output,output_df],sort=False)

    # save results
    output_df.to_csv(out_dir+out_name+'.csv',index=False)

    return output

### calculate rac and bic
def computeRacBic(data,models,mlcs,len_subjects):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from statsmodels.tools.eval_measures import bic

    # set up dummy variables
    rac = []
    BIC = []

    # setup output summary dataframe
    outputData = pd.DataFrame([])

    # calculate rac and append to input data structure
    for perc in range(len(data['percentages'])):
        if data['model'][perc] == models[0]:
            chance = 1/3
        else:
            chance = 1/2
        rac.append((data['percentages'][perc] - chance) / (1 - chance))
    data['rac'] = rac

    # append data to output summary dataframe
    for model in models:
            tmpData = pd.DataFrame([])
            tmpData['mlc'] = mlcs
            tmpData['model'] = [ model for f in range(len(tmpData['mlc'])) ]
            tmpData['medianRac'] = list(data[data['model']==model].groupby('mlc',sort=False).median()['rac'])
            tmpData['meanAcc'] = list(data[data['model']==model].groupby('mlc',sort=False).mean()['percentages'])
            tmpData['logLikelihoodRac'] = list(np.log10(data[data['model']==model].groupby('mlc',sort=False).mean()['rac']))
            outputData = pd.concat([outputData,tmpData],sort=False,ignore_index=True)  

    # compute bic and append to output summary dataframe
    for mlc in range(len(outputData['logLikelihoodRac'])):
        if outputData['mlc'][mlc] in [mlcs[0],mlcs[5]]:
            numParams = 4
        elif outputData['mlc'][mlc] == mlcs[1]:
            numParams = 1
        else:
            numParams = 5

        BIC.append(bic(outputData['logLikelihoodRac'][mlc],len_subjects,numParams))

    outputData['bic'] = BIC

    return [data,outputData]

### plot data
def plotModelPerformance(x_measure,y_measure,data,dir_out,out_name):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        # set up output names
        img_out=out_name+'.eps'
        img_out_png=out_name+'.png'

        # generate figures
        fig = plt.figure(figsize=(15,15))
        fig.patch.set_visible(False)
        p = plt.subplot()

        # set spines and ticks
        p.spines['right'].set_visible(False)
        p.spines['top'].set_visible(False)
        p.yaxis.set_ticks_position('left')
        p.xaxis.set_ticks_position('bottom')

        # plot data
        ax = sns.violinplot(x=x_measure,y=y_measure,data=data)

        ax.set(ylim=(0,1.2))
        ax.set(yticks=np.arange(0,1.1,step=0.1))

        # save or show plot
        if dir_out:
            if not os.path.exists(dir_out):
                os.mkdir(dir_out)

            plt.savefig(os.path.join(dir_out, img_out))
            plt.savefig(os.path.join(dir_out, img_out_png))       
        else:
            plt.show()

def plotMlcModelPerformance(x_measure,y_measure,data,kind,dir_out,out_name):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # set up output names
    img_out=out_name+'.eps'
    img_out_png=out_name+'.png'

    # generate figures
    fig = plt.figure(figsize=(15,15))
    fig.patch.set_visible(False)
    p = plt.subplot()

    # set spines and ticks
    p.spines['right'].set_visible(False)
    p.spines['top'].set_visible(False)
    p.yaxis.set_ticks_position('left')
    p.xaxis.set_ticks_position('bottom')

    # plot data
    ax = sns.catplot(x=x_measure,y=y_measure,col="model",data=data,kind=kind)

    if kind == 'bar':
        ax.set(ylim=(0,28))
        ax.set(yticks=np.arange(0,30,step=2))
    else:
        ax.set(ylim=(0,1.2))
        ax.set(yticks=np.arange(0,1.1,step=0.1))

 
    # save or show plot
    if dir_out:
        if not os.path.exists(dir_out):
            os.mkdir(dir_out)

        plt.savefig(os.path.join(dir_out, img_out))
        plt.savefig(os.path.join(dir_out, img_out_png))       
    else:
        plt.show()
