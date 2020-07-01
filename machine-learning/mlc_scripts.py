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

# def tissueTypeSubsetAnalyses(node_type,):
#     import sys
#     import numpy as np
#     import pandas as pd
#     from Model_LOO_subset import Model_LOO_subset
#     from scipy.io import savemat

#         ## read in tract profile data (nodes.csv) and subject identification data (subjects.csv) and merge into single structure
#     df_nodes = pd.read_csv('nodes.csv')
#     df_subjects = pd.read_csv('subjects.csv')
#     df = pd.merge(df_nodes,df_subjects,on="subjectID")

#     ## Associative tracts
#         ## list of associative tract names
#     df_tract = df[(df['tractID'] == "Left pArc") | (df['tractID'] == "Right pArc") | (df['tractID'] == "Left VOF") | (df['tractID'] == "Right VOF") | (df['tractID'] == "Left IFOF") | (df['tractID'] == "Right IFOF") | (df['tractID'] == "Left ILF") | (df['tractID'] == "Right ILF") | (df['tractID'] == "Left Uncinate") | (df['tractID'] == "Right Uncinate") | (df['tractID'] == "Left anterior thalamic") | (df['tractID'] == "Right anterior thalamic") | (df['tractID'] == "Left inferior thalamic") | (df['tractID'] == "Right inferior thalamic") | (df['tractID'] == "Left superior thalamic") | (df['tractID'] == "Right superior thalamic")]
    
#         ## grab tract profile data from associative tracts
#         df_measure = df_tract[['icvf','od','isovf']].values.reshape((df_subjects.shape[0],-1))

#         ## run classification analysis across iterations on associative tracts
#     [model,results,df_measure,y,perc] = Model_LOO_subset(df_measure,"Associative",model,model_name,iters)
    
#         ## save results from associative tracts
#         savemat('results_%s_associative' %model_name,mdict={'tot_perc': perc})

#     ## Projection tracts
#         ## list of projection tract names
#     df_tract = df[(df['tractID'] == "Left cortico-spinal") | (df['tractID'] == "Right cortico-spinal") | (df['tractID'] == "Left IpsiFrontoPontine") | (df['tractID'] == "Right IpsiFrontoPontine") | (df['tractID'] == "Left IpsiCorticoPontine") | (df['tractID'] == "Right IpsiCorticoPontine") | (df['tractID'] == "Left Thal2ceb") | (df['tractID'] == "Right Thal2ceb") | (df['tractID'] == "Left Thalamico-spinal") | (df['tractID'] == "Right Thalamico-spinal")]
    
#         ## grab tract profile data from projection tracts
#         df_measure = df_tract[['icvf','od','isovf']].values.reshape((df_subjects.shape[0],-1))
        
#         ## run classification analysis across iterations on projection tracts
#     [model,results,df_measure,y,perc] = Model_LOO_subset(df_measure,"Projection",model,model_name,iters)
    
#         ## save results from projection tracts
#         savemat('results_%s_projection' %model_name,mdict={'tot_perc': perc})

#     ## Callosal tracts
#         ## list of callosal tract names
#     df_tract = df[(df['tractID'] == "Corpus Callosum") | (df['tractID'] == "Forceps Major") | (df['tractID'] == "Forceps Minor")]
    
#         ## grab tract profile data from callosal tracts
#         df_measure = df_tract[['icvf','od','isovf']].values.reshape((df_subjects.shape[0],-1))
    
#         ## run classification analysis across iterations on callosal tracts
#         [model,results,df_measure,y,perc] = Model_LOO_subset(df_measure,"Callosal",model,model_name,iters)
    
#         ## save results from callosal tracts
#         savemat('results_%s_callosal' %model_name,mdict={'tot_perc': perc})

