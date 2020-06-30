# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 22:11:38 2017

@author: lindseykitchell

Updated on Mon Jun 29 18:09:00 2020

@updated: Brad Caron (bacaron@iu.edu)
"""
import os,sys

def findBestParameters(mlc_dictionary,mlc_name,df_data,textfile_name):

    # open textfile for writing
    #textfile = open(textfile_name,"a")

    # print out and write name of mlc
    print('Testing '+mlc_name+'\n',file=open(textfile_name,"a"))
    #textfile.write('Testing '+mlc_name+'\n')

    # setting up leave-one-out structure
    loo = LeaveOneOut()
    kf = loo.split(df_data)

    # grab model and parameters from dictionary and print/write parameters
    svr = mlc_dictionary[mlc_name]['model']
    parameters = mlc_dictionary[mlc_name]['parameters']

    print(parameters,file=open(textfile_name,"a"))
    #textfile.write(str(parameters))
    
    # run gridsearch to identify best parameters and print to textfile
    clf = GridSearchCV(svr, parameters, cv=kf, n_jobs=-1)
    clf.fit(data, labels)
    print("\nBest Parameters for %s\n" %mlc_name,file=open(textfile_name,"a"))
    #textfile.write("\nBest Parameters for %s\n" %mlc_name)    
    print(clf.best_params_,file=open(textfile_name,"a"))
    #textfile.write(str(clf.best_params_))
    print("\nBest Score\n",file=open(textfile_name,"a"))
    #textfile.write("\nBest Score\n")
    print(clf.best_score_,file=open(textfile_name,"a"))
    #textfile.write(str(clf.best_score_))

    
    #textfile.close()

def gridsearch_algs(filepath,node_type,diffusion_measures,out_dir):
    import numpy as np
    import pandas as pd
    import gc
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


    ## set up mlcs label array
    mlcs = ['Random Forest Classifier','Adaboost Classifier','Support Vector Classifier','KNeighbors Classifier','Decision Tree Classifier','Logistic Regression']

    ## set up dictionary
    mlc_dict = {}
    # RFC
    mlc_dict[mlcs[0]] = {'model': RandomForestClassifier(),'name': mlcs[0],'parameters': {'n_estimators':[50,100, 150], 'criterion': ('gini', 'entropy'), 
        'max_features': ('auto', 'log2', None), 'min_samples_leaf':[1,5,10,50]}}
    # Ada
    mlc_dict[mlcs[1]] = {'model': AdaBoostClassifier(),'name': mlcs[1],'parameters': {'n_estimators':[100, 200, 300]}}
    # SVC
    mlc_dict[mlcs[2]] = {'model': SVC(),'name': mlcs[2],'parameters': {'C':[1,2,3,4,5,10], 'probability':(True,False), 'decision_function_shape':('ovo','ovr',None), 
        'kernel':('linear', 'poly', 'rbf', 'sigmoid'),'degree':[3,4,5]}}
    # KNN
    mlc_dict[mlcs[3]] = {'model': KNeighborsClassifier(),'name': mlcs[3],'parameters': {'n_neighbors':[5,6,7,8,9,10, 11, 12, 13], 'weights':('uniform', 'distance'), 'algorithm':('ball_tree','kd_tree','brute'),
        'leaf_size':[2,3,4,5,6,7,8],'p':[1,2]}}
    # DTC
    mlc_dict[mlcs[4]] = {'model': DecisionTreeClassifier(),'name': mlcs[4],'parameters': {'criterion':('gini', 'entropy'), 'splitter': ('best', 'random'), 'max_features':('auto', 'sqrt', 'log2', None), 'min_samples_leaf':[1,5,7,10,20]}}
    # LR
    mlc_dict[mlcs[5]] = {'model': LogisticRegression(),'name': mlcs[5],'parameters': { 'C':[.2,.3,.5,.6,.7], 'fit_intercept':(True, False),'solver': ('newton-cg', 'lbfgs', 'sag'), 'multi_class':('ovr','multinomial'), 'warm_start':(True,False)}}

    ## read in tract profile (nodes.csv) and subject (subjects.csv) data and merge into single structure
    df_nodes = pd.read_csv(filepath+node_type+'_nodes.csv')
    df_subjects = pd.read_csv(filepath+node_type+'_subjects.csv')
    df = pd.merge(df_nodes,df_subjects,on="subjectID")
    
    # remove tracks with NaNs
    nan_structs = list(df[df[diffusion_measures[0]].isnull()]['structureID'])
    nan_subs = list(df[df[diffusion_measures[0]].isnull()]['subjectID'])
    print("missing structs: \n"+str(df[df[diffusion_measures[0]].isnull()]))
    df = df[~df['structureID'].isin(nan_structs)]
    data = df[diffusion_measures].values.reshape((df_subjects.shape[0],-1))

    ## standardize data with StandardScaler()
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)

    ## generate sport classification array (1: football, 0: cross-country, 2: non-athlete)
    labels = [int(f.split('_')[0]) for f in df_subjects['subjectID']]
    
    ## open textfile titled 'gridsearchresults.txt'
    txtfile = open(out_dir+'gridsearchresults.txt',"w")
    txtfile.close()
    
    ## loop through mlcs
    for m in mlcs:
        findBestParameters(mlc_dict,m,data,txtfile.name)


    # testing RandomForestClassifier
    loo = LeaveOneOut()
    kf = loo.split(data)

    print('Testing Random Forest Classifier\n')
    txtfile.write('Testing Random Forest Classifier\n')
    # first use GridSearchCV to get best parameters
    
    parameters = {'n_estimators':[50,100, 150], 'criterion': ('gini', 'entropy'),
                  'max_features': ('auto', 'log2', None), 'min_samples_leaf':[1,5,10,50]}
    print(parameters)
    txtfile.write(str(parameters))
    
    svr = RandomForestClassifier()
    clf = GridSearchCV(svr, parameters, cv=kf, n_jobs=-1)
    clf.fit(data, labels)
    print("\nBest Parameters for Random Forest Classifier\n")
    txtfile.write("\nBest Parameters for Random Forest Classifier\n")    
    print(clf.best_params_)
    txtfile.write(str(clf.best_params_))
    print("\nBest Score\n")
    txtfile.write("\nBest Score\n")
    print(clf.best_score_)
    txtfile.write(str(clf.best_score_))
    
    # testing AdaBoost
    loo = LeaveOneOut()
    kf = loo.split(data)
    
    print('\n Testing AdaBoost Classifier\n')
    txtfile.write('\n Testing AdaBoost Classifier\n')    
    # first use GridSearchCV to get best parameters
     
    parameters = {'n_estimators':[100, 200, 300]}
    
    print parameters
    txtfile.write(str(parameters))
    
    svr = AdaBoostClassifier()
    clf = GridSearchCV(svr, parameters, cv=kf,)
    clf.fit(data, labels)
    print('\n Best Parameters for AdaBoost Classifier \n')
    txtfile.write('\nBest Parameters for AdaBoost Classifier \n')
    
    print clf.best_params_
    txtfile.write(str(clf.best_params_))
    print "\nBest Score\n"
    txtfile.write("\nBest Score\n")
    print clf.best_score_
    txtfile.write(str(clf.best_score_))
    # testing SVM
    loo = LeaveOneOut()
    kf = loo.split(data)

    print '\n Testing SVM Classifier\n'
    txtfile.write('\n Testing SVM Classifier\n')
    parameters = {'C':[1,2,3,4,5,10], 'probability':(True,False), 'decision_function_shape':('ovo','ovr',None), 'kernel':('linear', 'poly', 'rbf', 'sigmoid'),'degree':[3,4,5]}
    print parameters
    txtfile.write(str(parameters))
    
    svr = SVC()
    clf = GridSearchCV(svr, parameters, cv=kf,)
    clf.fit(data, labels)
    
    print '\n Best Parameters for SVM Classifier \n'
    txtfile.write('\n Best Parameters for SVM Classifier \n')
    print clf.best_params_
    txtfile.write(str(clf.best_params_))
    print "\nBest Score\n"
    txtfile.write("\nBest Score\n")
    print clf.best_score_
    txtfile.write(str(clf.best_score_))
    
    # testing KNN
    loo = LeaveOneOut()
    kf = loo.split(data)
   
    print '\n Testing KNN Classifier\n'
    txtfile.write('\n Testing KNN Classifier\n')

    parameters = {'n_neighbors':[5,6,7,8,9,10, 11, 12, 13], 'weights':('uniform', 'distance'), 'algorithm':('ball_tree','kd_tree','brute'),'leaf_size':[2,3,4,5,6,7,8],'p':[1,2]}
    
    print parameters
    txtfile.write(str(parameters))
    
    svr = KNeighborsClassifier()
    clf = GridSearchCV(svr, parameters, cv=kf, n_jobs=-1)
    clf.fit(data, labels)
    
    print '\n Best Parameters for KNN Classifier \n'
    txtfile.write('\n Best Parameters for KNN Classifier \n')
    print clf.best_params_
    txtfile.write(str(clf.best_params_))
    print "\nBest Score\n"
    txtfile.write("\nBest Score\n")
    print clf.best_score_
    txtfile.write(str(clf.best_score_))
    
    #testing decision tree classifier
    loo = LeaveOneOut()
    kf = loo.split(data)

    print '\n Testing Decision Tree Classifier\n'
    txtfile.write('\n Testing Decision Tree Classifier\n')
     
    parameters = {'criterion':('gini', 'entropy'), 'splitter': ('best', 'random'), 'max_features':('auto', 'sqrt', 'log2', None), 'min_samples_leaf':[1,5,7,10,20]}
    
    print parameters
    txtfile.write(str(parameters))
    svr = DecisionTreeClassifier()
    clf = GridSearchCV(svr, parameters, cv=kf)
    clf.fit(data, labels)
    
    print '\n Best Parameters for Decision Tree Classifier \n'
    txtfile.write('\n Best Parameters for Decision Tree Classifier \n')
    
    print clf.best_params_
    txtfile.write(str(clf.best_params_))
    print "\nBest Score\n"
    txtfile.write("\nBest Score\n")
    print clf.best_score_
    txtfile.write(str(clf.best_score_))

    #testing logistic regression
    print '\n Testing Logistic Regression Classifier\n'
    txtfile.write('\n Testing Logistic Regression Classifier\n')
    parameters = { 'C':[.2,.3,.5,.6,.7], 'fit_intercept':(True, False),'solver': ('newton-cg', 'lbfgs', 'sag'), 'multi_class':('ovr','multinomial'), 'warm_start':(True,False)}
    
    print parameters
    txtfile.write(str(parameters))
    
    svr = LogisticRegression()
    clf = GridSearchCV(svr, parameters, cv=kf)
    clf.fit(data, labels)
    
    print '\n Best Parameters for Logistic Regression Classifier \n'
    txtfile.write('\n Best Parameters for Logistic Regression Classifier \n')

    print clf.best_params_
    txtfile.write(str(clf.best_params_))
    print "\nBest Score\n"
    txtfile.write("\nBest Score\n")
    print clf.best_score_
    txtfile.write(str(clf.best_score_))
    

if __name__ == '__main__':
    gridsearch_algs()