#!/usr/bin/env python3

import os,sys,glob
from matplotlib import colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
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

### setting up variables and adding paths
print("setting up variables")
topPath = "/media/brad/APPS/athlete-updated-pipeline-qc"
os.chdir(topPath)
data_dir = topPath+'/data/'
scripts_dir = topPath+'/athlete_brain_study/machine-learning/'
mlc_dir = topPath+'/mlc/'
mlc_data_dir = mlc_dir+'/data/'
text_dir = mlc_dir+'gridsearch_results'
img_dir = mlc_dir+'img/'
results_dir = mlc_dir+'results/'
results_individ_dir = mlc_dir+'results_per_model_mlc/'

if not os.path.exists(mlc_dir):
	os.mkdir(mlc_dir)
if not os.path.exists(img_dir):
	os.mkdir(img_dir)
if not os.path.exists(text_dir):
	os.mkdir(text_dir)
if not os.path.exists(mlc_data_dir):
	os.mkdir(mlc_data_dir)
if not os.path.exists(results_dir):
	os.mkdir(results_dir)
if not os.path.exists(results_individ_dir):
	os.mkdir(results_individ_dir)
sys.path.insert(0,scripts_dir)

# tissue types
tissues = ['track_mean','cortex','subcortex','lobes','wholebrain']

# measures
measures = {}
measures_loop = ['all_diffusion','dti','noddi','track','cortex','subcortex','wholebrain']
measures[measures_loop[0]] = ['ad','fa','md','rd','ndi','isovf','odi']
measures[measures_loop[1]] = measures[measures_loop[0]][0:4]
measures[measures_loop[2]] = measures[measures_loop[0]][4::]
measures[measures_loop[3]] = ['volume','length','count']
measures[measures_loop[4]] = ['thickness','volume']
measures[measures_loop[5]] = measures[measures_loop[4]][1]
measures[measures_loop[6]] = ['Total Brain Volume','Total Cortical Gray Matter Volume','Total White Matter Volume','Total Cortical Thickness']

# comparison models
model_labels = {}
models = ['indvidual','athlete_v_nonathlete','ses','RHI']

### set up mlcs label array
mlcs = ['Random Forest Classifier','Adaboost Classifier','Support Vector Classifier','KNeighbors Classifier','Decision Tree Classifier','Logistic Regression']

### set up dictionary to identify best parameters from
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

### load data
print("loading data")
df_subjects = pd.read_csv(data_dir+'subjects.csv')
df = {}
for tt in tissues:
	df[tt] = pd.read_csv(data_dir+tt+'_nodes.csv')
	if tt != 'wholebrain':
		nan_structs = list(df[tt][df[tt][measures[measures_loop[0]][0]].isnull()]['structureID'])
		if nan_structs:
			print("missing structs: \n"+str(df[tt][df[tt][measures[measures_loop[0]][0]].isnull()]),file=open(mlc_dir+tt+'_missing_structs.txt',"w"))
			df[tt] = df[tt][~df[tt]['structureID'].isin(nan_structs)]
print("data loaded")

### set up labels to use
print("generating labels")
model_labels[models[0]] = [int(f.split('_')[0]) for f in df_subjects['subjectID']] # individual groups
model_labels[models[1]] = [ 1 if int(f.split('_')[0]) <= 2 else 2 for f in df_subjects['subjectID']] # athletes
model_labels[models[2]] = [ 1 if int(f.split('_')[0]) in [1,3] else 2 for f in df_subjects['subjectID']] # SES
model_labels[models[3]] = [ 1 if int(f.split('_')[0]) < 2 else 2 for f in df_subjects['subjectID']] # RHI
print("labels generated")

### identify best parameters for each tissue type
print("Identifying best mlc parameters")
if not os.path.exists(mlc_data_dir+'best_params_struct.json'):
	from mlc_scripts import gridsearch_algs
	best_parameters = {}
	for tt in tissues:
		print(tt)
		if tt != 'wholebrain':
			best_parameters[tt] = gridsearch_algs(tt,df[tt],df_subjects,measures[measures_loop[0]],model_labels[models[0]],mlc_dict,text_dir,mlc_data_dir)
		else:
			best_parameters[tt] = gridsearch_algs(tt,df[tt],df_subjects,measures[measures_loop[6]],model_labels[models[0]],mlc_dict,text_dir,mlc_data_dir)

	# write out best parameters for easier loading
	with open(mlc_data_dir+'best_params_struct.json',"w") as best_params_f:
		json.dump(best_parameters,best_params_f)
else:
	# load saved parameters
	with open(mlc_data_dir+'best_params_struct.json',"r") as best_params_f:
		best_parameters = json.load(best_params_f)
print("best mlc parameters identified")

### running mlc analyses - diffusion
from mlc_scripts import runModel
for ml in range(len(measures_loop[0:3])):
	print("running mlc analyses: %s" %measures_loop[ml])
	for tt in tissues[0:3]:
		print(tt)
		tissue_type_output = pd.DataFrame([],columns={'iterations','mlc','model','percentages'})
		for mc in range(len(mlcs)):
			print(mlcs[mc])
			model = mlc_dict[mlcs[mc]]['model'].set_params(**best_parameters[tt][mlcs[mc]]['parameters'])
			for mm in range(len(models)):
				print(models[mm])
				out_name = 'results_'+tt+'_'+models[mm]+'_'+mlcs[mc].replace(' ','_')+'_'+measures_loop[ml]+'.csv'
				print("results will be saved to "+results_dir+out_name)
				if tt != 'wholebrain':
					tissue_type_output = runModel(df[tt],df_subjects,model,mlcs[mc],model_labels[models[mm]],models[mm],measures[measures_loop[ml]],100,results_individ_dir,out_name,tissue_type_output)
		
		# output tissue type results
		tissue_type_output.to_csv(results_dir+'results_'+tt+'_'+measures_loop[ml]+'.csv',index=False)





# ### whole brain measures mlc
# 			else:
# 				tissue_type_output = runModel(df[tt],df_subjects,model,mlcs[mc],model_labels[models[mm]],models[mm],wholebrain_measures,2,results_individ_dir,out_name,tissue_type_output)


print("machine learning complete!")

