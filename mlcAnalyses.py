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
scripts_dir = topPath+'/athlete_brain_study/utils/'
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
tissues = ['functional_tracks','functional_lobes','subcortex','wholebrain']
tissue_names = ['track_mean','cortex','subcortex','wholebrain']

# functional labels
functional_labels = {}
functional_labels['track_mean'] = ['association','projection','commissural']
functional_labels['cortex'] = ['frontal','temporal','parietal','occipital','insular','limbic','motor','somatosensory']

# measures
measures = {}
measures_loop = ['all_diffusion','dti','noddi','track_mean_nondiffusion','cortex_nondiffusion','subcortex_nondiffusion','wholebrain_nondiffusion','noddi_ndi_odi','noddi_ndi_isovf','noddi_odi_isovf','dti_fa_md']
measures[measures_loop[0]] = ['ad','fa','md','rd','ndi','isovf','odi']
measures[measures_loop[1]] = measures[measures_loop[0]][0:4]
measures[measures_loop[2]] = measures[measures_loop[0]][4::]
measures[measures_loop[3]] = ['volume','length','count']
measures[measures_loop[4]] = ['thickness','volume']
measures[measures_loop[5]] = measures[measures_loop[4]][1]
measures[measures_loop[6]] = ['Total Brain Volume','Total Cortical Gray Matter Volume','Total White Matter Volume','Total Cortical Thickness']
measures[measures_loop[7]] = [measures[measures_loop[0]][4],measures[measures_loop[0]][6]]
measures[measures_loop[8]] = measures[measures_loop[0]][4:6]
measures[measures_loop[9]] = measures[measures_loop[0]][5:7]
measures[measures_loop[10]] = measures[measures_loop[0]][1:3]

# comparison models
model_labels = {}
models = ['individual','athlete_v_nonathlete','ses','RHI']

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
for tt in range(len(tissue_names)):
	df[tissue_names[tt]] = pd.read_csv(data_dir+tissues[tt]+'_nodes.csv')
	if tissue_names[tt] != 'wholebrain':
		nan_structs = list(df[tissue_names[tt]][df[tissue_names[tt]][measures[measures_loop[0]][0]].isnull()]['structureID'])
		if nan_structs:
			print("missing structs: \n"+str(df[tissue_names[tt]][df[tissue_names[tt]][measures[measures_loop[0]][0]].isnull()]),file=open(mlc_dir+tissue_names[tt]+'_missing_structs.txt',"w"))
			df[tissue_names[tt]] = df[tissue_names[tt]][~df[tissue_names[tt]]['structureID'].isin(nan_structs)]
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
from mlc_scripts import gridsearch_algs
# diffusion based measures
for ml in range(len(measures_loop)):
	if not os.path.exists(mlc_data_dir+'best_params_struct_'+measures_loop[ml]+'.json'):
		best_parameters = {}
		if not 'nondiffusion' in measures_loop[ml]:
			for tt in tissue_names[0:3]:
				print(tt)
				best_parameters[tt] = gridsearch_algs(tt,df[tt],df_subjects,measures[measures_loop[ml]],measures_loop[ml],model_labels[models[0]],mlc_dict,text_dir,mlc_data_dir)
		else:
			best_parameters[measures_loop[ml]] = gridsearch_algs(measures_loop[ml],df[measures_loop[ml]].split('_')[0],df_subjects,measures[measures_loop[ml]],measures_loop[ml],model_labels[models[0]],mlc_dict,text_dir,mlc_data_dir)

		# write out best parameters for easier loading
		out_name = mlc_data_dir+'best_params_struct_'+measures_loop[ml]+'.json'

		with open(out_name,"w") as best_params_f:
			json.dump(best_parameters,best_params_f)
print("best mlc parameters identified")

### running mlc analyses
from mlc_scripts import runModel
for ml in range(len(measures_loop)):
	print("running mlc analyses: %s" %measures_loop[ml])
	
	# load saved parameters
	with open(mlc_data_dir+'best_params_struct_'+measures_loop[ml]+'.json',"r") as best_params_f:
		best_parameters = json.load(best_params_f)
	
	# diffusion measures
	if not 'nondiffusion' in measures_loop[ml]:
		# individual structures
		print("individual structures")
		for tt in tissue_names[0:3]:
			if not os.path.exists(results_dir+'results_'+tt+'_'+measures_loop[ml]+'.csv'):
				print(tt)
				tissue_type_output = pd.DataFrame([],columns={'iterations','mlc','model','percentages'})
				for mc in range(len(mlcs)):
					print(mlcs[mc])
					model = mlc_dict[mlcs[mc]]['model'].set_params(**best_parameters[tt][mlcs[mc]]['parameters'])
					for mm in range(len(models)):
						print(models[mm])
						out_name = 'results_'+tt+'_'+models[mm]+'_'+mlcs[mc].replace(' ','_')+'_'+measures_loop[ml]+'.csv'
						tissue_type_output = runModel(df[tt],df_subjects,model,mlcs[mc],model_labels[models[mm]],models[mm],measures[measures_loop[ml]],100,results_individ_dir,out_name,tissue_type_output)
				
				# output tissue type results
				tissue_type_output.to_csv(results_dir+'results_'+tt+'_'+measures_loop[ml]+'.csv',index=False)

		# functional domain groupings
		print("functional domain groupings")
		for tt in tissue_names[0:2]:
			for fl in functional_labels[tt]:
				if not os.path.exists(results_dir+'results_'+tt+'_functional_'+fl+'_'+measures_loop[ml]+'.csv'):
					print(tt)
					print(fl)
					tissue_type_output = pd.DataFrame([],columns={'iterations','mlc','model','percentages'})
					for mc in range(len(mlcs)):
						print(mlcs[mc])
						model = mlc_dict[mlcs[mc]]['model'].set_params(**best_parameters[tt][mlcs[mc]]['parameters'])
						out_name = 'results_'+tt+'_functional_'+fl+'_'+models[0]+'_'+mlcs[mc].replace(' ','_')+'_'+measures_loop[ml]
						tissue_type_output = runModel(df[tt][df[tt]['functionalID'] == fl],df_subjects,model,mlcs[mc],model_labels[models[0]],models[0],measures[measures_loop[ml]],100,results_individ_dir,out_name,tissue_type_output)
				
					# output tissue type results
					tissue_type_output.to_csv(results_dir+'results_'+tt+'_functional_'+fl+'_'+measures_loop[ml]+'.csv',index=False)

	# non diffusion measures
	else:
		if not os.path.exists(results_dir+'results_'+measures_loop[ml]+'.csv'):
			print("non-diffusion analyses: %s" %measures_loop[ml].split('_')[0])
			tissue_type_output = pd.DataFrame([],columns={'iterations','mlc','model','percentages'})
			for mc in range(len(mlcs)):
				print(mlcs[mc])
				model = mlc_dict[mlcs[mc]]['model'].set_params(**best_parameters[measures_loop[ml]][mlcs[mc]]['parameters'])
				out_name = 'results_'+models[0]+'_'+mlcs[mc].replace(' ','_')+'_'+measures_loop[ml]
				tissue_type_output = runModel(df[measures_loop[ml].split('_')[0]],df_subjects,model,mlcs[mc],model_labels[models[0]],models[0],measures[measures_loop[ml]],100,results_individ_dir,out_name,tissue_type_output)
				
			# output tissue type results
			tissue_type_output.to_csv(results_dir+'results_'+measures_loop[ml]+'.csv',index=False)
print("machine learning complete.")

### generate plots of mlc results
from mlc_scripts import computeRacBic,plotModelPerformance,plotMlcModelPerformance
for ml in range(len(measures_loop)):
	print("plotting mlc analyses: %s" %measures_loop[ml])
	# diffusion measures
	if not 'nondiffusion' in measures_loop[ml]:
		print("individual structures")
		## individual structures
		for tt in tissue_names[0:3]:
			out_name = results_dir+'results_'+tt+'_'+measures_loop[ml]+'_summary.csv'
			mlcdata = pd.read_csv(results_dir+'results_'+tt+'_'+measures_loop[ml]+'.csv')

			# calculate rac and bic
			[mlcdata,mlcDataSummary] = computeRacBic(mlcdata,models,mlcs,len(df_subjects))

			# save summary data structure and appended dataframe
			mlcdata.to_csv(results_dir+'results_'+tt+'_'+measures_loop[ml]+'.csv',index=False)
			mlcDataSummary.to_csv(out_name,index=False)

			# plot violin plots of model accuracy
			plotModelPerformance("model","percentages",mlcdata,img_dir,tt+"_"+measures_loop[ml]+"_accuracy")

			# plot violin plots of model rac performance
			plotModelPerformance("model","medianRac",mlcDataSummary,img_dir,tt+"_"+measures_loop[ml]+"_medianRac")

			# plot violin plots of mlc-by-model bic
			plotMlcModelPerformance("mlc","bic",mlcDataSummary,"bar",img_dir,tt+"_"+measures_loop[ml]+"_mlc_model_bic")

			# plot violin plots of mlc-by-model rac
			plotMlcModelPerformance("mlc","rac",mlcdata,"violin",img_dir,tt+"_"+measures_loop[ml]+"_mlc_model_rac")

			# plot violin plots of mlc-by-model accuracy
			plotMlcModelPerformance("mlc","percentages",mlcdata,"violin",img_dir,tt+"_"+measures_loop[ml]+"_mlc_model_accuracy")

		## functional-based
		print("functional domain groupings")
		for tt in tissue_names[0:2]:
			out_name = results_dir+'results_'+tt+'_functional_'+measures_loop[ml]+'_summary.csv'
			mlcfuncdata = pd.DataFrame([])

			# load data, concatenate into single structure and compute rac
			for fl in functional_labels[tt]:
				mlcdata = pd.read_csv(results_dir+'results_'+tt+'_functional_'+fl+'_'+measures_loop[ml]+'.csv')
				mlcdata['functionalID'] = [ fl for f in range(len(mlcdata['model'])) ]
				mlcfuncdata = pd.concat([mlcfuncdata,mlcdata],sort=False,ignore_index=True)

			# save data
			mlcfuncdata.to_csv(results_dir+'results_'+tt+'_functional_'+measures_loop[ml]+'.csv',index=False)

			# plot violin plot of accuracy
			plotModelPerformance("functionalID","percentages",mlcfuncdata,img_dir,tt+"_functional_"+measures_loop[ml]+"_accuracy")
	else:
		print("non-diffusion analyses: %s" %tissue_names[ml-3])
		# non diffusion measures
		out_name = results_dir+'results_'+measures_loop[ml]+'_summary.csv'
		mlcdata = pd.read_csv(results_dir+'results_'+measures_loop[ml]+'.csv')
		
		# plot violin plot of accuracy
		plotModelPerformance("mlc","percentages",mlcdata,img_dir,measures_loop[ml]+"_accuracy")
print("machine learning plotting complete.")

# ANOVA ANALYSES (TO DO)
