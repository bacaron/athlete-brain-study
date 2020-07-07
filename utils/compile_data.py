#!/usr/bin/env python

import os,glob,sys
import numpy as np
import pandas as pd
import json

### subjects
def collectSubjectData(topPath,dataPath,groups,subjects,colors):

	# set up variables
	data_columns = ['subjectID','class','colors']
	data =  pd.DataFrame([],columns=data_columns)

	# populate structure
	data['subjectID'] = [ f for g in groups for f in subjects[g] ]
	data['class'] = [ g for g in groups for f in range(len(subjects[g]))]
	data['colors'] = [ colors[c] for c in colors for f in subjects[c]]

	# output data structure for records and any further analyses
	if not os.path.exists(dataPath):
		os.mkdir(dataPath)

	data.to_csv(dataPath+'subjects.csv',index=False)

	return data

### snr data
def load_snr_stat(snr_json):

	dim = len(snr_json["SNR in b0, X, Y, Z"])
	snr_data = np.zeros(dim)
	for i in range(dim):
	    snr_data[i] = float(snr_json["SNR in b0, X, Y, Z"][i])
	    
	return snr_data

def collectSNRData(topPath,dataPath,groups,subjects):
	from compile_data import load_snr_stat
	import json

	# set up variables
	snr = []
	data = pd.DataFrame([])
	data_columns = ['subjectID','snr']

	for g in range(len(groups)):
		for s in range(len(subjects[groups[g]])):
			filepath = str(topPath+'/'+str(subjects[groups[g]][s])+'/snr/output/snr.json')
			with open(filepath) as filepath_j:
				config = json.load(filepath_j)
			snr.append(load_snr_stat(config))

	data['subjectID'] = [ f for g in groups for f in subjects[g] ]
	data['snr'] = snr

	# output data structure for records and any further analyses
	if not os.path.exists(dataPath):
		os.mkdir(dataPath)

	data.to_csv(dataPath+'snr.csv',index=False)

	return data

### white matter
def collectTrackMacroData(topPath,dataPath,groups,subjects):
	
	# set up variables
	data_columns = ['subjectID','nodeID','structureID','count','length','volume']

	# set up empty data frame
	data = pd.DataFrame([])

	# create pandas dataframe. specific to only my dataset. need to find better way to generalize
	for g in range(len(groups)):
		for s in range(len(subjects[groups[g]])):
			filepath = str(topPath+'/'+str(subjects[groups[g]][s])+'/tractmeasures-cleaned/output_FiberStats.csv')
			tmpdata = pd.read_csv(filepath)
			tmpdata['subjectID'] = [ subjects[groups[g]][s] for f in range(len(tmpdata['TractName']))]
			tmpdata['nodeID'] = [ 1 for f in range(len(tmpdata['TractName'])) ]
			data = data.append(tmpdata,ignore_index=True) 

	data = data[['subjectID','nodeID','TractName','StreamlineCount','avgerageStreamlineLength','volume']]
	data.columns = data_columns

	# identify track_names
	track_names = list(data['structureID'][data['structureID'] != 'wbfg'].unique())

	# output data structure for records and any further analyses
	if not os.path.exists(dataPath):
		os.mkdir(dataPath)

	data.to_csv(dataPath+'track_macro_nodes.csv',index=False)

	# output track names
	if not os.path.exists(dataPath):
		os.mkdir(dataPath)

	with open((dataPath+'track_list.json'),'w') as track_listf:
		json.dump(track_names,track_listf)

	return [track_names,data]

def collectTrackMicroData(topPath,dataPath,groups,subjects,num_nodes):
	
	# set up empty data frame
	data = pd.DataFrame([])

	# create pandas dataframe. specific to only my dataset. need to find better way to generalize
	for g in range(len(groups)):
		for s in range(len(subjects[groups[g]])):
			filepath = str(topPath+'/'+str(subjects[groups[g]][s])+'/tractmeasures-profiles/tracts.csv')
			data = data.append(pd.read_csv(filepath),ignore_index=True)

	# identify inner n nodes based on num_nodes input
	total_nodes = len(data['nodeID'].unique())
	cut_nodes = int((total_nodes - num_nodes) / 2)

	# remove cut_nodes from dataframe
	data = data[data['nodeID'].between((cut_nodes)+1,(num_nodes+cut_nodes))]

	# replace empty spaces with nans
	data = data.replace(r'^\s+$', np.nan, regex=True)

	# output data structure for records and any further analyses
	if not os.path.exists(dataPath):
		os.mkdir(dataPath)

	data.to_csv(dataPath+'track_micro_nodes.csv',index=False)

	return data

def combineTrackMacroMicro(dataPath,macroData,microData):

	# remove wbfg from macro data
	macroData =  macroData[macroData['structureID'] != 'wbfg']

	# merge data frames
	data = pd.merge(microData,macroData.drop(columns='nodeID'),on=['subjectID','structureID'])

	# make mean data frame
	data_mean =  data.groupby(['subjectID','structureID']).mean().reset_index()
	data_mean['nodeID'] = [ 1 for f in range(len(data_mean['nodeID'])) ]

	# output data structure for records and any further analyses
	if not os.path.exists(dataPath):
		os.mkdir(dataPath)

	data.to_csv(dataPath+'track_total_nodes.csv',index=False)
	data_mean.to_csv(dataPath+'track_mean_nodes.csv',index=False)

	return [data,data_mean]

### functional track/lobedata
def compileFunctionalData(dataPath,structureData,functionalLabels,labelsPath):

	# set important and dumby variables
	parcels = {}
	parcel_id = {}
	functional_data = pd.DataFrame([])

	# loop through functional domains and append data to dataframe
	for fl in range(len(functionalLabels)):
		print(functionalLabels[fl])
		if functionalLabels[fl] in ['association','projection','commissural']:
			tissue = 'tracks'
		else:
			tissue = 'lobes'

		parcels[functionalLabels[fl]] = pd.read_csv((labelsPath+'/'+functionalLabels[fl]+'_'+tissue+'.txt'),header=None)[0].tolist()
		parcel_id[functionalLabels[fl]] = range(len(parcels[functionalLabels[fl]]))
		tmpdata = structureData[structureData['structureID'].isin(parcels[functionalLabels[fl]])]
		tmpnodes = list(parcel_id[functionalLabels[fl]])
		for s in structureData['subjectID'].unique().tolist():
			tmpdata['nodeID'].loc[tmpdata['subjectID'] == s] = [ x+1 for x in tmpnodes ]
		
		tmpdata['functionalID'] = [functionalLabels[fl] for f in range(len(tmpdata['structureID']))]
		functional_data = functional_data.append(tmpdata)

	# output track names
	if not os.path.exists(dataPath):
		os.mkdir(dataPath)

	functional_data.to_csv(dataPath+'functional_'+tissue+'_nodes.csv',index=False)

	return functional_data

### gray matter
def collectCorticalParcelData(topPath,dataPath,groups,subjects):
	
	# define top variables
	data = pd.DataFrame([])
	data_volume = pd.DataFrame([])
	
	# loop through subjects and load parcellation data. specific to only my dataset. need to find better way to generalize
	for g in range(len(groups)):
		for s in range(len(subjects[groups[g]])):
			filepath = str(topPath+'/'+str(subjects[groups[g]][s])+'/parc-stats-cortex/aparc_MEAN.csv')
			data = data.append(pd.read_csv(filepath),ignore_index=True) 
			
			# load actual volume data, not vertex volume
			volume_filepath = str(topPath+'/'+str(subjects[groups[g]][s])+'/parc-stats-aparc/cortex.csv')
			data_volume = pd.read_csv(volume_filepath)

			data['volume'][data['subjectID'] == subjects[groups[g]][s]] = data_volume['gray_matter_volume_mm^3'].tolist()

	# output data structure for records and any further analyses
	if not os.path.exists(dataPath):
		os.mkdir(dataPath)

	data.to_csv(dataPath+'cortex_nodes.csv',index=False)

	return data

def collectSubCorticalParcelData(topPath,dataPath,groups,subjects):
	
	# define top variables
	data = pd.DataFrame([])
	
	# list of subcortical structures of interest
	subcort_list = ['Left-Cerebellum-Cortex','Left-Thalamus-Proper','Left-Caudate','Left-Putamen','Left-Pallidum','Brain-Stem','Left-Hippocampus','Left-Amygdala','Left-Accumbens-area','Left-VentralDC','Right-Cerebellum-Cortex','Right-Thalamus-Proper','Right-Caudate','Right-Putamen','Right-Pallidum','Right-Hippocampus','Right-Amygdala','Right-Accumbens-area','Right-VentralDC'] 
	
	# loop through subjects and load parcellation data. specific to only my dataset. need to find better way to generalize
	for g in range(len(groups)):
		for s in range(len(subjects[groups[g]])):
			filepath = str(topPath+'/'+str(subjects[groups[g]][s])+'/parc-stats-subcort/aseg_nodes.csv')
			data = data.append(pd.read_csv(filepath))

	data = data[data['structureID'].isin(subcort_list)]

	# output data structure for records and any further analyses
	if not os.path.exists(dataPath):
		os.mkdir(dataPath)

	data.to_csv(dataPath+'subcortex_nodes.csv',index=False)

	return data

def combineCorticalSubcortical(dataPath,corticalData,subcorticalData):
	
	# remove unnecessary columns
	corticalData = corticalData.drop(columns=['snr','thickness'])
	subcorticalData = subcorticalData.drop(columns=['parcID','number_of_voxels'])

	# merge data frames
	data = pd.concat([corticalData,subcorticalData],sort=False)

	# output data structure for records and any further analyses
	if not os.path.exists(dataPath):
		os.mkdir(dataPath)

	data.to_csv(dataPath+'graymatter_nodes.csv',index=False)

	# identify gray matter names
	graymatter_names = list(data['structureID'].unique())

	# output track names
	if not os.path.exists(dataPath):
		os.mkdir(dataPath)

	with open((dataPath+'graymatter_list.json'),'w') as gm_listf:
		json.dump(graymatter_names,gm_listf)

	return [graymatter_names,data]

# def compileLobeData(dataPath,corticalData,lobes,labelsPath):

# 	# set important and dumby variables
# 	parcels = {}
# 	parcel_lobe_id = {}
# 	lobe_data = pd.DataFrame([])

# 	# loop through lobes and append data to lobes_data dataframe
# 	for l in range(len(lobes)):
# 		print(lobes[l])
# 		parcels[lobes[l]] = pd.read_csv((labelsPath+'/'+lobes[l]+'_lobes.txt'),header=None)[0].tolist()
# 		parcel_lobe_id[lobes[l]] = range(len(parcels[lobes[l]]))
# 		tmpdata = corticalData[corticalData['structureID'].isin(parcels[lobes[l]])]
# 		tmpnodes = list(parcel_lobe_id[lobes[l]])
# 		for s in corticalData['subjectID'].unique().tolist():
# 			tmpdata['nodeID'].loc[tmpdata['subjectID'] == s] = [ x+1 for x in tmpnodes ]
		
# 		tmpdata['structureID'] = [lobes[l] for f in range(len(tmpdata['structureID']))]
# 		lobe_data = lobe_data.append(tmpdata)

# 	lobe_data_mean = lobe_data.groupby(['subjectID','structureID']).mean().reset_index()
# 	lobe_data_mean['nodeID'] = [ 1 for f in range(len(lobe_data_mean['nodeID'])) ]

# 	# output track names
# 	if not os.path.exists(dataPath):
# 		os.mkdir(dataPath)

# 	lobe_data.to_csv(dataPath+'lobes_nodes.csv',index=False)
# 	lobe_data_mean.to_csv(dataPath+'lobes_mean_nodes.csv')

	return [lobe_data,lobe_data_mean]

### whole brain
def collectWholeBrainStats(topPath,dataPath,groups,subjects):

	# define top variables
	filepath = []
	data = pd.DataFrame([])

	# measures to loop through
	data_columns = ['subjectID','Total Brain Volume','Total Gray Matter Volume','Total White Matter Volume','Total Cortical Thickness']

	# set up dumby data structure for loading
	# specific to only my dataset. need to find better way to generalize
	for g in range(len(groups)):
		for s in range(len(subjects[groups[g]])):
			filepath = str(topPath+'/'+str(subjects[groups[g]][s])+'/parc-stats-aparc/whole_brain.csv')
			data = data.append(pd.read_csv(filepath),ignore_index=True)
		
	# set average whole brain cortical thickness
	data['Total Cortical Thickness'] = [ np.mean([data['Left Hemisphere Mean Cortical Gray Matter Thickness'][f],data['Left Hemisphere Mean Cortical Gray Matter Thickness'][f]]) for f in range(len(data['subjectID']))]

	# output data structure for records and any further analyses
	if not os.path.exists(dataPath):
		os.mkdir(dataPath)

	data.to_csv(dataPath+'wholebrain_nodes.csv',index=False)

	return data

 