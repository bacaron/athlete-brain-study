import pandas as pd
import os,glob,sys
import numpy as np
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from plot_cortex_data import plotWholeBrainData

def computeTotalBrainStats(topPath,dataPath,groups,subjects,colors):

	# define top variables
	filepath = []
	data = pd.DataFrame([])

	# measures to loop through
	data_columns = ['subjectID','Total Brain Volume','Total Cortical Gray Matter Volume','Total White Matter Volume','Total Cortical Thickness']

	# loop through subjects and load parcellation data. specific to only my dataset. need to find better way to generalize
	for g in range(len(groups)):
		for s in range(len(subjects[groups[g]])):
			filepath = str(topPath+'/'+str(subjects[groups[g]][s])+'/parc-stats-aparc/whole_brain.csv')

			data = data.append(pd.read_csv(filepath),ignore_index=True)
	
	# no whole brain cortical thickness. need to average left and right hemisphere
	data['Total Cortical Thickness'] = [ np.mean([data['Left Hemisphere Mean Cortical Gray Matter Thickness'][f],data['Left Hemisphere Mean Cortical Gray Matter Thickness'][f]]) for f in range(len(data['subjectID']))]

	# generate plots for each measure column
	for dc in data_columns:
		plotWholeBrainData(groups,colors,dc,data,dir_out=topPath+'/img/')

	# output data structure for records and any further analyses
	if not os.path.exists(dataPath):
		os.mkdir(dataPath)

	data.to_csv(dataPath+'aparc_whole_brain.csv',index=False)

if __name__ == '__main__':
	computeTotalBrainStats(sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])