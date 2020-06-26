import pandas as pd
import os,glob,sys
import numpy as np
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from plot_cortex_data import plotCorticalParcelData

def computeThicknessVolumeDiffRoi(topPath,dataPath,groups,subjects,colors):

	# define top variables
	filepath = []
	volume_filepath = []
	data = pd.DataFrame([])
	data_volume = pd.DataFrame([])

	# measures to loop through
	data_columns = ['subjectID','ad','fa','md','rd','ndi','isovf','odi','volume','thickness']
	diff_measures = ['ad','fa','md','rd','ndi','isovf','odi']
	stats_measure =  ['volume','thickness']
	
	# loop through subjects and load parcellation data. specific to only my dataset. need to find better way to generalize
	for g in range(len(groups)):
		for s in range(len(subjects[groups[g]])):
			filepath = str(topPath+'/'+str(subjects[groups[g]][s])+'/parc-stats-cortex/aparc_MEAN.csv')
			data = data.append(pd.read_csv(filepath),ignore_index=True) 
			
			# load actual volume data, not vertex volume
			volume_filepath = str(topPath+'/'+str(subjects[groups[g]][s])+'/parc-stats-aparc/cortex.csv')
			data_volume = pd.read_csv(volume_filepath)

			data['volume'][data['subjectID'] == subjects[groups[g]][s]] = data_volume['gray_matter_volume_mm^3'].tolist()

	# fix scaling
	data[['ad','md','rd']] = data[['ad','md','rd']] * 1000
	
	# generate plots for each measure column
	for dc in stats_measure:
		plotCorticalParcelData(groups,colors,dc,data,diff_measures,dir_out=topPath+'/img/')

	# output data structure for records and any further analyses
	if not os.path.exists(dataPath):
		os.mkdir(dataPath)

	data.to_csv(dataPath+'cortex_nodes.csv',index=False)

if __name__ == '__main__':
	computeThicknessVolumeDiffRoi(sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])