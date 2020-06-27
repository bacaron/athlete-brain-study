import pandas as pd
import os,glob,sys
import numpy as np
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from plot_cortex_data import plotLobeMicrostructureAverage

def computeLobeMicrostructure(topPath,labelsPath,dataPath,groups,subjects,colors):

	# load newly created cortex_nodes.csv file
	data = pd.read_csv(dataPath+'/cortex_nodes.csv')

	# measures to loop through
	diff_measures = ['ad','fa','md','rd','ndi','isovf','odi']

	# set important and dumby variables
	lobes = ['frontal','temporal','occipital','parietal','insular','limbic','motor','somatosensory']
	parcels = {}
	parcel_lobe_id = {}
	lobe_data = pd.DataFrame([])

	# loop through lobes and append data to lobes_data dataframe
	for l in range(len(lobes)):
		print(lobes[l])
		parcels[lobes[l]] = pd.read_csv((labelsPath+'/'+lobes[l]+'_lobes.txt'),header=None)[0].tolist()
		parcel_lobe_id[lobes[l]] = range(len(parcels[lobes[l]]))
		tmpdata = data[data['structureID'].isin(parcels[lobes[l]])]
		tmpnodes = list(parcel_lobe_id[lobes[l]])
		for s in data['subjectID'].unique().tolist():
			tmpdata['nodeID'].loc[tmpdata['subjectID'] == s] = [ x+1 for x in tmpnodes ]
		
		tmpdata['structureID'] = [lobes[l] for f in range(len(tmpdata['structureID']))]
		lobe_data = lobe_data.append(tmpdata)
	
	# generate plots for each measure column
	plotLobeMicrostructureAverage(groups,colors,lobes,lobe_data,diff_measures,dir_out=topPath+'/img/')

	# output data structure for records and any further analyses
	if not os.path.exists(dataPath):
		os.mkdir(dataPath)

	lobe_data.to_csv(dataPath+'lobes_nodes.csv',index=False)

if __name__ == '__main__':
	computeLobeMicrostructure(sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])