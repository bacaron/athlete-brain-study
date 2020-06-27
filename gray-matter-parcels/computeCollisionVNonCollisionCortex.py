import pandas as pd
import os,glob,sys
import numpy as np
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from plot_cortex_data import collisionVNonCollisionScatter

def computeCollisionVNonCollisionCortex(topPath,dataPath,groups,subjects,colors):

	# create structure for subcort structures
	data_subcort = pd.DataFrame([])

	# list of subcortical structures of interest
	subcort_list = ['Left-Cerebellum-Cortex','Left-Thalamus-Proper','Left-Caudate','Left-Putamen','Left-Pallidum','Brain-Stem','Left-Hippocampus','Left-Amygdala','Left-Accumbens-area','Left-VentralDC','Right-Cerebellum-Cortex','Right-Thalamus-Proper','Right-Caudate','Right-Putamen','Right-Pallidum','Right-Hippocampus','Right-Amygdala','Right-Accumbens-area','Right-VentralDC'] 

	# measures to loop through
	diff_measures = ['ad','fa','md','rd','ndi','isovf','odi']

	# load newly created cortex_nodes.csv file
	data = pd.read_csv(dataPath+'/cortex_nodes.csv')

	for g in range(len(groups)):
		for s in range(len(subjects[groups[g]])):
			filepath = str(topPath+'/'+str(subjects[groups[g]][s])+'/parc-stats-subcort/aseg_nodes.csv')
			data_subcort = data_subcort.append(pd.read_csv(filepath))

	data_subcort = data_subcort[data_subcort['structureID'].isin(subcort_list)]

	# generate plots
	collisionVNonCollisionScatter(groups,colors,data,data_subcort,diff_measures,dir_out=topPath+'/img/')

	# save subcort nodes file
	if not os.path.exists(dataPath):
		os.mkdir(dataPath)

	data_subcort.to_csv(dataPath+'subcortex_nodes.csv',index=False)

if __name__ == '__main__':
	collisionVNonCollisionScatter(sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])