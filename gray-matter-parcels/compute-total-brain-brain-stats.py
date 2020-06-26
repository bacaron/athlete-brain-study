import pandas as pd
import os,glob,sys
import numpy as np
from matplotlib import colors as mcolors
import pandas as pd
import matplotlib.pyplot as plt

def compute_total_brain_stats(topPath,groups,subjects,colors):

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
		
	data['Total Cortical Thickness'] = [ np.mean([data['Left Hemisphere Mean Cortical Gray Matter Thickness'][f],data['Left Hemisphere Mean Cortical Gray Matter Thickness'][f]]) for f in range(len(data['subjectID']))]

	## generate plots
	for dc in data_columns:
		plotAparcWholeBrainData(groups,colors,dc,data,dir_out="/img/")

if __name__ == '__main__':
	compute_total_brain_stats(sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3])