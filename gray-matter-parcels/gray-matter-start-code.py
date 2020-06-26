import pandas as pd
import os,glob,sys
import numpy as np
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt

def (topPath,groups,colors):

	filepaths = {}
	filepaths['paths'] = []

	# specific to only my dataset. need to find better way to generalize
	for g in range(len(groups)):
		if g == 0:
			filepaths['paths'].append([f for f in glob.glob(topPath+"/tract-stats/*1_0*")])
		elif g == 1:
			filepaths['paths'].append([f for f in glob.glob(topPath+"/tract-stats/*2_0*")])
		elif g == 2:
			filepaths['paths'].append([f for f in glob.glob(topPath+"/tract-stats/*3_0*")])

	filepaths['paths'].sort()

	# create pandas dataframe of data by reading in csvs
	data = {}
	data['raw'] = []
	data['mean'] = []
	data['sd'] = []
	streamline_counts = pd.DataFrame([],columns=['mean','sd'])
	streamline_lengths = pd.DataFrame([],columns=['mean','sd'])
	volume = pd.DataFrame([],columns=['mean','sd'])

	for g in range(len(groups)):
		data['raw'].append(pd.concat(map(pd.read_csv,filepaths['paths'][g])))
		data['raw'][g] = data['raw'][g][data['raw'][g].TractName != 'wbfg']
		data['mean'].append(data['raw'][g].groupby('TractName').mean())
		data['sd'].append(data['raw'][g].groupby('TractName').std())

		# build streamline count dataframe
		streamline_counts_raw = {'mean': data['mean'][g].StreamlineCount.tolist(), 'sd': data['sd'][g].StreamlineCount.tolist()}
		streamline_counts = streamline_counts.append(streamline_counts_raw,ignore_index=True)

		# build volume dataframe
		volume_raw = {'mean': data['mean'][g].volume.tolist(), 'sd': data['sd'][g].volume.tolist()}
		volume = volume.append(volume_raw,ignore_index=True)

		# build streamline lenght dataframe
		streamline_lengths_raw = {'mean': data['mean'][g].avgerageStreamlineLength.tolist(), 'sd': data['sd'][g].avgerageStreamlineLength.tolist()}
		streamline_lengths = streamline_lengths.append(streamline_lengths_raw,ignore_index=True)

	tract_names = data['raw'][0].TractName[0:61].tolist()

	## generate plots
	plot_tract_data(groups,colors,'Streamline Counts',streamline_counts,tract_names,dir_out=topPath+"/img/")
	plot_tract_data(groups,colors,'Volume',volume,tract_names,dir_out=topPath+"/img/")
	plot_tract_data(groups,colors,'Streamline Length',streamline_lengths,tract_names,dir_out=topPath+"/img/")

	#return [streamline_counts,volume,streamline_lengths]

if __name__ == '__main__':
	compute_plot_length_volume_final(sys.argv[0],sys.argv[1],sys.argv[2])