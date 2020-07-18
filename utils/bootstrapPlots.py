#!/usr/bin/env python

def bootstrapDifferencePlots(groups,subjects,stat,diffusion_measures,colors,repetitions,alpha,dir_out,data_dir):
	import matplotlib.pyplot as plt
	import os,sys
	import seaborn as sns
	import pandas as pd
	import numpy as np
	from itertools import combinations

	if not os.path.exists(data_dir):
		os.mkdir(data_dir)
	
	img_out = "track_difference_bootstrap_histograms.eps"
	img_out_png = "track_difference_bootstrap_histograms.png"

	f, axes = plt.subplots(2, 4, figsize=(15, 15), sharex=True, sharey=True)
	f.suptitle("Average Bootrap Difference - Tracts")

	resample = {}
	for g in groups:
		resample[g] = {}
		for i in range(repetitions):
			resample[g][i] = list(pd.Series(subjects[g]).sample(len(subjects[g]),replace=True))

	comparison_array = list(combinations(groups,2))

	for dm in range(len(diffusion_measures)):
		if diffusion_measures[dm] in ['ad','fa','md','rd']:
			row = 1
			column = dm
		else:
			row = 0
			column = dm - 4

		#plt.xlim([-0.1,0.1])

		# set spines and ticks
		axes[row,column].spines['right'].set_visible(False)
		axes[row,column].spines['top'].set_visible(False)
		axes[row,column].yaxis.set_ticks_position('left')
		axes[row,column].xaxis.set_ticks_position('bottom')

		axes[row,column].patch.set_visible(False)

		comparison = {}
		cli = {}
		g=0
		for compar in comparison_array:
			comparison[compar[0]+"_"+compar[1]] = pd.DataFrame([])
			cli[compar[0]+"_"+compar[1]] = {}
			for i in range(repetitions):
				comparison[compar[0]+"_"+compar[1]] = comparison[compar[0]+"_"+compar[1]].append(np.mean(stat[['structureID',diffusion_measures[dm]]][stat['subjectID'].isin(resample[compar[0]][i])].groupby('structureID').mean()) - np.mean(stat[['structureID',diffusion_measures[dm]]][stat['subjectID'].isin(resample[compar[1]][i])].groupby('structureID').mean()),ignore_index=True)
			comparison[compar[0]+"_"+compar[1]] = comparison[compar[0]+"_"+compar[1]].sort_values(by=diffusion_measures[dm])
			comparison[compar[0]+"_"+compar[1]].to_csv(data_dir+'track_boostrap_difference_data_'+compar[0]+"_"+compar[1]+"_"+diffusion_measures[dm]+".csv",index=False)
			cli[compar[0]+"_"+compar[1]] = [np.percentile(comparison[compar[0]+"_"+compar[1]],alpha/2*100),np.percentile(comparison[compar[0]+"_"+compar[1]],100-alpha/2*100),]
			if g == 1:
				Colors = colors[compar[1]]
			else:
				Colors = colors[compar[0]]
			
			sns.distplot(comparison[compar[0]+"_"+compar[1]],kde=False,hist=True,hist_kws={"histtype": "step","linewidth": 3},color=Colors,bins=20,ax=axes[row,column],axlabel=diffusion_measures[dm],label="mean: %.3f" %np.mean(comparison[compar[0]+"_"+compar[1]]))
			axes[row,column].axvline(cli[compar[0]+"_"+compar[1]][0],color=Colors,linestyle='--',linewidth=3)
			axes[row,column].axvline(cli[compar[0]+"_"+compar[1]][1],color=Colors,linestyle='--',linewidth=3)
			g=g+1

	# save or show plot
	if dir_out:
		if not os.path.exists(dir_out):
			os.mkdir(dir_out)

		plt.savefig(os.path.join(dir_out, img_out))
		plt.savefig(os.path.join(dir_out, img_out_png))       
	else:
		plt.show()