#!/usr/bin/env python

import os,sys
import numpy as np

def plotSNR(all_stat,all_sub,colors,dir_out):
	
	import matplotlib.pyplot as plt
	import os,sys
	import numpy as np
	from matplotlib import colors as mcolors

	# generate figures
	fig = plt.figure(figsize=(10,10))
	p = plt.subplot()

	# set up output names
	img_out='snr.eps'
	img_out_png='snr.png'

	all_sub.append("")
	all_sub.reverse()
	all_stat.reverse()
	colors.reverse()

	xmin = 0
	xmax = 45
	p.set_xlim([xmin, xmax])
	ymin = 1
	ymax = len(all_sub)
	p.set_xlim([xmin, xmax])
	p.set_ylim([ymin, ymax])
	p.spines['right'].set_visible(False)
	p.spines['top'].set_visible(False)
	p.spines['left'].set_position(('axes', -0.05))
	p.spines['bottom'].set_position(('axes', -0.05))
	p.yaxis.set_ticks(np.arange(len(all_sub)))
	p.yaxis.set_ticks_position('left')
	p.xaxis.set_ticks_position('bottom')
	p.set_yticklabels(all_sub)

	plt.title("SNR of diffusion signal")
	for s in range(len(all_stat)):
	    stat = all_stat[s]
	    color = colors[s]

	    p.errorbar(stat[1:].mean(), s+1, xerr=stat[1:].std(), marker='o', linestyle='None', color=color)
	    p.plot(stat[0], s+1, marker='x', color=color)
	
	# save or show plot
	if dir_out:
		plt.savefig(os.path.join(dir_out, img_out))
		plt.savefig(os.path.join(dir_out, img_out_png))       
	else:
		plt.show()

def plotTrackMacroData(groups,colors,stat_name,stat,diffusion_measures,dir_out):
	import matplotlib.pyplot as plt
	import os,sys
	import seaborn as sns
	from scipy import stats

	for dm in diffusion_measures:
		# set up output names
		img_out=str('track_'+stat_name+'_'+dm+'.pdf').replace(' ','_')
		img_out_png=str('track_'+stat_name+'_'+dm+'.png').replace(' ','_')

		# generate figures
		fig = plt.figure(figsize=(15,15))
		fig.patch.set_visible(False)
		p = plt.subplot()

		# set spines and ticks
		p.spines['right'].set_visible(False)
		p.spines['top'].set_visible(False)
		# p.spines['left'].set_position(('axes', -0.05))
		# p.spines['bottom'].set_position(('axes', -0.05))
		p.yaxis.set_ticks_position('left')
		p.xaxis.set_ticks_position('bottom')

		# set title
		plt.title("%s %s" %(stat_name,dm))

		# plot data
		for g in range(len(groups)):
			if stat_name == 'volume':
				slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(stat[['structureID',stat_name]][stat['subjectID'].str.contains('%s_' %str(g+1))].groupby('structureID',as_index=False).mean()[stat_name]),stat[['structureID',dm]][stat['subjectID'].str.contains('%s_' %str(g+1))].groupby('structureID',as_index=False).mean()[dm])
				ax = sns.regplot(x=np.log10(stat[['structureID',stat_name]][stat['subjectID'].str.contains('%s_' %str(g+1))].groupby('structureID',as_index=False).mean()[stat_name]),y=stat[['structureID',dm]][stat['subjectID'].str.contains('%s_' %str(g+1))].groupby('structureID',as_index=False).mean()[dm],color=colors[groups[g]],scatter=True,line_kws={'label':"y={0:.5f}x+{1:.4f}".format(slope,intercept)})

			else:
				slope, intercept, r_value, p_value, std_err = stats.linregress(stat[['structureID',stat_name]][stat['subjectID'].str.contains('%s_' %str(g+1))].groupby('structureID',as_index=False).mean()[stat_name],stat[['structureID',dm]][stat['subjectID'].str.contains('%s_' %str(g+1))].groupby('structureID',as_index=False).mean()[dm])
				ax = sns.regplot(x=stat[['structureID',stat_name]][stat['subjectID'].str.contains('%s_' %str(g+1))].groupby('structureID',as_index=False).mean()[stat_name],y=stat[['structureID',dm]][stat['subjectID'].str.contains('%s_' %str(g+1))].groupby('structureID',as_index=False).mean()[dm],color=colors[groups[g]],scatter=True,line_kws={'label':"y={0:.5f}x+{1:.4f}".format(slope,intercept)})

			ax.legend()
			# if stat_name == 'volume':
			# 	ax.set_xscale('log')
			# 	p.set_xlim([1000,1000000])

		# save or show plot
		if dir_out:
			if not os.path.exists(dir_out):
				os.mkdir(dir_out)

			plt.savefig(os.path.join(dir_out, img_out))
			plt.savefig(os.path.join(dir_out, img_out_png))       
		else:
			plt.show()

# def collisionVNonCollisionTrackScatter(groups,colors,stat,diffusion_measures,dir_out):
# 	import matplotlib.pyplot as plt
# 	import os,sys
# 	import seaborn as sns

# 	for dm in diffusion_measures:
# 		print(dm)
# 		# set up output names
# 		img_out=str('collision_noncollison_track_'+dm+'.eps').replace(' ','_')
# 		img_out_png=str('collision_noncollison_track_'+dm+'.png').replace(' ','_')

# 		# generate figures
# 		fig = plt.figure(figsize=(15,15))
# 		fig.patch.set_visible(False)
# 		p = plt.subplot()

# 		# set spines and ticks
# 		p.spines['right'].set_visible(False)
# 		p.spines['top'].set_visible(False)
# 		# p.spines['left'].set_position(('axes', -0.05))
# 		# p.spines['bottom'].set_position(('axes', -0.05))
# 		p.yaxis.set_ticks_position('left')
# 		p.xaxis.set_ticks_position('bottom')

# 		# set title
# 		plt.title("%s" %(dm))
		
# 		# plot data
# 		ax_ath = sns.regplot(x=stat[[dm,'structureID']][stat['subjectID'].str.contains('%s_' %str(2))].groupby('structureID').mean(),y=stat[[dm,'structureID']][stat['subjectID'].str.contains('%s_' %str(1))].groupby('structureID').mean(),color=colors[groups[1]],scatter=True,fit_reg=False,scatter_kws={'s':100})
# 		ax_fb_na = sns.regplot(x=stat[[dm,'structureID']][stat['subjectID'].str.contains('%s_' %str(3))].groupby('structureID').mean(),y=stat[[dm,'structureID']][stat['subjectID'].str.contains('%s_' %str(1))].groupby('structureID').mean(),color=colors[groups[2]],scatter=True,marker='s',fit_reg=False,scatter_kws={'s':100})

# 		# set axes to be equal
# 		ymin = stat[[dm,'structureID']][stat['subjectID'].str.contains('%s_' %str(1))].groupby('structureID').mean().min().tolist()
# 		ymax = stat[[dm,'structureID']][stat['subjectID'].str.contains('%s_' %str(1))].groupby('structureID').mean().max().tolist()
# 		xmin = min(stat[[dm,'structureID']][stat['subjectID'].str.contains('%s_' %str(2))].groupby('structureID').mean().min().tolist(),stat[[dm,'structureID']][stat['subjectID'].str.contains('%s_' %str(3))].groupby('structureID').mean().min().tolist())
# 		xmax = max(stat[[dm,'structureID']][stat['subjectID'].str.contains('%s_' %str(2))].groupby('structureID').mean().max().tolist(),stat[[dm,'structureID']][stat['subjectID'].str.contains('%s_' %str(3))].groupby('structureID').mean().max().tolist())

# 		if xmin >= ymin:
# 			axes_min = ymin[0] - 0.05
# 		else:
# 			axes_min =  xmin[0] - 0.05

# 		if xmax >= ymax:
# 			axes_max = ymax[0] + 0.05
# 		else:
# 			axes_max =  xmax[0] + 0.05

# 		p.set_xlim([axes_min,axes_max])
# 		p.set_ylim([axes_min,axes_max])

# 		# plot diagonal line
# 		ax_fb_na.plot(ax_fb_na.get_xlim(), ax_fb_na.get_ylim(), ls="--", c=".3") #this is the error line.

# 		# save or show plot
# 		if dir_out:
# 			if not os.path.exists(dir_out):
# 				os.mkdir(dir_out)

# 			plt.savefig(os.path.join(dir_out, img_out))
# 			plt.savefig(os.path.join(dir_out, img_out_png))       
# 		else:
# 			plt.show()

def plotTrackMicrostructureAverage(groups,colors,tracks,stat,diffusion_measures,dir_out):

	import matplotlib.pyplot as plt
	import os,sys
	import seaborn as sns
	from scipy import stats

	if len(diffusion_measures) > 1:
		for dm in diffusion_measures:
			# set up output names
			img_out=str('track_average_'+dm+'.pdf').replace(' ','_')
			img_out_png=str('track_average_'+dm+'.png').replace(' ','_')

			# generate figures
			fig = plt.figure(figsize=(15,15))
			fig.patch.set_visible(False)
			p = plt.subplot()

			# set y range
			p.set_ylim([0,(len(tracks)*len(groups))+3])

			# set spines and ticks
			p.spines['right'].set_visible(False)
			p.spines['top'].set_visible(False)
			# p.spines['left'].set_position(('axes', -0.0))
			# p.spines['bottom'].set_position(('axes', -0.05))
			p.yaxis.set_ticks_position('left')
			p.xaxis.set_ticks_position('bottom')
			p.set_xlabel(dm)
			p.set_ylabel("Tracks")
			p.set_yticks(np.arange((len(groups)-1),(len(tracks)*len(groups)),step=len(groups)))
			p.set_yticklabels(tracks)

			# set title
			plt.title("%s" %(dm))

			for l in range(len(tracks)):
				for g in range(len(groups)):
					p.errorbar(stat[dm][stat['structureID'] == tracks[l]][stat['subjectID'].str.contains('%s_' %str(g+1))].mean(),(3*(l+1)-3)+(g+1),xerr=(stat[dm][stat['structureID'] == tracks[l]][stat['subjectID'].str.contains('%s_' %str(g+1))].std() / np.sqrt(len(stat[stat['subjectID'].str.contains('%s_' %str(g+1))]['subjectID'].unique()))),barsabove=True,ecolor='black',color=colors[groups[g]],marker='o',ms=10)

			# save or show plot
			if dir_out:
				if not os.path.exists(dir_out):
					os.mkdir(dir_out)

				plt.savefig(os.path.join(dir_out, img_out))
				plt.savefig(os.path.join(dir_out, img_out_png))       
			else:
				plt.show()
	else:
			# set up output names
			img_out=str('track_average_'+diffusion_measures+'.pdf').replace(' ','_')
			img_out_png=str('track_average_'+diffusion_measures+'.png').replace(' ','_')

			# generate figures
			fig = plt.figure(figsize=(15,15))
			fig.patch.set_visible(False)
			p = plt.subplot()

			# set y range
			p.set_ylim([0,(len(tracks)*len(groups))+3])

			# set spines and ticks
			p.spines['right'].set_visible(False)
			p.spines['top'].set_visible(False)
			# p.spines['left'].set_position(('axes', -0.0))
			# p.spines['bottom'].set_position(('axes', -0.05))
			p.yaxis.set_ticks_position('left')
			p.xaxis.set_ticks_position('bottom')
			p.set_xlabel(dm)
			p.set_ylabel("Tracks")
			p.set_yticks(np.arange((len(groups)-1),(len(tracks)*len(groups)),step=len(groups)))
			p.set_yticklabels(tracks)

			# set title
			plt.title("%s" %(dm))

			for l in range(len(tracks)):
				for g in range(len(groups)):
					p.errorbar(stat[diffusion_measures][stat['structureID'] == tracks[l]][stat['subjectID'].str.contains('%s_' %str(g+1))].mean(),(3*(l+1)-3)+(g+1),xerr=(stat[dm][stat['structureID'] == tracks[l]][stat['subjectID'].str.contains('%s_' %str(g+1))].std() / np.sqrt(len(stat[stat['subjectID'].str.contains('%s_' %str(g+1))]['subjectID'].unique()))),barsabove=True,ecolor='black',color=colors[groups[g]],marker='o',ms=10)

			# save or show plot
			if dir_out:
				if not os.path.exists(dir_out):
					os.mkdir(dir_out)

				plt.savefig(os.path.join(dir_out, img_out))
				plt.savefig(os.path.join(dir_out, img_out_png))       
			else:
				plt.show()

def plotTrackMicrostructureProfiles(groups,colors,tracks,stat,diffusion_measures,dir_out):

	import matplotlib.pyplot as plt
	import os,sys
	import seaborn as sns
	from scipy import stats

	for t in tracks:
		for dm in diffusion_measures:
			# set up output names
			img_out=str('track_'+t+'_'+dm+'.pdf').replace(' ','_')
			img_out_png=str('track_'+t+'_'+dm+'.png').replace(' ','_')

			# generate figures
			fig = plt.figure(figsize=(15,15))
			fig.patch.set_visible(False)
			p = plt.subplot()

			# set title
			plt.title("%s" %(dm))

			for g in range(len(groups)):
				sns.lineplot(x='nodeID',y=dm,data=stat[['nodeID',dm]][stat['structureID'] == t][stat['subjectID'].str.contains('%s_' %str(g+1))],color=colors[groups[g]],ci="sd")

			# save or show plot
			if dir_out:
				if not os.path.exists(dir_out):
					os.mkdir(dir_out)

				plt.savefig(os.path.join(dir_out, img_out))
				plt.savefig(os.path.join(dir_out, img_out_png))       
			else:
				plt.show()

			plt.close(fig)

def plotDifferenceHistograms(groups,subjects,stat,diffusion_measures,colors,dir_out):
	import matplotlib.pyplot as plt
	import os,sys
	import seaborn as sns
	from itertools import combinations
	import pandas as pd


	img_out = "track_difference_histograms.eps"
	img_out_png = "track_difference_histograms.png"

	f, axes = plt.subplots(2, 4, figsize=(15, 15), sharex=True, sharey=True)
	f.suptitle("Average Difference - Tracts")

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

		axes[row,column].yaxis.set_ticks_position('left')
		axes[row,column].xaxis.set_ticks_position('bottom')
		
		comparison = {}
		g=0
		for compar in comparison_array:
			comparison[compar[0]+"_"+compar[1]] = pd.DataFrame([])
			comparison[compar[0]+"_"+compar[1]] = comparison[compar[0]+"_"+compar[1]].append(stat[['structureID',diffusion_measures[dm]]][stat['subjectID'].isin(subjects[compar[0]])].groupby('structureID').mean() - stat[['structureID',diffusion_measures[dm]]][stat['subjectID'].isin(subjects[compar[1]])].groupby('structureID').mean(),ignore_index=True)
			if g == 1:
				Colors = colors[compar[1]]
			else:
				Colors = colors[compar[0]]
			
			sns.distplot(comparison[compar[0]+"_"+compar[1]],kde=True,hist=False,color=Colors,bins=20,ax=axes[row,column],axlabel=diffusion_measures[dm],label="mean: %.3f" %np.mean(comparison[compar[0]+"_"+compar[1]]))
			g=g+1
		
		axes[row,column].axvline(x=0, color='black', linestyle='--', linewidth=3)
		axes[row,column].legend()
	
	# save or show plot
	if dir_out:
		if not os.path.exists(dir_out):
			os.mkdir(dir_out)

		plt.savefig(os.path.join(dir_out, img_out))
		plt.savefig(os.path.join(dir_out, img_out_png))       
	else:
		plt.show()

def plotBootstrappedDifference(groups,subjects,stat,diffusion_measures,colors,repetitions,alpha,dir_out,data_dir):
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
			comparison[compar[0]+"_"+compar[1]].to_csv(os.path.join(data_dir,'track_boostrap_difference_data_'+compar[0]+"_"+compar[1]+"_"+diffusion_measures[dm]+".csv"),index=False)
			cli[compar[0]+"_"+compar[1]] = [np.percentile(comparison[compar[0]+"_"+compar[1]],alpha/2*100),np.percentile(comparison[compar[0]+"_"+compar[1]],100-alpha/2*100),]
			if g == 1:
				Colors = colors[compar[1]]
			else:
				Colors = colors[compar[0]]
			
			sns.distplot(comparison[compar[0]+"_"+compar[1]],kde=True,hist=False,color=Colors,bins=20,ax=axes[row,column],axlabel=diffusion_measures[dm],label="mean: %.3f; cli: %.3f %.3f" %(np.mean(comparison[compar[0]+"_"+compar[1]]),cli[compar[0]+"_"+compar[1]][0],cli[compar[0]+"_"+compar[1]][1]))
			axes[row,column].axvline(cli[compar[0]+"_"+compar[1]][0],color=Colors,linestyle='--',linewidth=3)
			axes[row,column].axvline(cli[compar[0]+"_"+compar[1]][1],color=Colors,linestyle='--',linewidth=3)
			g=g+1

		axes[row,column].legend()


	# save or show plot
	if dir_out:
		if not os.path.exists(dir_out):
			os.mkdir(dir_out)

		plt.savefig(os.path.join(dir_out, img_out))
		plt.savefig(os.path.join(dir_out, img_out_png))       
	else:
		plt.show()

def plotBootstrappedH0TrackAverageDifference(groups,subjects,stat,diffusion_measures,colors,repetitions,dir_out):
	import matplotlib.pyplot as plt
	import os,sys
	import seaborn as sns
	import pandas as pd
	import numpy as np
	from itertools import combinations

	img_out = "track_h0_bootstrap_histograms.eps"
	img_out_png = "track_h0_bootstrap_histograms.png"

	f, axes = plt.subplots(2, 4, figsize=(15, 15), sharex=True, sharey=True)
	f.suptitle("Average Bootstrap (H0) Difference - Tracts")

	comparison_array = list(combinations(groups,2)) # 3 x 2 array; 3 different comparisons, with two pairs per comparison. comparison_array[0] = ("football","cross_country")

	for dm in range(len(diffusion_measures)): # loop through all diffusion measures
		if diffusion_measures[dm] in ['ad','fa','md','rd']:
			row = 1
			column = dm
		else:
			row = 0
			column = dm - 4

		# set spines and ticks for plot
		axes[row,column].spines['right'].set_visible(False)
		axes[row,column].spines['top'].set_visible(False)
		axes[row,column].yaxis.set_ticks_position('left')
		axes[row,column].xaxis.set_ticks_position('bottom')

		axes[row,column].patch.set_visible(False)

		# set up empty data structures
		comparison = {} # for pooled track averages (H0)
		g_diff = {} # H0 group differences array 
		p_value = {} # pvalue array
		g=0 # groups counter to help pick colors so i don't have to range(len()) the comparison array
		for compar in comparison_array:
			# data structure set up
			comparison[compar[0]+"_"+compar[1]] = pd.DataFrame([]) # set up dataframe for each comparison
			comparison[compar[0]+"_"+compar[1]] = pd.concat([stat[['structureID',diffusion_measures[dm]]][stat['subjectID'].isin(subjects[compar[0]])].groupby('structureID').mean(),stat[['structureID',diffusion_measures[dm]]][stat['subjectID'].isin(subjects[compar[1]])].groupby('structureID').mean()],ignore_index=True) # pooled group average track data
			g_diff[compar[0]+"_"+compar[1]] = [] # set up dataframe for each comparison
			p_value[compar[0]+"_"+compar[1]] = [] # set up dataframe for each comparison
			
			# bootstrapping
			for i in range(repetitions): # loop through repetitions
				g_diff[compar[0]+"_"+compar[1]].append(np.mean(np.random.choice(list(comparison[compar[0]+"_"+compar[1]][diffusion_measures[dm]]),int(len(comparison[compar[0]+"_"+compar[1]][diffusion_measures[dm]])/2),replace=True)) \
					- np.mean(np.random.choice(list(comparison[compar[0]+"_"+compar[1]][diffusion_measures[dm]]),int(len(comparison[compar[0]+"_"+compar[1]][diffusion_measures[dm]])/2),replace=True))) # randomly select from pooled dataframe, and compute difference
			actual_difference = np.mean(stat[['structureID',diffusion_measures[dm]]][stat['subjectID'].isin(subjects[compar[0]])].groupby('structureID').mean()[diffusion_measures[dm]]) - np.mean(stat[['structureID',diffusion_measures[dm]]][stat['subjectID'].isin(subjects[compar[1]])].groupby('structureID').mean()[diffusion_measures[dm]]) # calculate actual difference between groups
			
			# calculate p-value
			p_value[compar[0]+"_"+compar[1]] = sum(np.abs(g_diff[compar[0]+"_"+compar[1]]) >= np.abs(actual_difference)) / repetitions # compute p-value
			
			# finish plotting data
			if g == 1:
				Colors = colors[compar[1]]
			else:
				Colors = colors[compar[0]]
			
			sns.distplot(g_diff[compar[0]+"_"+compar[1]],kde=True,hist=False,color=Colors,bins=20,ax=axes[row,column],axlabel=diffusion_measures[dm],label="p_value: %.6f" %p_value[compar[0]+"_"+compar[1]])
			axes[row,column].axvline(actual_difference,color=Colors,linestyle='--',linewidth=3)
			g=g+1

		# print out legend to finish plot for diffusion measures dm
		axes[row,column].legend()

	# save or show plot
	if dir_out:
		if not os.path.exists(dir_out):
			os.mkdir(dir_out)

		plt.savefig(os.path.join(dir_out, img_out))
		plt.savefig(os.path.join(dir_out, img_out_png))       
	else:
		plt.show()