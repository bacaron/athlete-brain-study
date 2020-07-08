#!/usr/bin/env python

import os,sys
import numpy as np

def plotWholeBrainData(groups,colors,stat_name,stat,dir_out):
	import matplotlib.pyplot as plt
	import os,sys

	# set up output names
	img_out=str(stat_name+'.eps').replace(' ','_')
	img_out_png=str(stat_name+'.png').replace(' ','_')

	# generate figures
	fig = plt.figure(figsize=(15,15))
	fig.patch.set_visible(False)
	p = plt.subplot()

	# set x axis limits & out names
	p.set_xlim([0.5,len(groups)+0.5])

	# set spines and ticks
	p.spines['right'].set_visible(False)
	p.spines['top'].set_visible(False)
	p.spines['left'].set_position(('axes', -0.05))
	p.spines['bottom'].set_position(('axes', -0.05))
	p.xaxis.set_ticks(range(1,len(groups)+1))
	p.yaxis.set_ticks_position('left')
	p.xaxis.set_ticks_position('bottom')
	p.set_xticklabels(groups)
	p.set_ylabel(stat_name)
	p.set_xlabel('Groups')

	# set title
	plt.title("%s" %stat_name)

	# plot data
	for g in range(len(groups)):
		p.scatter(np.ones(len(stat[stat_name][stat['subjectID'].str.contains('%s_' %str(g+1))]))*(g+1),stat[stat_name][stat['subjectID'].str.contains('%s_' %str(g+1))],s=100,c=colors[groups[g]],edgecolors='black')

	# save or show plot
	if dir_out:
		if not os.path.exists(dir_out):
			os.mkdir(dir_out)

		plt.savefig(os.path.join(dir_out, img_out))
		plt.savefig(os.path.join(dir_out, img_out_png))       
	else:
		plt.show()

def plotCorticalParcelData(groups,colors,stat_name,stat,diffusion_measures,dir_out):
	import matplotlib.pyplot as plt
	import os,sys
	import seaborn as sns
	from scipy import stats

	for dm in diffusion_measures:
		# set up output names
		img_out=str('aparc_'+stat_name+'_'+dm+'.pdf').replace(' ','_')
		img_out_png=str('aparc_'+stat_name+'_'+dm+'.png').replace(' ','_')

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
			slope, intercept, r_value, p_value, std_err = stats.linregress(stat[['structureID',stat_name]][stat['subjectID'].str.contains('%s_' %str(g+1))].groupby('structureID',as_index=False).mean()[stat_name],stat[['structureID',dm]][stat['subjectID'].str.contains('%s_' %str(g+1))].groupby('structureID',as_index=False).mean()[dm])
			ax = sns.regplot(x=stat[['structureID',stat_name]][stat['subjectID'].str.contains('%s_' %str(g+1))].groupby('structureID',as_index=False).mean()[stat_name],y=stat[['structureID',dm]][stat['subjectID'].str.contains('%s_' %str(g+1))].groupby('structureID',as_index=False).mean()[dm],color=colors[groups[g]],scatter=True,line_kws={'label':"y={0:.5f}x+{1:.4f}".format(slope,intercept)})
			ax.legend()
			if stat_name == 'volume':
				ax.set_xscale('log')
				p.set_xlim([100,100000])

		# save or show plot
		if dir_out:
			if not os.path.exists(dir_out):
				os.mkdir(dir_out)

			plt.savefig(os.path.join(dir_out, img_out))
			plt.savefig(os.path.join(dir_out, img_out_png))       
		else:
			plt.show()

def collisionVNonCollisionParcelScatter(groups,colors,stat,stat_subcort,diffusion_measures,dir_out):
	import matplotlib.pyplot as plt
	import os,sys
	import seaborn as sns

	for dm in diffusion_measures:
		print(dm)
		# set up output names
		img_out=str('collision_noncollison_cortex_'+dm+'.eps').replace(' ','_')
		img_out_png=str('collision_noncollison_cortex_'+dm+'.png').replace(' ','_')

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
		plt.title("%s" %(dm))
		
		# plot data
		ax_ath = sns.regplot(x=stat[[dm,'structureID']][stat['subjectID'].str.contains('%s_' %str(2))].groupby('structureID').mean(),y=stat[[dm,'structureID']][stat['subjectID'].str.contains('%s_' %str(1))].groupby('structureID').mean(),color=colors[groups[1]],scatter=True,fit_reg=False,scatter_kws={'s':100})
		ax_fb_na = sns.regplot(x=stat[[dm,'structureID']][stat['subjectID'].str.contains('%s_' %str(3))].groupby('structureID').mean(),y=stat[[dm,'structureID']][stat['subjectID'].str.contains('%s_' %str(1))].groupby('structureID').mean(),color=colors[groups[2]],scatter=True,marker='s',fit_reg=False,scatter_kws={'s':100})
		ax_ath_subcort = plt.scatter(stat_subcort[[dm,'structureID']][stat_subcort['subjectID'].str.contains('%s_' %str(2))].groupby('structureID').mean()[dm].tolist(),stat_subcort[[dm,'structureID']][stat_subcort['subjectID'].str.contains('%s_' %str(1))].groupby('structureID').mean()[dm].tolist(),s=100,c=colors[groups[1]],edgecolors='green',linewidths=2)
		ax_fb_na_subcort = plt.scatter(stat_subcort[[dm,'structureID']][stat_subcort['subjectID'].str.contains('%s_' %str(3))].groupby('structureID').mean()[dm].tolist(),stat_subcort[[dm,'structureID']][stat_subcort['subjectID'].str.contains('%s_' %str(1))].groupby('structureID').mean()[dm].tolist(),s=100,c=colors[groups[2]],edgecolors='green',linewidths=2,marker='s')

		# set axes to be equal
		ymin = stat[[dm,'structureID']][stat['subjectID'].str.contains('%s_' %str(1))].groupby('structureID').mean().min().tolist()
		ymax = stat[[dm,'structureID']][stat['subjectID'].str.contains('%s_' %str(1))].groupby('structureID').mean().max().tolist()
		xmin = min(stat[[dm,'structureID']][stat['subjectID'].str.contains('%s_' %str(2))].groupby('structureID').mean().min().tolist(),stat[[dm,'structureID']][stat['subjectID'].str.contains('%s_' %str(3))].groupby('structureID').mean().min().tolist())
		xmax = max(stat[[dm,'structureID']][stat['subjectID'].str.contains('%s_' %str(2))].groupby('structureID').mean().max().tolist(),stat[[dm,'structureID']][stat['subjectID'].str.contains('%s_' %str(3))].groupby('structureID').mean().max().tolist())

		subcort_ymin = stat_subcort[[dm,'structureID']][stat_subcort['subjectID'].str.contains('%s_' %str(1))].groupby('structureID').mean().min().tolist()
		subcort_ymax = stat_subcort[[dm,'structureID']][stat_subcort['subjectID'].str.contains('%s_' %str(1))].groupby('structureID').mean().max().tolist()
		subcort_xmin = min(stat_subcort[[dm,'structureID']][stat_subcort['subjectID'].str.contains('%s_' %str(2))].groupby('structureID').mean().max().tolist(),stat_subcort[[dm,'structureID']][stat_subcort['subjectID'].str.contains('%s_' %str(3))].groupby('structureID').mean().max().tolist())
		subcort_xmax = max(stat_subcort[[dm,'structureID']][stat_subcort['subjectID'].str.contains('%s_' %str(2))].groupby('structureID').mean().max().tolist(),stat_subcort[[dm,'structureID']][stat_subcort['subjectID'].str.contains('%s_' %str(3))].groupby('structureID').mean().max().tolist())

		if xmin >= ymin:
			axes_min = ymin[0] - 0.05
		else:
			axes_min =  xmin[0] - 0.05

		if xmax >= ymax:
			axes_max = ymax[0] + 0.05
		else:
			axes_max =  xmax[0] + 0.05

		if axes_min > np.min([subcort_xmin,subcort_ymin]):
			axes_min = np.min([subcort_xmin,subcort_ymin]) - 0.05

		if axes_max < np.max([subcort_xmax,subcort_ymax]):
			axes_max = np.max([subcort_xmax,subcort_ymax]) + 0.05

		p.set_xlim([axes_min,axes_max])
		p.set_ylim([axes_min,axes_max])

		
		# plot diagonal line
		ax_fb_na.plot(ax_fb_na.get_xlim(), ax_fb_na.get_ylim(), ls="--", c=".3") #this is the error line.

		# save or show plot
		if dir_out:
			if not os.path.exists(dir_out):
				os.mkdir(dir_out)

			plt.savefig(os.path.join(dir_out, img_out))
			plt.savefig(os.path.join(dir_out, img_out_png))       
		else:
			plt.show()

def plotLobeMicrostructureAverage(groups,colors,lobes,stat,diffusion_measures,dir_out):
	import matplotlib.pyplot as plt
	import os,sys
	import seaborn as sns
	from scipy import stats

	for dm in diffusion_measures:
		# set up output names
		img_out=str('lobes_'+dm+'.pdf').replace(' ','_')
		img_out_png=str('lobes_'+dm+'.png').replace(' ','_')

		# generate figures
		fig = plt.figure(figsize=(15,15))
		fig.patch.set_visible(False)
		p = plt.subplot()

		# set y range
		p.set_ylim([0,(len(lobes)*len(groups))+1])

		# set spines and ticks
		p.spines['right'].set_visible(False)
		p.spines['top'].set_visible(False)
		p.spines['left'].set_position(('axes', -0.05))
		p.spines['bottom'].set_position(('axes', -0.05))
		p.yaxis.set_ticks_position('left')
		p.xaxis.set_ticks_position('bottom')
		p.set_xlabel(dm)
		p.set_ylabel("Lobes")
		p.set_yticks(np.arange((len(groups)-1),(len(lobes)*len(groups)),step=len(groups)))
		p.set_yticklabels(lobes)

		# set title
		plt.title("%s" %(dm))

		for l in range(len(lobes)):
			for g in range(len(groups)):
				p.errorbar(stat[dm][stat['structureID'] == lobes[l]][stat['subjectID'].str.contains('%s_' %str(g+1))].mean(),(3*(l+1)-3)+(g+1),xerr=(stat[dm][stat['structureID'] == lobes[l]][stat['subjectID'].str.contains('%s_' %str(g+1))].std() / np.sqrt(len(np.unique(stat['subjectID'])) - 1)),color=colors[groups[g]],marker='o',ms=25)

		# save or show plot
		if dir_out:
			if not os.path.exists(dir_out):
				os.mkdir(dir_out)

			plt.savefig(os.path.join(dir_out, img_out))
			plt.savefig(os.path.join(dir_out, img_out_png))       
		else:
			plt.show()
