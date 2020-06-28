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
			slope, intercept, r_value, p_value, std_err = stats.linregress(stat[['structureID',stat_name]][stat['subjectID'].str.contains('%s_' %str(g+1))].groupby('structureID',as_index=False).mean()[stat_name],stat[['structureID',dm]][stat['subjectID'].str.contains('%s_' %str(g+1))].groupby('structureID',as_index=False).mean()[dm])
			ax = sns.regplot(x=stat[['structureID',stat_name]][stat['subjectID'].str.contains('%s_' %str(g+1))].groupby('structureID',as_index=False).mean()[stat_name],y=stat[['structureID',dm]][stat['subjectID'].str.contains('%s_' %str(g+1))].groupby('structureID',as_index=False).mean()[dm],color=colors[groups[g]],scatter=True,line_kws={'label':"y={0:.5f}x+{1:.4f}".format(slope,intercept)})
			ax.legend()
			if stat_name == 'volume':
				ax.set_xscale('log')
				p.set_xlim([1000,1000000])

		# save or show plot
		if dir_out:
			if not os.path.exists(dir_out):
				os.mkdir(dir_out)

			plt.savefig(os.path.join(dir_out, img_out))
			plt.savefig(os.path.join(dir_out, img_out_png))       
		else:
			plt.show()

def collisionVNonCollisionTrackScatter(groups,colors,stat,diffusion_measures,dir_out):
	import matplotlib.pyplot as plt
	import os,sys
	import seaborn as sns

	for dm in diffusion_measures:
		print(dm)
		# set up output names
		img_out=str('collision_noncollison_track_'+dm+'.eps').replace(' ','_')
		img_out_png=str('collision_noncollison_track_'+dm+'.png').replace(' ','_')

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

		# set axes to be equal
		ymin = stat[[dm,'structureID']][stat['subjectID'].str.contains('%s_' %str(1))].groupby('structureID').mean().min().tolist()
		ymax = stat[[dm,'structureID']][stat['subjectID'].str.contains('%s_' %str(1))].groupby('structureID').mean().max().tolist()
		xmin = min(stat[[dm,'structureID']][stat['subjectID'].str.contains('%s_' %str(2))].groupby('structureID').mean().min().tolist(),stat[[dm,'structureID']][stat['subjectID'].str.contains('%s_' %str(3))].groupby('structureID').mean().min().tolist())
		xmax = max(stat[[dm,'structureID']][stat['subjectID'].str.contains('%s_' %str(2))].groupby('structureID').mean().max().tolist(),stat[[dm,'structureID']][stat['subjectID'].str.contains('%s_' %str(3))].groupby('structureID').mean().max().tolist())

		if xmin >= ymin:
			axes_min = ymin[0] - 0.05
		else:
			axes_min =  xmin[0] - 0.05

		if xmax >= ymax:
			axes_max = ymax[0] + 0.05
		else:
			axes_max =  xmax[0] + 0.05

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

def plotTrackMicrostructureAverage(groups,colors,tracks,stat,diffusion_measures,dir_out):

	import matplotlib.pyplot as plt
	import os,sys
	import seaborn as sns
	from scipy import stats

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
		p.set_ylabel("Lobes")
		p.set_yticks(np.arange((len(groups)-1),(len(tracks)*len(groups)),step=len(groups)))
		p.set_yticklabels(tracks)

		# set title
		plt.title("%s" %(dm))

		for l in range(len(tracks)):
			for g in range(len(groups)):
				p.errorbar(stat[dm][stat['structureID'] == tracks[l]][stat['subjectID'].str.contains('%s_' %str(g+1))].mean(),(3*(l+1)-3)+(g+1),xerr=(stat[dm][stat['structureID'] == tracks[l]][stat['subjectID'].str.contains('%s_' %str(g+1))].std() / np.sqrt(len(np.unique(stat['subjectID'])) - 1)),color=colors[groups[g]],marker='o',ms=10)

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

			# # set spines and ticks
			# p.spines['right'].set_visible(False)
			# p.spines['top'].set_visible(False)
			# # p.spines['left'].set_position(('axes', -0.0))
			# # p.spines['bottom'].set_position(('axes', -0.05))
			# p.yaxis.set_ticks_position('left')
			# p.xaxis.set_ticks_position('bottom')
			# p.set_xlabel(dm)
			# p.set_ylabel("Lobes")
			# p.set_yticks(np.arange((len(groups)-1),(len(tracks)*len(groups)),step=len(groups)))
			# p.set_yticklabels(tracks)

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

# if __name__ == '__main__':
# 	plotTractMacro(sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
