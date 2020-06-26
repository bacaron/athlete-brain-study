#!/usr/bin/env python

import os,sys
import numpy as numpy

def plotTrackMacro(groups,colors,stat_name,stat,tract_names,dir_out):
	import matplotlib.pyplot as plt
	import os,sys

	# generate figures
	fig = plt.figure(figsize=(10,10))
	p = plt.subplot()

	# set x axis limits & out names
	if stat_name == 'Streamline Counts':
		xmin = 1
		xmax = 100000
		img_out = 'streamline_count.eps'
		img_out_png = 'streamline_count.png'
		logscale = True
	elif stat_name == 'Volume':
		xmin = 1000
		xmax = 100000
		img_out = 'volume.eps'
		img_out_png = 'volume.png'
		logscale = True
	elif stat_name == 'Streamline Length':
		xmin = 1
		xmax = 200
		img_out = 'streamline_length.eps'
		img_out_png = 'streamline_length.png'
		logscale = False
	elif stat_name == 'AD':
		xmin = 1
		xmax = 200
		img_out = 'streamline_length.eps'
		img_out_png = 'streamline_length.png'
		logscale = False
	else:
		xmin = 0
		xmax = 1
		img_out = 'streamline_length.eps'
		img_out_png = 'streamline_length.png'
		logscale = False

	# set log scale if logscale == true
	if logscale == True:
		plt.xscale("log",basex=10)
		xlabel="Log %s" %stat_name
	else:
		xlabel="%s" %stat_name

	# set y axis limits
	y_axis_ticks = list(range(2,len(groups)*len(tract_names)+1,3))
	ymin = 0
	ymax = len(groups)*len(tract_names)+1

	## start plotting
	# set x and y axes on plot
	p.set_xlim([xmin, xmax])
	p.set_ylim([ymin, ymax])

	# set spines and ticks
	p.spines['right'].set_visible(False)
	p.spines['top'].set_visible(False)
	p.spines['left'].set_position(('axes', -0.05))
	p.spines['bottom'].set_position(('axes', -0.05))
	p.yaxis.set_ticks(y_axis_ticks)
	p.yaxis.set_ticks_position('left')
	p.xaxis.set_ticks_position('bottom')
	p.set_yticklabels(tract_names)
	p.set_ylabel('Tracts')
	p.set_xlabel('%s' %xlabel)

	# set title
	plt.title("%s" %stat_name)

	# plot data
	for g in range(len(groups)):
		y_axis = list(range(g+1,3*len(tract_names)+1,3))
		p.errorbar(stat['mean'][g],y_axis,xerr=stat['sd'][g],
			marker='o',linestyle='None',color=colors['%s' %groups[g]])

	# save or show plot
	if dir_out:
		plt.savefig(os.path.join(dir_out, img_out))
		plt.savefig(os.path.join(dir_out, img_out_png))       
	else:
		plt.show()

def plotTrackMicro(groups,colors,measure_name,stat,tract_names,dir_out):
	import matplotlib.pyplot as plt
	import os,sys

	# generate figures
	fig = plt.figure(figsize=(10,10))
	p = plt.subplot()

	# set x axis and y axis ticks
	xmin = 0
	xmax = 180
	x_axis_ticks = list(range(xmin,xmax,20))

	# set y axis and y axis ticks
	if measure_name in ['ad','md','rd']:
		# img_out = 'streamline_count.eps'
		# img_out_png = 'streamline_count.png'
		ymin = 0.2
		ymax = 2.2
		y_axis_ticks = list(range(ymin,ymax,0.2))
	else measure_name in ['fa','ndi','isovf','odi']:
		ymin = 0
		ymax = 1

	# set x- & y-label name
	xlabel="Position along track"
	ylabel="%s" %measure_name

	## start plotting
	# set x and y axes on plot
	p.set_xlim([xmin, xmax])
	p.set_ylim([ymin, ymax])

	# set spines and ticks
	p.spines['right'].set_visible(False)
	p.spines['top'].set_visible(False)
	p.spines['left'].set_position(('axes', -0.05))
	p.spines['bottom'].set_position(('axes', -0.05))
	p.yaxis.set_ticks(y_axis_ticks)
	p.yaxis.set_ticks_position('left')
	p.xaxis.set_ticks_position('bottom')
	p.set_yticklabels(tract_names)
	p.set_ylabel('Tracts')
	p.set_xlabel('%s' %xlabel)

	# set title
	plt.title("%s" %stat_name)

	# plot data
	for g in range(len(groups)):
		y_axis = list(range(g+1,3*len(tract_names)+1,3))
		p.errorbar(stat['mean'][g],y_axis,xerr=stat['sd'][g],
			marker='o',linestyle='None',color=colors['%s' %groups[g]])

	# save or show plot
	if dir_out:
		plt.savefig(os.path.join(dir_out, img_out))
		plt.savefig(os.path.join(dir_out, img_out_png))       
	else:
		plt.show()


if __name__ == '__main__':
	plotTractMacro(sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
