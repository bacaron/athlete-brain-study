#!/bin/env python3

import os,sys
import numpy as np
from os import listdir
from os.path import isfile, join, isdir
import json
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors as mcolors
import glob

def load_snr_stat(snr_json):

	dim = len(snr_json["SNR in b0, X, Y, Z"])
	snr_data = np.zeros(dim)
	for i in range(dim):
	    snr_data[i] = float(snr_json["SNR in b0, X, Y, Z"][i])
	    
	return snr_data

def plot_snr_set(all_stat,all_sub,dir_out):

	all_sub.append("")
	all_sub.reverse()
	all_stat.reverse()
	fig = plt.figure(figsize=(10,10))
	p = plt.subplot()
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
	color = mcolors.cnames['blue']
	plt.title("SNR of diffusion signal")
	for s in range(len(all_stat)):
	    stat = all_stat[s]
	    if all_sub[s].split("_")[0] == '1':
	    	color = mcolors.cnames['orange']
	    elif all_sub[s].split("_")[0] == '2':
	    	color = mcolors.cnames['pink']
	    elif all_sub[s].split("_")[0] == '3':
	    	color = mcolors.cnames['blue']

	    p.errorbar(stat[1:].mean(), s+1, xerr=stat[1:].std(), marker='o', linestyle='None', color=color)
	    p.plot(stat[0], s+1, marker='x', color=color)

	#plt.show()
	snr_out = "snr_allsubjs.eps"
	snr_out_png = "snr_allsubjs.png"
	plt.savefig(os.path.join(dir_out, snr_out))
	plt.savefig(os.path.join(dir_out, snr_out_png))

def compute_plot_snr():

	topPath = sys.argv[1]
	subjects = [ f  for f in glob.glob(os.path.join(topPath+'*')) if f != './img' if isdir(os.path.join(topPath,f)) if f.split('_')[0] != './' ] 
	subjects = [ f.split('./')[1] for f in subjects ]
	subjects.sort()
	snr = []

	for subjs in range(0,len(subjects)):
		with open(topPath+subjects[subjs]+'/snr/output/snr.json') as config_f:
			config = json.load(config_f)
		snr.append(load_snr_stat(config))

	#os.mkdir("img")
	
	plot_snr_set(snr,subjects,dir_out="img")

if __name__ == "__main__":
	compute_plot_snr()
