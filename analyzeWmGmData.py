#!/usr/bin/env python3

import os,sys,glob
from matplotlib import colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

### setting up variables and adding paths
print("setting up variables")
topPath = "/media/brad/APPS/athlete-updated-pipeline-qc"
os.chdir(topPath)
scripts_dir = topPath+'/athlete_brain_study/'
data_dir = topPath+'/data/'
img_dir = topPath+'/img/'
if not os.path.exists(img_dir):
	os.mkdir(img_dir)
sys.path.insert(0,scripts_dir)
sys.path.insert(1,topPath+'/athlete_brain_study/utils')
sys.path.insert(2,topPath+'/athlete_brain_study/gray-matter-parcels')
sys.path.insert(2,topPath+'/athlete_brain_study/white-matter-tracks')

groups = ['football','cross_country','non_athlete']
colors_array = ['orange','pink','blue']
diff_measures = ['ad','fa','md','rd','ndi','isovf','odi']
lobes = ['frontal','temporal','occipital','parietal','insular','limbic','motor','somatosensory']

colors = {}
subjects = {}

img_dir = topPath + '/img/'
data_dir = topPath + '/data/'
lobes_dir = topPath + '/athlete_brain_study/configs/'

# loop through groups and identify subjects and set color schema for each group
for g in range(len(groups)):
	# set subjects array
	subjects[groups[g]] =  [f.split(topPath+'/')[1] for f in glob.glob(topPath+'/*'+str(g+1)+'_0*')]
	subjects[groups[g]].sort()

	# set colors array
	colors_name = colors_array[g]
	colors[groups[g]] = colors_array[g]
print("setting up variables complete")

### create subjects.csv
print("creating subjects.csv")
from compile_data import collectSubjectData
subjects_data = collectSubjectData(topPath,data_dir,groups,subjects,colors)
print("creating subjects.csv complete")

### generate snr plot
print("plotting snr data")
# grab data
from compile_data import collectSNRData
snr = collectSNRData(topPath,data_dir,groups,subjects)

# plot data
from plot_track_data import plotSNR
plotSNR(list(snr['snr']),list(snr['subjectID']),list(subjects_data['colors']),dir_out=topPath+"/img/")
print("plotting snr data complete")

### generate wholebrain plots
print("plotting whole brain stats")
# grab data
from compile_data import collectWholeBrainStats
wholebrain = collectWholeBrainStats(topPath,data_dir,groups,subjects)

# plot data
from plot_cortex_data import plotWholeBrainData
for dc in ['subjectID','Total Brain Volume','Total Cortical Gray Matter Volume','Total White Matter Volume','Total Cortical Thickness']:
	plotWholeBrainData(groups,colors,dc,wholebrain,dir_out=topPath+"/img/")
print("plotting whole brain stats complete")

### group average white matter analyses
print("computing group average white matter track analyses")
## create data structures
# macro measures
from compile_data import collectTrackMacroData
[track_names,track_macro] = collectTrackMacroData(topPath,data_dir,groups,subjects)

# micro measures
from compile_data import collectTrackMicroData
track_micro =  collectTrackMicroData(topPath,data_dir,groups,subjects,180)

# combine the two
from compile_data import combineTrackMacroMicro
[track_data,track_mean_data] = combineTrackMacroMicro(data_dir,track_macro[track_macro['structureID'] != 'wbfg'],track_micro)

## length, volume, streamline count of tracks
from plot_track_data import plotTrackMacroData
for dc in ['volume','length','count']:
	plotTrackMacroData(groups,colors,dc,track_mean_data,diff_measures,dir_out=topPath+'/img/')

## DTI/NODDI tract profiles
from plot_track_data import plotTrackMicrostructureProfiles
plotTrackMicrostructureProfiles(groups,colors,track_names,track_data,diff_measures,dir_out=topPath+'/img/')

## collision vs non-collision scatter
from plot_track_data import collisionVNonCollisionTrackScatter
collisionVNonCollisionTrackScatter(groups,colors,track_mean_data,diff_measures,dir_out=topPath+'/img/')

## DTI/NODDI scatter plots (group averages)
from plot_track_data import plotTrackMicrostructureAverage
plotTrackMicrostructureAverage(groups,colors,track_names,track_mean_data,diff_measures,dir_out=topPath+'/img/')

print("computing group average white matter track analyses complete")

### group average cortex mapping analyses
print("computing group average gray matter parcel analyses")
## create data structures
# cortical measures
from compile_data import collectCorticalParcelData
cortical = collectCorticalParcelData(topPath,data_dir,groups,subjects)

# subcortical measures
from compile_data import collectSubCorticalParcelData
subcortical =  collectSubCorticalParcelData(topPath,data_dir,groups,subjects)

# combine the two
from compile_data import combineCorticalSubcortical
[graymatter_names,graymatter] = combineCorticalSubcortical(data_dir,cortical,subcortical)

# lobe-specific measures
from compile_data import compileLobeData
[lobe_data,lobe_data_mean] = compileLobeData(data_dir,cortical,lobes,labelsPath=scripts_dir+'/configs/')

## volume, cortical thickness analyses
# cortical thickness/volume by diffusion measure per cortical parcel
from plot_cortex_data import plotCorticalParcelData
for dc in ['volume','thickness']:
	plotCorticalParcelData(groups,colors,dc,cortical,diff_measures,dir_out=topPath+'/img/')

## collision vs non-collision scatter
from plot_cortex_data import collisionVNonCollisionParcelScatter
collisionVNonCollisionParcelScatter(groups,colors,cortical,subcortical,diff_measures,dir_out=topPath+'/img/')

## lobe averages
from plot_cortex_data import plotLobeMicrostructureAverage
plotLobeMicrostructureAverage(groups,colors,lobes,lobe_data,diff_measures,dir_out=topPath+'/img/')
print("computing group average gray matter parcel analyses")

print("project data has been generated and plotted. time for machine learning!")



