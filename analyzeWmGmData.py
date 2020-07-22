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
utils_dir = scripts_dir+'/utils/'
data_dir = topPath+'/data/'
if not os.path.exists(data_dir):
	os.mkdir(data_dir)
img_dir = topPath+'/img/'
if not os.path.exists(img_dir):
	os.mkdir(img_dir)
sys.path.insert(0,scripts_dir)
sys.path.insert(1,utils_dir)

groups = ['football','cross_country','non_athlete']
colors_array = ['orange','pink','blue']
diff_measures = ['ad','fa','md','rd','ndi','isovf','odi']
functional_tracks = ['association','projection','commissural']
lobes = ['frontal','temporal','occipital','parietal','insular','limbic','motor','somatosensory']
configs_dir = topPath + '/athlete_brain_study/configs/'

colors = {}
subjects = {}

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
plotSNR(list(snr['snr']),list(snr['subjectID']),list(subjects_data['colors']),dir_out=img_dir)
print("plotting snr data complete")

### generate wholebrain plots
print("plotting whole brain stats")
# grab data
from compile_data import collectWholeBrainStats
wholebrain = collectWholeBrainStats(topPath,data_dir,groups,subjects)

# plot data
from plot_cortex_data import plotWholeBrainData
for dc in ['subjectID','Total Brain Volume','Total Cortical Gray Matter Volume','Total White Matter Volume','Total Cortical Thickness']:
	plotWholeBrainData(groups,colors,dc,wholebrain,dir_out=img_dir)
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

# functional-specific track measures
from compile_data import compileFunctionalData
functional_track_data = compileFunctionalData(data_dir,track_mean_data,functional_tracks,labelsPath=configs_dir)

## length, volume, streamline count of tracks
from plot_track_data import plotTrackMacroData
for dc in ['volume','length','count']:
	plotTrackMacroData(groups,colors,dc,track_mean_data,diff_measures,dir_out=img_dir)

## DTI/NODDI tract profiles
from plot_track_data import plotTrackMicrostructureProfiles
plotTrackMicrostructureProfiles(groups,colors,track_names,track_data,diff_measures,dir_out=img_dir)

# ## collision vs non-collision scatter
# from plot_track_data import collisionVNonCollisionTrackScatter
# collisionVNonCollisionTrackScatter(groups,colors,track_mean_data,diff_measures,dir_out=img_dir)

## DTI/NODDI scatter plots (group averages)
from compile_data import computeRankOrderEffectSize
from plot_track_data import plotTrackMicrostructureAverage
rank_order_tracks = computeRankOrderEffectSize(groups,subjects,'tracks',measures,track_mean,[diff_measures[0:4],diff_measures[4:]],data_dir)
plotTrackMicrostructureAverage(groups,colors,rank_order_tracks['tensor'],track_mean_data,diff_measures[0:4],dir_out=img_dir)
plotTrackMicrostructureAverage(groups,colors,rank_order_tracks['noddi'],track_mean_data,diff_measures[4:],dir_out=img_dir)

## group difference histograms
from plot_track_data import plotDifferenceHistograms
plotDifferenceHistograms(groups,subjects,track_mean_data,diff_measures,colors,dir_out=img_dir)

## h0 boostrapping test
from plot_track_data import plotBootstrappedH0TrackAverageDifference
plotBootstrappedH0TrackAverageDifference(groups,subjects,track_mean_data,diff_measures,colors,10000,img_dir)

## bootstrapped histograms
from plot_track_data import plotBootstrappedDifference
plotBootstrappedDifference(groups,subjects,track_mean_data,diff_measures,colors,10000,0.05,img_dir,data_dir+"/tracks_boostrapped")
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
from compile_data import compileFunctionalData
functional_lobe_data = compileFunctionalData(data_dir,cortical,lobes,labelsPath=configs_dir)

## volume, cortical thickness analyses
# cortical thickness/volume by diffusion measure per cortical parcel
from plot_cortex_data import plotCorticalParcelData
for dc in ['volume','thickness']:
	plotCorticalParcelData(groups,colors,dc,cortical,diff_measures,dir_out=img_dir)

## collision vs non-collision scatter
from plot_cortex_data import collisionVNonCollisionParcelScatter
collisionVNonCollisionParcelScatter(groups,colors,cortical,subcortical,diff_measures,dir_out=img_dir)

## lobe averages
from plot_cortex_data import plotMicrostructureAverage
rank_order_lobes = computeRankOrderEffectSize(groups,subjects,'lobes',diff_measures,lobes_data,[diff_measures[0:4],diff_measures[4:]],data_dir)
plotMicrostructureAverage(groups,colors,'lobes',rank_order_lobes['tensor'],lobe_data,diff_measures[0:4],dir_out=img_dir)
plotMicrostructureAverage(groups,colors,'lobes',rank_order_lobes['noddi'],lobe_data,diff_measures[4:],dir_out=img_dir)

## subcortical averages
rank_order_subcortex = computeRankOrderEffectSize(groups,subjects,'subcortex',diff_measures,subcortical,[diff_measures[0:4],diff_measures[4:]],data_dir)
plotMicrostructureAverage(groups,colors,'subcortex',rank_order_subcortex['tensor'],subcortical,diff_measures[0:4],dir_out=img_dir)
plotMicrostructureAverage(groups,colors,'subcortex',rank_order_subcortex['noddi'],subcortical,diff_measures[4:],dir_out=img_dir)

## group difference histograms
from plot_cortex_data import plotDifferenceHistograms
plotDifferenceHistograms(groups,subjects,"cortical",cortical,diff_measures,colors,dir_out=img_dir)
plotDifferenceHistograms(groups,subjects,"subcortical",subcortical,diff_measures,colors,dir_out=img_dir)

## h0 boostrapping test
from plot_cortex_data import plotBootstrappedH0PooledParcelAverageDifference
plotBootstrappedH0PooledParcelAverageDifference(groups,subjects,cortical,'cortical',diff_measures,colors,10000,img_dir)
plotBootstrappedH0PooledParcelAverageDifference(groups,subjects,subcortical,'subcortical',diff_measures,colors,10000,img_dir)

## bootstrapped histograms
from plot_cortex_data import plotBoostrappedDifference
plotBootstrappedDifference(groups,subjects,cortical,"cortical",diff_measures,colors,10009,0.05,img_dir,data_dir+"/cortex_boostrapped")
plotBootstrappedDifference(groups,subjects,subcortical,"subcortical",diff_measures,colors,10000,0.05,img_dir,data_dir+"/subcortex_boostrapped")
print("computing group average gray matter parcel analyses complete")
print("project data has been generated and plotted!")



