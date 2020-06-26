#!/usr/bin/env python
import os,sys,glob
from matplotlib import colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from compute_plot_length_volume_final import compute_plot_length_volume_final

topPath = "/media/brad/APPS/athlete-updated-pipeline-qc"
os.chdir(topPath)
scripts_dir = topPath+'/athlete_brain_study/'
sys.path.insert(0,scripts_dir)
sys.path.insert(1,topPath+'/athlete_brain_study/utils')
sys.path.insert(2,topPath+'/athlete_brain_study/gray-matter-parcels')
sys.path.insert(2,topPath+'/athlete_brain_study/white-matter-tracks')

groups = ['football','cross_country','non_athlete']
colors_array = ['orange','pink','blue']

colors = {}
subjects = {}

img_dir = topPath + '/img/'
data_dir = topPath + '/data/'

# loop through groups and identify subjects and set color schema for each group
for g in range(len(groups)):
	# set subjects array
	subjects[groups[g]] =  [f.split(topPath+'/')[1] for f in glob.glob(topPath+'/*'+str(g+1)+'_0*')]
	subjects[groups[g]].sort()

	# set colors array
	colors_name = colors_array[g]
	colors[groups[g]] = colors_array[g]

### group average white matter analyses
## length, volume, streamline count of tracks (add subjects)
compute_plot_length_volume_final(topPath,groups,colors)



## DTI/NODDI tract profiles


## athlete vs non-athlete scatter



## collision vs non-collision scatter



## DTI/NODDI scatter plots (group averages)






### group average cortex mapping analyses
## volume, cortical thickness analyses
# total brain measures
from computeTotalBrainStats import computeTotalBrainStats
computeTotalBrainStats(topPath,data_dir,groups,subjects,colors)

# cortical thickness/volume by diffusion measure per cortical parcel
from computeThicknessVolumeDiffRoi import computeThicknessVolumeDiffRoi
computeThicknessVolumeDiffRoi(topPath,data_dir,groups,subjects,colors)

## collision vs non-collision scatter
from computeCollisionVNonCollisionCortex import computeCollisionVNonCollisionCortex
computeCollisionVNonCollisionCortex(topPath,data_dir,groups,subjects,colors)


## DTI/NODDI scatter plots (group averages)




### MLC analyses
