import pandas as pd
import os,glob,sys
import numpy as np
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from plot_cortex_data import collisionVNonCollisionScatter

def computeCollisionVNonCollisionCortex(topPath,dataPath,groups,subjects,colors):

	# measures to loop through
	diff_measures = ['ad','fa','md','rd','ndi','isovf','odi']

	# load newly created cortex_nodes.csv file
	data = pd.read_csv(dataPath+'/cortex_nodes.csv')

	# generate plots
	collisionVNonCollisionScatter(groups,colors,data,diff_measures,dir_out=topPath+'/img/')

if __name__ == '__main__':
	collisionVNonCollisionScatter(sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])