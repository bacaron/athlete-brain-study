#!/usr/bin/env python3

import os,sys,glob
from matplotlib import colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import pingouin as pg
import seaborn as sns


# covariates
covar =  'age','Total Brain Volume'


# individual structures - MIXED EFFECTS ANOVA
for tt in tissue_names:
	for mn in measures[tt]:
		aov[tt][mn] = pg.mixed_anova(dv=mn,within='structureID',between='classID',subject='subjectID',data=df[tt])

# pairwise t-tests
for tt in tissue_names:
	for mn in measures[tt]:
		posthoc[tt][mn] = pg.pairwise_ttests(dv=mn,within='structureID',between='classID',subject='subjectID',data=df[tt],padjust='bonf',effsize='cohen')
		sig_posthoc[tt][mn] = posthocs[tt][mn][posthocs[tt][mn]['Contrast'].str.contains('classID')][posthocs['p-corr'] < 0.05]
