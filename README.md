# Human white matter microstructure predicts elite sports participation

This is the code repository for the paper entitled _Human white matter microstructure predicts elite sports participation_ (link). This repository contains the code responsible for all figures and analyses generated in the paper. Specifically, this repository contains code for downloading the relevant data from brainlife.io, collating the data into relevant .csv files, analyzing the data, performing machine learning, and producing figure plots. The code here was written exclusively in python3.x.
<!--
#![fig1](./reports/figures/fig1.png)

#![fig2](./reports/figures/fig2.png)
-->

### Authors 

- Brad Caron (bacaron@iu.edu)
- Daniel Bullock
- Lindsey Kitchell
- Brent McPherson
- Derek Kellar
- Hu Cheng
- Sharlene Newman
- Nicholas Port
- Franco Pestilli

### Data availability

#Data used in this project can be found at the accompanying [brainlife.io project](LINKTOPROJ).

### Project Directory Organization

For a better understanding of how this code was run locally, here is the local directory structure:

	.
	├── analyzeWmGmData.py
	├── bl_download.sh
	├── configs
	│   ├── frontal_lobes.txt
	│   ├── insular_lobes.txt
	│   ├── limbic_lobes.txt
	│   ├── motor_lobes.txt
	│   ├── occipital_lobes.txt
	│   ├── parietal_lobes.txt
	│   ├── somatosensory_lobes.txt
	│   └── temporal_lobes.txt
	├── __pycache__
	│   └── analyzeWmGmData.cpython-36.pyc
	├── README.md
	├── todos.txt
	└── utils
	    ├── compile_data.py
	    ├── plot_cortex_data.py
	    ├── plot_track_data.py
	    └── __pycache__
	        ├── compile_data.cpython-36.pyc
	        ├── computeLobeMicrostructure.cpython-36.pyc
	        ├── plot_cortex_data.cpython-36.pyc
	        └── plot_track_data.cpython-36.pyc
	
	4 directories, 20 files

<!--
<sub> This material is based upon work supported by the National Science Foundation Graduate Research Fellowship under Grant No. 1342962. Any opinion, findings, and conclusions or recommendations expressed in this material are those of the authors(s) and do not necessarily reflect the views of the National Science Foundation. </sub>
-->

### Dependencies

This repository requires the following libraries when run locally. 

- npm: https://www.npmjs.com/get-npm
- brainlife CLI: https://brainlife.io/docs/cli/install/
- jsonlab: https://github.com/fangq/jsonlab.git
- python3.6: https://www.python.org/downloads/
- numpy: https://numpy.org/doc/stable/user/install.html
- pandas: https://pandas.pydata.org/
- seaborn: https://seaborn.pydata.org/installing.html
- matplotlib: https://matplotlib.org/faq/installing_faq.html
- scipy: https://www.scipy.org/install.html
- scikit-learn: https://scikit-learn.org/stable/install.html
- statsmodels: https://www.statsmodels.org/stable/install.html

### Docker container

A docker container exists containing all of the dependencies necessary for running the analyses in this paper (https://hub.docker.com/r/bacaron/athlete-brain-container). This container can be executed or downloaded via singularity. See the "main" script for examples of how to execute the analyses via singularity.

### To run locally

To run locally, you'll first need to download the appropriate data using the bl_download.sh shell script. Once the data is downloaded, you can run via python3 the analyzeWmGmData.py script to generate the summary data structures and figures. To run the machine learning analyses, you can run via python3 the mlcAnalyses.py script to run the machine-learning analyses, including model fitting, data summary, and figure generation. This script will take over 30 hours to complete as the machine-learning is not fully optimized.

If you have singularity installed, you can run the entire analysis pipeline by running via shell/bash the main script. This will run the scripts for downloading and analyzing the data via singularity using the docker container described above.

