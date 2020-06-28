# Human white matter microstructure predicts elite sports participation

This is the code repository for the paper entitled _Human white matter microstructure predicts elite sports participation_ (link). This repository contains the code responsible for all figures and analyses generated in the paper. Specifically, this repository contains code for downloading the relevant data from brainlife.io, collating the data into relevant .csv files, analyzing the data, performing machine learning, and producing figure plots. The code here was written exclusively in python3.x.

#![fig1](./reports/figures/fig1.png)

#![fig2](./reports/figures/fig2.png)

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

<sub> This material is based upon work supported by the National Science Foundation Graduate Research Fellowship under Grant No. 1342962. Any opinion, findings, and conclusions or recommendations expressed in this material are those of the authors(s) and do not necessarily reflect the views of the National Science Foundation. </sub>

