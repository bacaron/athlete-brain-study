# Advanced mapping of the human white matter microstructure better separates elite sports participation

This is the code repository for the paper entitled _Advanced mapping of the human white matter microstructure better separates elite sports participation_ (Caron et al, Neuroimage: Clinical, _in review_) (link). This repository contains the code responsible for all figures and analyses generated in the paper. Specifically, this repository contains code for downloading the relevant data from brainlife.io, collating the data into relevant .csv files, analyzing the data, performing machine learning, and producing figure plots. The code here has been implemented in python3.6.
<!--
#![fig1](./reports/figures/fig1.png)

#![fig2](./reports/figures/fig2.png)
-->

### Authors 

- Brad Caron (bacaron@iu.edu)

### Acknowledgements  

This research was supported by NSF OAC-1916518, NSF IIS-1912270, NSF, IIS-1636893, NSF BCS-1734853, NIH NIDCD 5R21DC013974-02, NIH 1R01EB029272-01, NIH NIMH 5T32MH103213, the Indiana Spinal Cord and Brain Injury Research Fund, Microsoft Faculty Fellowship, the Indiana University Areas of Emergent Research initiative “Learning: Brains, Machines, Children.” We thank Soichi Hayashi, and David Hunt for contributing to the development of brainlife.io, Craig Stewart, Robert Henschel, David Hancock and Jeremy Fischer for support with jetstream-cloud.org (NSF ACI-1445604). We also thank The Indiana University Lawrence D. Rink Center for Sports Medicine and Technology and Center for Elite Athlete Development for contributing funding to athletic scientific research and for the development of a new research facility.

### Data availability

Data used in this project can be found at the accompanying [brainlife.io project](https://brainlife.io/project/5cb8973c71a8630036207a6a).

### Project Directory Organization

For a better understanding of how this code was run locally, here is the local directory structure:

	.
	├── analyzeWmGmData.py
	├── bl_download.sh
	├── config.json
	├── configs
	│   ├── frontal_lobes.txt
	│   ├── insular_lobes.txt
	│   ├── limbic_lobes.txt
	│   ├── motor_lobes.txt
	│   ├── occipital_lobes.txt
	│   ├── parietal_lobes.txt
	│   ├── somatosensory_lobes.txt
	│   └── temporal_lobes.txt
	│   └── association_tracks.txt
	│   └── projection_tracks.txt
	│   └── commissural_tracks.txt
	│   └── mass.csv
	├── README.md
	└── utils
	|   ├── compile_data.py
	|   ├── plot_cortex_data.py
	|   └── plot_track_data.py
	└── data_descriptor
	|   └── XX
	
	4 directories, 19+XX files

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
- pinguoin: https://pingouin-stats.org/

### Docker container

A docker container exists containing all of the dependencies necessary for running the analyses in this paper (https://hub.docker.com/r/bacaron/athlete-brain-container). This container can be executed or downloaded via singularity. See the "main" script for examples of how to execute the analyses via singularity.

### To run locally

Before any of the scripts provided are ran, the user must set up two specific paths in the config.json file. 'topPath' is the filepath to the directory in where you want the data to be downloaded and analyses to be ran. This directory will be created by the bl_download.sh script if it does not already exist. 'scriptsPath' is the path to this repository on your local machine. All of the other variables in the config.json should be left as is.

To run locally, you'll first need to download the appropriate data using the bl_download.sh shell script. Once the data is downloaded, you can run via python3 the analyzeWmGmData.py script to generate the summary data structures and figures. This route requires all of the dependencies to be installed on your machine.

If you have singularity installed, you can run the entire analysis pipeline by running via shell/bash the main script. This will run the scripts for downloading and analyzing the data via singularity using the docker container described above. This route does not require all of the dependencies to be installed on your machine, and will reproduce the analyses exactly as they were ran for the paper.



