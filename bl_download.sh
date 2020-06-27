#!/bin/bash

topDir=$1
projectID="5cb8973c71a8630036207a6a"
datatypes="snr tractmeasures-profiles tractmeasures-cleaned parc-stats-cortex parc-stats-aparc parc-stats-subcort"

for DTYPES in ${datatypes}
do
	echo ${DTYPES}
	if [ ! -f data_${DTYPES}.json ]; then
		if [[ ${DTYPES} == 'snr' ]]; then
			bl dataset query --project ${projectID} --datatype raw --datatype_tag "snr-cc" --json > data_${DTYPES}.json
		elif [[ ${DTYPES} == 'tractmeasures-cleaned' ]]; then
			bl dataset query --project ${projectID} --datatype neuro/tractmeasures --datatype_tag "cleaned" --json > data_${DTYPES}.json
		elif [[ ${DTYPES} == 'tractmeasures-profiles' ]]; then
			bl dataset query --project ${projectID} --datatype neuro/tractmeasures --tag "profiles" --json > data_${DTYPES}.json
		elif [[ ${DTYPES} == 'parc-stats-cortex' ]]; then
			bl dataset query --project ${projectID} --datatype neuro/parc-stats --datatype_tag "cortex_mapping_stats" --json > data_${DTYPES}.json
		elif [[ ${DTYPES} == 'parc-stats-aparc' ]]; then
			bl dataset query --project ${projectID} --datatype neuro/parc-stats --datatype_tag "acpc_aligned" --json > data_${DTYPES}.json
		elif [[ ${DTYPES} == 'parc-stats-subcort' ]]; then
			bl dataset query --project ${projectID} --datatype neuro/parc-stats --datatype_tag "subcort_stats" --json > data_${DTYPES}.json
		fi
	fi

	for subject in $(jq -r '.[].meta.subject' data_${DTYPES}.json | sort -u)
	do
		# make subject directory if not made
		if [ ! -d $topDir/$subject ]; then
			mkdir -p $topDir/$subject
		fi

		# make datatype directory if not made
	    if [ ! -d $topDir/$subject/${DTYPES}/ ];
	    then
			echo "downloading subject:$subject ---------------"
			mkdir -p $topDir/$subject/${DTYPES}
			ids=$(jq -r '.[] | select(.meta.subject == '\"$subject\"') | ._id' data_${DTYPES}.json)
			for id in $ids
			do
			        echo $id
			        outdir=$topDir/$subject/${DTYPES}/
			        bl dataset download -i $id --directory $outdir
			done
	    fi
	done
done
