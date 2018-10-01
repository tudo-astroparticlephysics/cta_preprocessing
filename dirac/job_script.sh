#!/bin/bash
date >> cta_preprocessing.log 2>&1
echo "Starting pilot script" >> cta_preprocessing.log 2>&1


echo "sourcing miniconda" >> cta_preprocessing.log 2>&1
export PATH=/cvmfs/cta.in2p3.fr/software/miniconda/bin:$PATH
source /cvmfs/cta.in2p3.fr/software/miniconda/bin/activate ctapipe_v0.5.3 >> cta_preprocessing.log 2>&1

echo "installing dependencies" >> cta_preprocessing.log 2>&1
python install_dependencies.py >> cta_preprocessing.log 2>&1
echo "Starting preprocessing" >> cta_preprocessing.log 2>&1
python process_simtel.py "*.simtel.gz"  "./processing_output/" >> cta_preprocessing.log 2>&1
