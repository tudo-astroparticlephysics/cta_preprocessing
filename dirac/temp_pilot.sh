#!/bin/bash

export PATH=/cvmfs/cta.in2p3.fr/software/miniconda/bin:$PATH


date >> t.log 2>&1
source /cvmfs/cta.in2p3.fr/software/miniconda/bin/activate ctapipe_v0.5.3 >> t.log 2>&1

python install_dependencies.py >> t.log 2>&1
python preprocessing.py filename >> t.log 2>&1

dirac-dms-add-file /vo.cta.in2p3.fr/user/k/kai.bruegge/log/filename.log t.log uploade_server

rm t.log
rm c.py
rm preprocess.py
