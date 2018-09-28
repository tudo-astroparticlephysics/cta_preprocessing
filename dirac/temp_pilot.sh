#!/bin/bash

export PATH=/cvmfs/cta.in2p3.fr/software/miniconda/bin:$PATH


date >> t.log 2>&1
source /cvmfs/cta.in2p3.fr/software/miniconda/bin/activate ctapipe_v0.5.3 >> t.log 2>&1

python c.py >> t.log 2>&1
python preprocessing.py filename tel_type camera_type >> t.log 2>&1

dirac-dms-add-file /vo.cta.in2p3.fr/user/t/thomas.jung/neu_log_Number/trunid.log t.log uploade_server

rm t.log
rm c.py
rm preprocess.py
