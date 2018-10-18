# cta_preprocessing
Processing of CTA simtel files to hdf5 DL2 files.

The `process_simtel.py` script is a little command line tool to convert multiple simtel files
into one hdf5 file with the groups. 'runs', 'array_events', 'telescope_events'.

This hdf5 file will contain important monte carlo truths and image features as well as reconstructed
directions and other stereo parameters.

Call like this:
  `python process_simtel.py /data/somewhere/protons/ protons.hdf5 -j 24`

It will collect all files in the given folder with the 'simtel.gz' suffix and convert them to a single
dl2 file.

These files can be put into the classifier tools for learning.
See https://github.com/fact-project/classifier-tools


### dirac grid stuff

First follow the steps given in the user guide here:

https://forge.in2p3.fr/projects/cta_dirac/wiki/CTA-DIRAC_Users_Guide

This will install its own version of python and pip for some effing reason. I do not think there is an excuse for that.
Anyhow. After thats all over you might want to install dependencies in that python installation. This works by simply doing this then:
```
pip install --trusted-host pypi.org --trusted-host pypi.python.org  --trusted-host files.pythonhosted.org  numpy click tqdm
```

Some of the gird servers seem to come with a miniconda version. So one might be able to use pip to install things at the nodes. Im gonna try. 
