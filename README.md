# cta_preprocessing
Processing of CTA simtel files to hdf5 DL1 files.

The `process_multiple_simtel_files.py` script is a little command line tool to convert multiple simtel files
into one hdf5 file for each simtel file.
The hdf5 files consist of the groups 'runs', 'array_events', 'telescope_events'.

This hdf5 file will contain important monte carlo truths and image features as well as reconstructed directions and other stereo parameters.

Parameters for the processing of the simtel files can be specified via the 
config file that needs to be provided.
A sample config can be found at
`config/sample_config.yaml`


Call like this:
`python process_simtel.py input_pattern output_folder config_file`
Additional options are:

  ```
  -n (--n_events: events to process per file)
  -j (--n_jobs: jobs to start)
  --overwrite/--no-overwrite (whether to overwrite existing output files)
  -v (--verbose: informations to provide while running)
  -c (--chunksize: number of files to join as one chunk)
  ```

These files can then be merged via 
`merge_files.py`
and put into the classifier tools for machine learning.
See https://github.com/fact-project/classifier-tools


## Probably deprecated
### dirac grid stuff

First follow the steps given in the user guide here:

https://forge.in2p3.fr/projects/cta_dirac/wiki/CTA-DIRAC_Users_Guide

This will install its own version of python and pip for some effing reason. I do not think there is an excuse for that.
Anyhow. After thats all over you might want to install dependencies in that python installation. This works by simply doing this then:
```
pip install --trusted-host pypi.org --trusted-host pypi.python.org  --trusted-host files.pythonhosted.org  numpy click tqdm
```

Some of the gird servers seem to come with a miniconda version. So one might be able to use pip to install things at the nodes. Im gonna try. 
