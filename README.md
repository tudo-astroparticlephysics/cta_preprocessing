# cta_preprocessing

Currently tested on ctapipe commit 63fe206b1d603b0fff61486b6d0c18f73ef4e321 (17.07.2019).

https://github.com/cta-observatory/ctapipe/commit/63fe206b1d603b0fff61486b6d0c18f73ef4e321

## About
Processing of CTA simtel files to hdf5 dL1 files.

The `process_multiple_simtel_files.py` script is a little command line tool to convert multiple simtel files
into one hdf5 file for each simtel file.
The hdf5 files consist of the groups 'runs', 'array_events', 'telescope_events'.

This hdf5 file will contain important monte carlo truths and image features as well as reconstructed directions and other stereo parameters.

## Usage 
Parameters for the processing of the simtel files can be specified via the 
config file that needs to be provided.
A sample config can be found at
`config/sample_config.yaml`

If you want to log into a file or display debug messages, you can also specify a config file for the root logger. A sample config that outputs to two files and the terminal can be fount at
`config/logging_config.yaml`


Call like this:
`python process_simtel.py input_pattern output_folder config_file`
Additional options are:

  ```
  -l (--logger_config_file: (optional) config file to specify logging behaviour)
  -n (--n_events: events to process per file)
  -j (--n_jobs: jobs to start)
  --overwrite/--no-overwrite (whether to overwrite existing output files)
  -v (--verbose: informations to provide while running)
  -c (--chunksize: number of files to join as one chunk)
  ```

These files can then be merged via 
`merge_files.py` to receive a single file.


## A "complete" Analysis
Starting from the monte carlo simtel files:
- Call `process_multiple_simtel_files.py` with parameters on a bunch of gamma and a bunch of proton files.
- Merge gamma/protons via `merge_files.py` to get two files: one for gammas and one for protons
- Use the [aict-tools](https://github.com/fact-project/aict-tools) or similar tools for machine learning (Split into train/test sets, train models for energy regression and gamma/hadron separation and apply these models on your test splits.)

## Download
Helper script to download Prod-3 monte carlo.
Requires a password.


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
