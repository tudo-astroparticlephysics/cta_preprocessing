# cta_preprocessing
Processing of CTA simtel files to hdf5 DL2 files.

The `process_simtel.py` script is a little command line tool to convert multiple simtel files
into one hdf5 file with the groups. 'runs', 'array_events', 'telescope_events'.

This hdf5 file will contain important monte carlo truths and image features as well as reconstructed
directions and other stereo parameters.

Call like this:
  `python process_simtel.py /data/somewhere/proton_20deg_0deg_run*.simtel.gz protons.hdf5 -j 24`

These files can be put into the classifier tools for learning.
See https://github.com/fact-project/classifier-tools
