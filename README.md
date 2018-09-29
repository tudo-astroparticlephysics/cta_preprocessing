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

I guess one first has to upload a tarball containing the ctapipe version (and other stuff) one wants to use.
`dirac-dms-add-file --help`

Like this for example
```
tar -cvf ctapipe_custom.tar /home/kbruegge/ctapipe
tar -cvf pyhessio.tar /home/kbruegge/pyhessio
tar -cvf ctapipe-extra.tar /home/kbruegge/ctapipe-extra
dirac-dms-add-file /vo.cta.in2p3.fr/user/k/kai.bruegge/tar/ctapipe_custom.tar ctapipe_custom.tar
dirac-dms-add-file /vo.cta.in2p3.fr/user/k/kai.bruegge/tar/pyhessio.tar pyhessio.tar
dirac-dms-add-file /vo.cta.in2p3.fr/user/k/kai.bruegge/tar/ctapipe-extra.tar ctapipe-extra.tar


```
