import os
from DIRAC.Core.Base import Script
Script.parseCommandLine()  # this might be needed. No idea why. This seems highly fishy. 
from DIRAC.Interfaces.API.Job import Job
from DIRAC.Interfaces.API.Dirac import Dirac
import click
import numpy as np
from tqdm import tqdm


@click.command()
@click.argument('dataset')
@click.option('-c', '--chunksize', help='delete template Files', default=50)
def main(dataset, delete, chunksize):
    '''
    The DATASET argument is a list of paths to MC files on the grid. Like the output of
    cta-prod3-dump-dataset for example. See also
    https://forge.in2p3.fr/projects/cta_dirac/wiki/CTA-DIRAC_MC_PROD3_Status

    Keep in mind that for some effing reason this needs to be executed within this weird 'dirac'
    environment which comes with its own glibc, python and pip. I guess the real Mr. Dirac would turn in his grave.

    '''
    dirac = Dirac()

    with open(dataset) as f:
        simtel_files = f.readlines()
        print('Analysing {}'.format(len(simtel_files)))

    server_list = ["TORINO-USER", "CYF-STORM-USER", "CYF-STORM-Disk", "M3PEC-Disk", "OBSPM-Disk", "POLGRID-Disk", "FRASCATI-USER", "LAL-Disk", "CIEMAT-Disk", "CIEMAT-USER", "CPPM-Disk", "LAL-USER", "CYFRONET-Disk", "DESY-ZN-USER", "M3PEC-USER", "LPNHE-Disk", "LPNHE-USER", "LAPP-USER", "LAPP-Disk"]
    desy_server = 'DESY-ZN-USER'

    servers_with_miniconda = ['LCG.IN2P3-CC.fr', 'LCG.DESY-ZEUTHEN.de', 'LCG.CNAF.it',
                          'LCG.GRIF.fr', 'LCG.CYFRONET.pl',
                          'LCG.Prague.cz', 'LCG.CIEMAT.es']


    chunks = np.array_split(sorted(simtel_files), int(len(simtel_files) / chunksize))

    print('Got a total of {} chunks'.format(len(chunks)))
    for c, simtel_filenames in tqdm(enumerate(chunks[0:2])): # send just 2 jobs for now.
        # convert chunk to a list of strings. becasue this dirac thing cant take numpy arrays 
        simtel_filenames = [str(s) for s in simtel_filenames]
        print('Starting processing for chunk {}'.format(c))
        j = Job()
        # set runtime to 0.5h
        j.setCPUTime(30 * 60)
        j.setName('cta_preprocessing_{}'.format(c))
        j.setInputData(simtel_filenames)
        j.setOutputData(['./processing_output/*.hdf5'], outputSE=None, outputPath='cta_preprocessing/')

        j.setInputSandbox( ['../process_simtel.py', './install_dependencies.py'])
        j.setOutputSandbox(['cta_preprocessing.log'])
        j.setExecutable('./job_script.sh')
        # These servers seem to  have mini conda installed
        #destination = np.random.choice(servers_with_miniconda)
        j.setDestination(servers_with_miniconda)


        value = dirac.submit(j)
        print('Number {} Submission Result: {}'.format(c, value))



if __name__ == '__main__':
    main()
