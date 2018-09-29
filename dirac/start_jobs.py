import os
from DIRAC.Interfaces.API.Job import Job
from DIRAC.Interfaces.API.Dirac import Dirac
import click
import numpy as np
from tqdm import tqdm
from DIRAC.Core.Base import Script
Script.parseCommandLine()  # this might be needed. No idea why.


@click.command()
@click.argument('dataset')
@click.option('-d', '--delete', help='delete template Files', is_flag=True)
@click.option('-c', '--chunksize', help='delete template Files', default=50)
def main(dataset, delete, chunksize):
    '''
    Dataset is the output of cta-prod3-dump-dataset as described here for example
    https://forge.in2p3.fr/projects/cta_dirac/wiki/CTA-DIRAC_MC_PROD3_Status

    '''
    dirac = Dirac()

    with open(dataset) as f:
        simtel_files = f.readlines()
        print('Analysing {}'.format(len(simtel_files)))

    server_list = ["TORINO-USER", "CYF-STORM-USER", "CYF-STORM-Disk", "M3PEC-Disk", "OBSPM-Disk", "POLGRID-Disk", "FRASCATI-USER", "LAL-Disk", "CIEMAT-Disk", "CIEMAT-USER", "CPPM-Disk", "LAL-USER", "CYFRONET-Disk", "DESY-ZN-USER", "M3PEC-USER", "LPNHE-Disk", "LPNHE-USER", "LAPP-USER", "LAPP-Disk"]
    desy_server = 'DESY-ZN-USER'

    chunks = np.array_split(sorted(simtel_files), int(len(simtel_files) / chunksize))

    print('Got a total of {} chunks'.format(len(chunks)))
    for c, simtel_filenames in tqdm(enumerate(chunks[0:2])):
        print('Starting processing for chunk {}'.format(c))
        j = Job()
        # Set Runtime 0.5h
        j.setCPUTime(30 * 60)
        j.setName('cta_preprocessing_{}'.format(c))
        j.setInputData([simtel_filenames])
        j.setOutputData(['./processing_output/*.hdf5'], outputSE=None, outputPath='cta_preprocessing/')

        j.setInputSandbox( ['pilot.sh', '../process_simtel.py', './install_dependencies.py'])
        j.setOutputSandbox(['cta_preprocessing.log'])
        j.setExecutable('./pilot.sh')
        # These servers seem to  have mini conda installed
        j.setDestination(['LCG.IN2P3-CC.fr', 'LCG.DESY-ZEUTHEN.de', 'LCG.CNAF.it',
                          'LCG.GRIF.fr', 'LCG.CYFRONET.pl',
                          'LCG.Prague.cz', 'LCG.CIEMAT.es'])


        value = dirac.submit(j)['Value']
        print(c + ' Submission Result: {}'.format(value))



if __name__ == '__main__':
    main()
