from DIRAC.Core.Base import Script
Script.setUsageMessage( '\n'.join( ['Usage:',
                                     '%s file_path' % Script.scriptName,
                                     'Arguments:',
                                     'file_path: path to the input file with the list of LFNs to process',
]))
Script.parseCommandLine()
import DIRAC
from DIRAC.Interfaces.API.Job import Job
from DIRAC.Interfaces.API.Dirac import Dirac


def load_files_from_list(path):
    with open(path) as f:
        simtel_files = f.readlines()
        simtel_files = [s.strip() for s in simtel_files if 'SCT' not in s]
        print('Analysing {}'.format(len(simtel_files)))

    return simtel_files[:5]


def run_test_job(args):

    simtel_files = load_files_from_list(args[0])
    #simtel_files = ["/vo.cta.in2p3.fr/MC/PROD3/LaPalma/proton/simtel/1260/Data/071xxx/proton_40deg_180deg_run71001___cta-prod3-lapalma3-2147m-LaPalma.simtel.gz",
    #"/vo.cta.in2p3.fr/MC/PROD3/LaPalma/proton/simtel/1260/Data/070xxx/proton_40deg_180deg_run70502___cta-prod3-lapalma3-2147m-LaPalma.simtel.gz"]
    dirac = Dirac()
    j = Job()
    j.setCPUTime(500)
    j.setInputData(simtel_files[0])
    j.setExecutable('echo', 'Hello World!')
    j.setName('Hello World')
    res = dirac.submit(j)
    print('Submission Result: {}'.format(res))

    return res


if __name__ == '__main__':

    args = Script.getPositionalArgs()
    print(args)
    if len(args) != 1:
        Script.showHelp()
    try:
        res = run_test_job(args)
        if not res['OK']:
            DIRAC.gLogger.error(res['Message'])
            DIRAC.exit(-1)
        else:
            DIRAC.gLogger.notice('Done')
    except Exception:
        DIRAC.gLogger.exception()
        DIRAC.exit(-1)
