from DIRAC.Core.Base import Script
Script.parseCommandLine()
import DIRAC
from DIRAC.Interfaces.API.Job import Job
from DIRAC.Interfaces.API.Dirac import Dirac


def run_test_job(args):
    dirac = Dirac()
    j = Job()
    j.setCPUTime(500)
    j.setExecutable('echo', 'Hello World!')
    j.setName('Hello World')
    res = dirac.submit(j)
    print('Submission Result: {}'.format(res))


if __name__ == '__main__':

    args = Script.getPositionalArgs()
    if len(args) not in [3, 4]:
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
