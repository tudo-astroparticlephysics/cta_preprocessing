from DIRAC.Core.Base import Script
Script.parseCommandLine()
from DIRAC.Interfaces.API.Job import Job
from DIRAC.Interfaces.API.Dirac import Dirac
dirac = Dirac()
j = Job()
j.setCPUTime(500)
j.setExecutable('echo', 'Hello World!')
j.setName('Hello World')
res = dirac.submit(j)
print 'Submission Result: ',res['Value']
