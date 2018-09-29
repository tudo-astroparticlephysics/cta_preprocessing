import os
os.system("python -m pip install --user --trusted-host pypi.python.org click")
os.system("python -m pip install --user --trusted-host pypi.python.org h5py")
# os.system("python -m pip install --user --trusted-host pypi.python.org astropy")
os.system("python -m pip install --user --trusted-host pypi.python.org pandas")
os.system("python -m pip install --user --trusted-host pypi.python.org pyfact")
os.system("python -m pip install --user --trusted-host github.com https://github.com/cta-observatory/ctapipe-extra/archive/master.zip")
os.system("python -m pip install --user --trusted-host github.com https://github.com/cta-observatory/pyhessio/archive/master.zip")
os.system("python -m pip install --user --trusted-host github.com https://github.com/mackaiver/ctapipe/archive/new_processing.zip")

try:
    from ctapipe.io.hessio import hessio_event_source
    from ctapipe.calib.camera.r1 import HessioR1Calibrator
    from ctapipe.calib.camera.dl0 import CameraDL0Reducer
    from ctapipe.calib.camera import CameraDL1Calibrator
    source = hessio_event_source("filename", max_events=1)
    print("Dependencies seem to be working")
except:
    print("Dependency installation error")
