import os

try:
    import click
except:
    print('installing click')
    os.system("python -m pip install --user --trusted-host pypi.python.org click")

try:
    import h5py
except:
    print('installing h5py')
    os.system("python -m pip install --user --trusted-host pypi.python.org h5py")

try:
    import astropy
except:
    print('installing astropy')
    os.system("python -m pip install --user --trusted-host pypi.python.org astropy")

try:
    import pandas
except:
    print('installing pandas')
    os.system("python -m pip install --user --trusted-host pypi.python.org pandas")

try:
    import fact.io
except:
    print('installing pyfact')
    os.system("python -m pip install --user --trusted-host pypi.python.org pyfact")

os.system("python -m pip install --user --trusted-host github.com https://github.com/cta-observatory/ctapipe-extra/archive/master.zip")
os.system("python -m pip install --user --trusted-host github.com https://github.com/cta-observatory/pyhessio/archive/master.zip")
os.system("python -m pip install --user --trusted-host github.com https://github.com/mackaiver/ctapipe/archive/new_processing.zip")

try:
    from ctapipe.io.hessio import hessio_event_source
    from ctapipe.calib.camera.r1 import HessioR1Calibrator
    from ctapipe.calib.camera.dl0 import CameraDL0Reducer
    from ctapipe.calib.camera import CameraDL1Calibrator
    print("Dependencies seem to be working")
except:
    print("Dependency installation error")
