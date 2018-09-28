import os
os.system("python -m pip install --user --trusted-host pypi.python.org click")
os.system("python -m pip install --user --trusted-host pypi.python.org h5py")
try:
    from ctapipe.io.hessio import hessio_event_source
    from ctapipe.calib.camera.r1 import HessioR1Calibrator
    from ctapipe.calib.camera.dl0 import CameraDL0Reducer
    from ctapipe.calib.camera import CameraDL1Calibrator
    source = hessio_event_source("filename", max_events=1)
    for event in source:
        if event.mc.spectral_index != 0:
            print("CTApipe Ja")
        else:
            bla
except:
    os.system("tar xvfJ pyhessio.tar.xz > ../bla.log 2>&1; cd pyhessio; python setup.py install --user >> ../t.log 2>&1; cd ..")
    os.system("tar xvfJ ctapipe-extra.tar.xz > ../bla.log 2>&1; cd ctapipe-extra; python setup.py install --user >> ../t.log 2>&1; cd ..")
    os.system("tar xvfJ ctapipe2.tar.xz > ../bla.log 2>&1; cd ctapipe; python setup.py install --user >> ../t.log 2>&1; cd ..")
    try:
        from ctapipe.io.hessio import hessio_event_source
        from ctapipe.calib.camera.r1 import HessioR1Calibrator
        from ctapipe.calib.camera.dl0 import CameraDL0Reducer
        from ctapipe.calib.camera import CameraDL1Calibrator
        print("CTApipe Ja")
    except:
        print("CTApipe error")

try:
    import h5py
    print("h5py Ja")
except:
    os.system("python -m pip install --user --trusted-host pypi.python.org h5py")
    try:
        import h5py
        print("h5py Ja")
    except:
        print("h5py error")
try:
    import pandas
    print("pandas Ja")
except:
    try:
        import pandas
        print("pandas Ja")
    except:
        print("pandas error")
