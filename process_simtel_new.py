from ctapipe.io.eventsourcefactory import EventSourceFactory
from ctapipe.calib import CameraCalibrator
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError
from ctapipe.image import leakage, concentration
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.reco import HillasReconstructor
from ctapipe.reco.HillasReconstructor import TooFewTelescopesException
from ctapipe.coordinates import HorizonFrame

from collections import Counter
import fact.io
import click
import numpy as np
from tqdm import tqdm
import astropy.units as u
from astropy.coordinates import SkyCoord

import copy
import os

names_to_id = {'LSTCam': 1, 'NectarCam': 2, 'FlashCam': 3, 'DigiCam': 4, 'CHEC': 5}
types_to_id = {'LST': 1, 'MST': 2, 'SST': 3}
allowed_cameras = ['LSTCam', 'NectarCam', 'DigiCam']


cleaning_level = {
    # 'ASTRICam': (5, 7, 2),  # (5, 10)?
    # 'FlashCam': (12, 15, 2),
    'LSTCam': (3.5, 7.5, 2),  
    'NectarCam': (3, 5.5, 2),
    # "FlashCam": (4, 8),  # there is some scaling missing?
    'DigiCam': (2, 4.5, 2),
    # 'CHEC': (2, 4, 2),
    # 'SCTCam': (1.5, 3, 2)
}

class ReconstructionError(Exception):
    pass

from ctapipe.io.containers import  TelescopePointingContainer, MCEventContainer, Container, ReconstructedShowerContainer
from ctapipe.io.containers import LeakageContainer, HillasParametersContainer, ConcentrationContainer
from ctapipe.core import Field

class TelescopeParameterContainer(Container):
    telescope_id = Field(-1, 'telescope id')
    run_id = Field(-1, 'run id')
    array_event_id = Field(-1, 'array event id')

    leakage = Field(LeakageContainer(), 'Leakage container')
    hillas = Field(HillasParametersContainer(), 'HillasParametersContainer')
    concentration = Field(ConcentrationContainer(), 'ConcentrationContainer')

    pointing = Field(TelescopePointingContainer, 'pointing information')

    distance_to_reconstructed_core_position = Field(np.nan, 'Distance from telescope to reconstructed impact position', unit=u.m)

    mirror_area = Field(np.nan, 'Mirror Area', unit=u.m**2)
    focal_length = Field(np.nan, 'focal length', unit=u.m)


class ArrayEventContainer(Container):
    run_id = Field(-1, 'run id')
    array_event_id = Field(-1, 'array event id')
    reco = Field(ReconstructedShowerContainer(), 'reconstructed shower container')
    mc = Field(MCEventContainer(), 'array wide MC information')

    total_intensity = Field(np.nan, 'sum of all intensities')

    num_triggered_telescopes = Field(np.nan, 'Number of triggered Telescopes')

    num_triggered_lst = Field(np.nan, 'Number of triggered LSTs')
    num_triggered_mst = Field(np.nan, 'Number of triggered MSTs')
    num_triggered_sst = Field(np.nan, 'Number of triggered SSTs')


@click.command()
@click.argument('input_file', type=click.Path(dir_okay=False))
@click.argument('output_file', type=click.Path(dir_okay=True,file_okay=False))
@click.option('-n', '--n_events', default=-1, help='number of events to process in each file.')
@click.option('-j', '--n_jobs', default=1, help='number of jobs to start. this is usefull when passing more than one simtel file.')
@click.option('--overwrite/--no-overwrite', default=False, help='If false (default) will only process non-existing filenames')
def main(input_file, output_file, n_events, n_jobs, overwrite):
    '''
    process simtel files given as INPUT_FILE into one hdf5 file saved in OUTPUT_FILE.
    The hdf5 file will contain three groups. 'runs', 'array_events', 'telescope_events'.

    These files can be put into the classifier tools for learning.
    See https://github.com/fact-project/classifier-tools

    '''    
    print(f'processing file {input_file}, writing to {output_file}')
    
    if not overwrite:
        if os.path.exists(output_file):
            print(f'Output file exists. Stopping.')
            return
    else:
        if os.path.exists(output_file):
            print(f'Output file exists. Overwriting.')
            os.remove(output_file)

    runs, array_events, telescope_events = process_file(input_file,  n_events=n_events)

    if runs is None or array_events is None or telescope_events is None:
        print('file contained no information.')

    # TODO write data somehow
    # verify_file(output_file)



def process_file(input_file, n_events=-1, silent=False):
    source = EventSourceFactory.produce(
        input_url=input_file,
        max_events=n_events if n_events > 1 else None,
    )
    calibrator = CameraCalibrator(
        eventsource=source,
        r1_product='HESSIOR1Calibrator',
        extractor_product='NeighbourPeakIntegrator',
    )

    # telescope_event_information = []
    # array_event_information = []
    allowed_tels = [id for id in source._subarray_info.tels if source._subarray_info.tels[id].camera.cam_id in allowed_cameras]
    source.allowed_tels = allowed_tels
    
    for event in tqdm(filter(lambda e: len(e.dl0.tels_with_data) > 1, source), disable=silent):
        try:
            result = process_event(event, calibrator)
        except ReconstructionError:
            pass

    return



def calculate_image_features(telescope_id, event, dl1):
    array_event_id = event.dl0.event_id
    run_id =  event.r0.obs_id
    camera = event.inst.subarray.tels[telescope_id].camera

    boundary_thresh, picture_thresh, min_number_picture_neighbors = cleaning_level[camera.cam_id]
    mask = tailcuts_clean(
        camera,
        dl1.image[0],
        boundary_thresh=boundary_thresh,
        picture_thresh=picture_thresh,
        min_number_picture_neighbors=min_number_picture_neighbors
    )

    
    cleaned = dl1.image[0].copy()
    cleaned[~mask] = 0
    hillas_container = hillas_parameters(
        camera,
        cleaned,
    )
    leakage_container = leakage(camera, dl1.image[0], mask)
    concentration_container = concentration(camera, dl1.image[0], hillas_container)
    
    alt_pointing = event.mc.tel[telescope_id].azimuth_raw * u.rad
    az_pointing = event.mc.tel[telescope_id].altitude_raw * u.rad
    pointing_container = TelescopePointingContainer(azimuth=az_pointing, altitude=alt_pointing)

    return TelescopeParameterContainer(
        telescope_id=telescope_id,
        run_id=run_id,
        array_event_id=array_event_id,
        leakage=leakage_container,
        hillas=hillas_container,
        concentration=concentration_container,
        pointing=pointing_container
    )


def calculate_distance_to_core(tel_params, event, reconstruction_result):
    for tel_id, container in tel_params.items():
        pos = event.inst.subarray.positions[tel_id]
        x, y = pos[0], pos[1]
        core_x = reconstruction_result.core_x
        core_y = reconstruction_result.core_y
        d = np.sqrt((core_x - x)**2 + (core_y - y)**2)*u.m

        container.distance_to_reconstructed_core_position = d



    # pointing_azimuth[telescope_id] = event.mc.tel[telescope_id].azimuth_raw * u.rad
    # pointing_altitude[telescope_id] = event.mc.tel[telescope_id].altitude_raw * u.rad
    # tel_x[telescope_id] = event.inst.subarray.positions[telescope_id][0]
    # tel_y[telescope_id] = event.inst.subarray.positions[telescope_id][1]

    # telescope_description = event.inst.subarray.tel[telescope_id]
    # tel_focal_lengths[telescope_id] = telescope_description.optics.equivalent_focal_length

    # d = {
    #     'array_event_id': event.dl0.event_id,
    #     'telescope_id': int(telescope_id),
    #     'camera_name': camera.cam_id,
    #     'camera_id': names_to_id[camera.cam_id],
    #     'run_id': event.r0.obs_id,
    #     'telescope_type_name': telescope_type_name,
    #     'telescope_type_id': types_to_id[telescope_type_name],
    #     'pointing_azimuth': event.mc.tel[telescope_id].azimuth_raw,
    #     'pointing_altitude': event.mc.tel[telescope_id].altitude_raw,
    #     'mirror_area': telescope_description.optics.mirror_area,
    #     'focal_length': telescope_description.optics.equivalent_focal_length,
    # }

    # d.update(hillas_container.as_dict())
    # d.update(leakage_container.as_dict())
    # features[telescope_id] = ({k: strip_unit(v) for k, v in d.items()})

def process_event(event, calibrator):
    '''
    Processes
    '''

    telescope_types = []

    hillas_reconstructor = HillasReconstructor()

    calibrator.calibrate(event)

    tel_parameters = {}
    for telescope_id, dl1 in event.dl1.tel.items():
        telescope_type_name = event.inst.subarray.tels[telescope_id].optics.tel_type
        telescope_types.append(telescope_type_name)

        try:
            tel_parameters[telescope_id] = calculate_image_features(telescope_id, event, dl1)
        except HillasParameterizationError:
            continue


    if len(tel_parameters) < 2:
        raise ReconstructionError('Not enough telescopes for which Hillas parameters could be reconstructed.')

    parameters = {tel_id: tel_parameters[tel_id].hillas for tel_id in tel_parameters}
    pointing_altitude = {tel_id: tel_parameters[tel_id].pointing.altitude for tel_id in tel_parameters}
    pointing_azimuth = {tel_id: tel_parameters[tel_id].pointing.azimuth for tel_id in tel_parameters}

    reconstruction_result = hillas_reconstructor.predict(parameters, event.inst, pointing_altitude, pointing_azimuth )

    calculate_distance_to_core(tel_parameters, event, reconstruction_result)
    
    mc_container = copy.deepcopy(event.mc)
    mc_container.tel = None
    mc_container.prefix = 'mc'

    counter = Counter(telescope_types)  
    event_params = ArrayEventContainer(
        array_event_id=event.dl0.event_id,
        run_id =  event.r0.obs_id,
        reco=reconstruction_result,
        total_intensity = sum([t.hillas.intensity for t in tel_parameters.values()]),
        num_triggered_lst =  counter['LST'],
        num_triggered_mst =  counter['MST'],
        num_triggered_sst =  counter['SST'],
        num_triggered_telescopes = len(telescope_types),
        mc=mc_container,
    )

    return tel_parameters, event_params



def number_of_valid_triggerd_cameras(event):
    triggerd_tel_ids = event.trig.tels_with_trigger
    triggerd_camera_names = [event.inst.subarray.tels[i].camera.cam_id for i in triggerd_tel_ids]
    valid_triggered_cameras = list(filter(lambda c: c in allowed_cameras, triggerd_camera_names))
    return len(valid_triggered_cameras)


def strip_unit(v):
    try:
        return v.si.value
    except AttributeError:
        return v


if __name__ == '__main__':
    main()
