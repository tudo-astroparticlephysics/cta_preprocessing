from collections import Counter
import click
import numpy as np
from tqdm import tqdm
import astropy.units as u

import copy
from functools import partial
import os

from joblib import delayed, Parallel

from ctapipe.io.eventsourcefactory import EventSourceFactory
from ctapipe.io import HDF5TableWriter
from ctapipe.calib import CameraCalibrator
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError
from ctapipe.image import leakage, concentration
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.reco import HillasReconstructor

import copy

from preprocessing.containers import TelescopeParameterContainer, ArrayEventContainer, RunInfoContainer
from ctapipe.io.containers import  TelescopePointingContainer


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


@click.command()
@click.argument('input_file', type=click.Path())
@click.argument('output_file', type=click.Path())
@click.option('-n', '--n_events', default=-1, help='number of events to process in each file.')
@click.option('-j', '--n_jobs', default=2, help='number of jobs to start. this is usefull when passing more than one simtel file.')
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

    run_info_container, array_events, telescope_events = process_file(input_file, n_events=n_events, n_jobs=n_jobs)
    write_result_to_file(run_info_container, array_events, telescope_events, output_file)


def write_result_to_file(run_info_container, array_events, telescope_events, output_file, mode='a'):
    with HDF5TableWriter(output_file, mode=mode, group_name='', add_prefix=True) as h5_table:
        run_info_container.mc.run_array_direction = 0
        h5_table.write('runs', [run_info_container, run_info_container.mc])
        for array_event in array_events:
            h5_table.write('array_events', [array_event, array_event.mc, array_event.reco ])

        for tel_event in telescope_events:
            h5_table.write(
                'telescope_events',
                [tel_event, tel_event.pointing, tel_event.hillas, tel_event.concentration, tel_event.leakage]
            )


def process_parallel(event, calibrator):
    try:
        return process_event(event, calibrator)
    except ReconstructionError:
        pass

def print_info(event):
    print(event.dl0.event_id)

def process_file(input_file, n_events=-1, silent=False, n_jobs=2):
    source = EventSourceFactory.produce(
        input_url=input_file,
        max_events=n_events if n_events > 1 else None,
    )
    calibrator = CameraCalibrator(
        eventsource=source,
        r1_product='HESSIOR1Calibrator',
        extractor_product='NeighbourPeakIntegrator',
    )



    allowed_tels = [id for id in source._subarray_info.tels if source._subarray_info.tels[id].camera.cam_id in allowed_cameras]
    source.allowed_tels = allowed_tels
    
    event_iterator = filter(lambda e: len(e.dl0.tels_with_data) > 1, source)

    with Parallel(n_jobs=n_jobs, verbose=50, prefer='processes') as parallel:
        p = parallel(delayed(partial(process_parallel, calibrator=calibrator))(copy.deepcopy(e)) for e in event_iterator)
        # result =  [a for a in tqdm(pool.imap(partial(process_parallel, calibrator=calibrator) , event_iterator, chunksize=10))]
        result = [a for a in tqdm(p)]
    
    array_event_containers = [r[0] for r in result if r]

    # flatten according to https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
    nested_tel_events = [r[1] for r in result if r]
    telescope_event_containers = [item for sublist in nested_tel_events for item in sublist] 
    
    mc_header_container = source.mc_header_information
    mc_header_container.prefix=''
    
    run_info_container = RunInfoContainer(run_id=array_event_containers[0].run_id, mc=mc_header_container)
    
    return run_info_container, array_event_containers, telescope_event_containers


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
    hillas_container.prefix = ''
    leakage_container = leakage(camera, dl1.image[0], mask)
    leakage_container.prefix = ''
    concentration_container = concentration(camera, dl1.image[0], hillas_container)
    concentration_container.prefix = ''
    
    alt_pointing = event.mc.tel[telescope_id].altitude_raw * u.rad
    az_pointing = event.mc.tel[telescope_id].azimuth_raw * u.rad
    pointing_container = TelescopePointingContainer(azimuth=az_pointing, altitude=alt_pointing, prefix='')

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
        d = np.sqrt((core_x - x)**2 + (core_y - y)**2)

        container.distance_to_reconstructed_core_position = d

def process_event(event, calibrator):
    '''
    Processes
    '''


    telescope_types = []

    hillas_reconstructor = HillasReconstructor()

    calibrator.calibrate(event)

    telescope_event_containers = {}
    for telescope_id, dl1 in event.dl1.tel.items():
        telescope_type_name = event.inst.subarray.tels[telescope_id].optics.tel_type
        telescope_types.append(telescope_type_name)

        try:
            telescope_event_containers[telescope_id] = calculate_image_features(telescope_id, event, dl1)
        except HillasParameterizationError:
            continue


    if len(telescope_event_containers) < 2:
        raise ReconstructionError('Not enough telescopes for which Hillas parameters could be reconstructed.')

    parameters = {tel_id: telescope_event_containers[tel_id].hillas for tel_id in telescope_event_containers}
    pointing_altitude = {tel_id: telescope_event_containers[tel_id].pointing.altitude for tel_id in telescope_event_containers}
    pointing_azimuth = {tel_id: telescope_event_containers[tel_id].pointing.azimuth for tel_id in telescope_event_containers}

    reconstruction_container = hillas_reconstructor.predict(parameters, event.inst, pointing_alt=pointing_altitude, pointing_az=pointing_azimuth )
    reconstruction_container.prefix = ''
    calculate_distance_to_core(telescope_event_containers, event, reconstruction_container)
    
    mc_container = copy.deepcopy(event.mc)
    mc_container.tel = None
    mc_container.prefix = 'mc'

    counter = Counter(telescope_types)  
    array_event = ArrayEventContainer(
        array_event_id=event.dl0.event_id,
        run_id=event.r0.obs_id,
        reco=reconstruction_container,
        total_intensity=sum([t.hillas.intensity for t in telescope_event_containers.values()]),
        num_triggered_lst=counter['LST'],
        num_triggered_mst=counter['MST'],
        num_triggered_sst=counter['SST'],
        num_triggered_telescopes=len(telescope_types),
        mc=mc_container,
    )

    return array_event, list(telescope_event_containers.values())


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
