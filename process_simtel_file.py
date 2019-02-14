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
from ctapipe.image.cleaning import tailcuts_clean, number_of_islands
from ctapipe.image.timing_parameters import timing_parameters
from ctapipe.reco import HillasReconstructor

from preprocessing.parameters import PREPConfig
from preprocessing.containers import TelescopeParameterContainer, ArrayEventContainer, RunInfoContainer, IslandContainer
from ctapipe.io.containers import TelescopePointingContainer


class ReconstructionError(Exception):
    pass


@click.command()
@click.argument('input_file', type=click.Path())
@click.argument('output_file', type=click.Path())
@click.argument('config_file',
                type=click.Path(file_okay=True)
                )
@click.option('-n', '--n_events', default=-1, help='number of events to process in each file.')
@click.option('-j', '--n_jobs', default=1, help='number of jobs to start. this is usefull when passing more than one simtel file.')
@click.option('--overwrite/--no-overwrite', default=False, help='If false (default) will only process non-existing filenames')
@click.option('-v', '--verbose', default=1, help='specifies the output being shown during processing')
def main(input_file, output_file, config_file, n_events, n_jobs, overwrite, verbose):
    '''
    process simtel files given as INPUT_FILE into one hdf5 file
    saved in OUTPUT_FILE.
    The hdf5 file will contain three groups.
    'runs', 'array_events', 'telescope_events'.

    These files can be put into the classifier tools for learning.
    See https://github.com/fact-project/classifier-tools
    '''

    config = PREPConfig(config_file)
    print(f'processing file {input_file}, writing to {output_file}')

    if not overwrite:
        if os.path.exists(output_file):
            print(f'Output file exists. Stopping.')
            return
    else:
        if os.path.exists(output_file):
            print(f'Output file exists. Overwriting.')
            os.remove(output_file)

    run_info_container, array_events, telescope_events = process_file(input_file, config, n_jobs=n_jobs, n_events=n_events, verbose=verbose)
    write_result_to_file(run_info_container,
                         array_events,
                         telescope_events,
                         output_file)


def write_result_to_file(run_info_container,
                         array_events,
                         telescope_events,
                         output_file,
                         mode='a'):
    with HDF5TableWriter(output_file,
                         mode=mode,
                         group_name='',
                         add_prefix=True) as h5_table:
        run_info_container.mc.run_array_direction = 0
        h5_table.write('runs', [run_info_container, run_info_container.mc])
        for array_event in array_events:
            h5_table.write('array_events',
                           [array_event,
                            array_event.mc,
                            array_event.reco])

        for tel_event in telescope_events:
            h5_table.write(
                'telescope_events',
                [tel_event,
                 tel_event.pointing,
                 tel_event.hillas,
                 tel_event.concentration,
                 tel_event.leakage,
                 tel_event.timing,
                 tel_event.islands]
            )


def process_parallel(event, calibrator, config):
    try:
        return process_event(event, calibrator, config)
    except ReconstructionError:
        pass


def print_info(event):
    print(event.dl0.event_id)


def process_file(input_file, config, n_jobs=1, n_events=-1, verbose=1):
    source = EventSourceFactory.produce(
        input_url=input_file,
        max_events=n_events if n_events > 1 else None,
    )
    calibrator = CameraCalibrator(
        eventsource=source,
        r1_product='HESSIOR1Calibrator',
        extractor_product=config.integrator,
    )

    allowed_tels = [id for id in source._subarray_info.tels if source._subarray_info.tels[id].camera.cam_id in config.allowed_cameras]
    source.allowed_tels = allowed_tels
    
    event_iterator = filter(lambda e: len(e.dl0.tels_with_data) > 1, source)

    with Parallel(n_jobs=n_jobs,
                  verbose=verbose,
                  prefer='processes') as parallel:
        p = parallel(delayed(partial(process_parallel, calibrator=calibrator, config=config))(copy.deepcopy(e)) for e in event_iterator)
        result = [a for a in tqdm(p)]

    array_event_containers = [r[0] for r in result if r]

    # flatten according to https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
    nested_tel_events = [r[1] for r in result if r]
    telescope_event_containers = [item for sublist in nested_tel_events for item in sublist]

    mc_header_container = source.mc_header_information
    mc_header_container.prefix='mc'
    
    run_info_container = RunInfoContainer(run_id=array_event_containers[0].run_id, mc=mc_header_container)
    # from IPython import embed; embed()
    return run_info_container, array_event_containers, telescope_event_containers


def calculate_image_features(telescope_id, event, dl1, config):
    array_event_id = event.dl0.event_id
    run_id = event.r0.obs_id
    camera = event.inst.subarray.tels[telescope_id].camera

    # Make sure to adapt cleaning level to the used algorithm
    # tailcuts: picture_thresh, picture_thresh, min_number_picture_neighbors
    # fact: picture_threshold, boundary_threshold, min_number_neighbors, time_limit
    valid_cleaning_methods = ['tailcuts_clean', 'fact_image_cleaning']

    if config.cleaning_method == 'tailcuts_clean':
        mask = tailcuts_clean(
            camera,
            dl1.image[0],
            *config.cleaning_level[camera.cam_id]
        )
    elif config.cleaning_method == 'fact_image_cleaning':
        mask = fact_image_cleaning(
            camera,
            dl1.image[0],
            dl1.peakpos[0],
            *config.cleaning_level[camera.cam_id]
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
    concentration_container = concentration(camera,
                                            dl1.image[0],
                                            hillas_container)
    concentration_container.prefix = ''
    timing_container = timing_parameters(camera, dl1.image[0],
                                         dl1.peakpos[0], hillas_container)
    timing_container.prefix = ''
    # membership missing for now as it causes problems with the hdf5tablewriter 
    num_islands, membership = number_of_islands(camera, mask)
    island_container = IslandContainer(
        num_islands=num_islands)
    island_container.prefix = ''


    alt_pointing = event.mc.tel[telescope_id].altitude_raw * u.rad
    az_pointing = event.mc.tel[telescope_id].azimuth_raw * u.rad
    pointing_container = TelescopePointingContainer(azimuth=az_pointing, altitude=alt_pointing, prefix='pointing')

    optics = event.inst.subarray.tels[telescope_id].optics
    
    return TelescopeParameterContainer(
        telescope_id=telescope_id,
        run_id=run_id,
        array_event_id=array_event_id,
        leakage=leakage_container,
        hillas=hillas_container,
        concentration=concentration_container,
        pointing=pointing_container,
        timing=timing_container,
        islands=island_container,
        telescope_type_id=config.types_to_id[optics.tel_type], #  dict in config?
        camera_type_id=config.names_to_id[camera.cam_id], # dict in config?
        focal_length=optics.equivalent_focal_length,
        mirror_area=optics.mirror_area,
    )


def calculate_distance_to_core(tel_params, event, reconstruction_result):
    for tel_id, container in tel_params.items():
        pos = event.inst.subarray.positions[tel_id]
        x, y = pos[0], pos[1]
        core_x = reconstruction_result.core_x
        core_y = reconstruction_result.core_y
        d = np.sqrt((core_x - x)**2 + (core_y - y)**2)

        container.distance_to_reconstructed_core_position = d


def process_event(event, calibrator, config):
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
            telescope_event_containers[telescope_id] = calculate_image_features(telescope_id, event, dl1, config)
        except HillasParameterizationError:
            continue

    if len(telescope_event_containers) < 2:
        raise ReconstructionError('Not enough telescopes for which Hillas parameters could be reconstructed.')

    parameters = {tel_id: telescope_event_containers[tel_id].hillas for tel_id in telescope_event_containers}
    pointing_altitude = {tel_id: telescope_event_containers[tel_id].pointing.altitude for tel_id in telescope_event_containers}
    pointing_azimuth = {tel_id: telescope_event_containers[tel_id].pointing.azimuth for tel_id in telescope_event_containers}

    reconstruction_container = hillas_reconstructor.predict(parameters,
                                                            event.inst,
                                                            pointing_alt=pointing_altitude,
                                                            pointing_az=pointing_azimuth)
    reconstruction_container.prefix = ''
    calculate_distance_to_core(telescope_event_containers,
                               event,
                               reconstruction_container)

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
