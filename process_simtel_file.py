from collections import Counter
import click
import numpy as np
from tqdm import tqdm
import astropy.units as u
import logging

import copy
from functools import partial
import os

from joblib import delayed, Parallel

from ctapipe.io.containers import TelescopePointingContainer, MCHeaderContainer
from ctapipe.io.eventsource import event_source
from ctapipe.io import HDF5TableWriter
from ctapipe.calib import CameraCalibrator
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError
from ctapipe.image import leakage, concentration
from ctapipe.image.cleaning import tailcuts_clean, fact_image_cleaning, number_of_islands
from ctapipe.image.timing_parameters import timing_parameters
from ctapipe.reco import HillasReconstructor
from ctapipe.io.containers import ReconstructedShowerContainer

from astropy.coordinates import SkyCoord, AltAz

from preprocessing.parameters import PREPConfig
from preprocessing.containers import (
    TelescopeParameterContainer,
    ArrayEventContainer,
    RunInfoContainer,
    IslandContainer,
)


class ReconstructionError(Exception):
    pass


@click.command()
@click.argument('input_file', type=click.Path())
@click.argument('output_file', type=click.Path())
@click.argument('config_file', type=click.Path(file_okay=True))
@click.option('-n', '--n_events', default=-1, help='number of events to process in each file.')
@click.option(
    '-j',
    '--n_jobs',
    default=1,
    help='number of jobs to start.' 'this is usefull when passing more than one simtel file.',
)
@click.option(
    '--overwrite/--no-overwrite',
    default=False,
    help='If false (default) will only process non-existing filenames',
)
@click.option('-v', '--verbose', default=1, help='specifies the output being shown during processing')
def main(input_file, output_file, config_file, n_events, n_jobs, overwrite, verbose):  # missing logger config
    '''
    ####DEPRECATED######
    THERE HASNT BEEN ANY WORK ON THIS FOR A WHILE
    USE process_multiple_simtel_files.py
    IT ALSO WORKS FOR A SINGLE FILE
    ####################
    Process a single simtel files given as
    'input_file'
    into one hdf5 file saved in
    'output_file'.
    Processing steps consist of:
    - Calibration
    - Calculating image features
    - Collecting MC header information
    The hdf5 file will contain three groups.
    'runs', 'array_events', 'telescope_events'.

    The config specifies which
    - telescopes
    - integrator
    - cleaning algorithm
    - cleaning levels per telescope type
    to use.

    These files can be put into the classifier tools for learning.
    See https://github.com/fact-project/classifier-tools
    '''

    config = PREPConfig(config_file)
    logging.info(f'Processing file {input_file}, writing to {output_file}')

    if not overwrite:
        if os.path.exists(output_file):
            logging.critical(f'Output file exists. Stopping.')
            return
    else:
        if os.path.exists(output_file):
            logging.warning(f'Output file exists. Overwriting.')
            os.remove(output_file)

    r = process_file(input_file, config, n_jobs=n_jobs, n_events=n_events, verbose=verbose)
    if r:
        run_info_container, array_events, telescope_events = r
        write_result_to_file(run_info_container, array_events, telescope_events, output_file)

    else:
        logging.critical('could not process file')


def write_result_to_file(run_info_container, array_events, telescope_events, output_file, mode='a'):
    '''Combines run_info, array_events and telescope_events
    into one file using the HDF5TableWriter.
    '''
    logging.info(f'writing to file {output_file}')
    with HDF5TableWriter(output_file, mode=mode, group_name='', add_prefix=True) as h5_table:
        run_info_container.mc['run_array_direction'] = 0
        h5_table.write('runs', [run_info_container, run_info_container.mc])
        for array_event in array_events:
            h5_table.write('array_events', [array_event, array_event.mc, array_event.reco])

        for tel_event in telescope_events:
            h5_table.write(
                'telescope_events',
                [
                    tel_event,
                    tel_event.pointing,
                    tel_event.hillas,
                    tel_event.concentration,
                    tel_event.leakage,
                    tel_event.timing,
                    tel_event.islands,
                ],
            )


def process_parallel(event, calibrator, config):
    'try block to avoid crashes during parallel processing. event will be skipped'
    try:
        return process_event(event, calibrator, config)
    except Exception:
        logging.warning('Error processing event', exc_info=True)
        pass


def process_file(input_file, config, n_jobs=1, n_events=-1, verbose=1):
    '''Performs all needed steps to process a given file according to a given config.
    Loads the file, chooses the defined telescopes, processes events and gets the MC header
    - Event processing is handled in process_event, mc header is handled in get_mc_header - 
    '''
    logging.info(f'processing file {input_file} in parallel with {n_jobs} jobs')
    allowed_tels = config.allowed_ids
    try:
        source = event_source(
            input_url=input_file, max_events=n_events if n_events > 1 else None, allowed_tels=allowed_tels
        )
    except (EOFError, StopIteration):
        logging.error(f'Could not produce eventsource. File might be truncated? {input_file}', exc_info=True)
        return None
    except Exception:
        logging.critical('Unhandled error trying to get the event source', exc_info=True)
    calibrator = CameraCalibrator()

    logging.info(f'Defined allowed telescopes to {source.allowed_tels}')

    # choose only events with at least one telescope
    event_iterator = filter(lambda e: len(e.dl0.tels_with_data) > 1, source)

    with Parallel(n_jobs=n_jobs, verbose=verbose, backend='loky') as parallel:
        p = parallel(
            delayed(partial(process_parallel, calibrator=calibrator, config=config))(copy.deepcopy(e))
            for e in event_iterator
        )
        result = [a for a in tqdm(p)]

    array_event_containers = [r[0] for r in result if r]

    nested_tel_events = [r[1] for r in result if r]
    # flatten according to https://stackoverflow.com/questions/952914
    telescope_event_containers = [item for sublist in nested_tel_events for item in sublist]

    ## super hacky, im sorry (necessary because our iterator is exhausted)
    mc_header_container = None
    mc_source = event_source(input_url=input_file, max_events=1)
    try:
        mc_header_container = get_mc_header(mc_source)
        mc_header_container.prefix = 'mc'
        logging.info(f'Got mc header for file {input_file}')
    except Exception:
        logging.error(f'Could not get mc header for file {input_file}', exc_info=True)
        raise

    if len(array_event_containers) > 0:
        logging.debug('array event container seems valid. returning containers')
        run_info_container = RunInfoContainer(run_id=array_event_containers[0].run_id, mc=mc_header_container)
        return run_info_container, array_event_containers, telescope_event_containers

    logging.error(f'Could not gather data from file. File might be truncated or just empty? {input_file}')
    return None


def calculate_image_features(telescope_id, event, dl1, config):
    ''' Performs cleaning and adds the following image parameters:
    - hillas
    - leakage
    - concentration
    - timing
    - number of islands

    Make sure to adapt cleaning levels to the used algorithm (-> config)
    - tailcuts:
        picture_thresh, picture_thresh, min_number_picture_neighbors
    - fact_image_cleaning:
        picture_threshold, boundary_threshold, min_number_neighbors, time_limit
    
    Returns:
    --------
    TelescopeParameterContainer
    '''
    array_event_id = event.dl0.event_id
    run_id = event.r0.obs_id
    telescope = event.inst.subarray.tels[telescope_id]
    image = dl1.image

    # might want to make the parameter names more consistent between methods
    if config.cleaning_method == 'tailcuts_clean':
        boundary_thresh, picture_thresh, min_number_picture_neighbors = config.cleaning_level[
            telescope.camera.cam_id
        ]
        mask = tailcuts_clean(
            telescope.camera,
            image,
            boundary_thresh=boundary_thresh,
            picture_thresh=picture_thresh,
            min_number_picture_neighbors=min_number_picture_neighbors,
        )
    elif config.cleaning_method == 'fact_image_cleaning':
        boundary_threshold, picture_threshold, time_limit, min_number_neighbors = config.cleaning_level[
            telescope.camera.cam_id
        ]
        mask = fact_image_cleaning(
            telescope.camera,
            image,
            dl1.pulse_time,
            boundary_threshhold=boundary_threshold,
            picture_threshold=picture_threshold,
            min_number_neighbors=min_number_neighbors,
            time_limit=time_limit,
        )

    cleaned = image.copy()
    cleaned[~mask] = 0
    logging.debug(f'calculating hillas for event {array_event_id, telescope_id}')
    hillas_container = hillas_parameters(telescope.camera, cleaned)
    hillas_container.prefix = ''
    logging.debug(f'calculating leakage for event {array_event_id, telescope_id}')
    leakage_container = leakage(telescope.camera, image, mask)
    leakage_container.prefix = ''
    logging.debug(f'calculating concentration for event {array_event_id, telescope_id}')
    concentration_container = concentration(telescope.camera, image, hillas_container)
    concentration_container.prefix = ''
    logging.debug(f'getting timing information for event {array_event_id, telescope_id}')
    timing_container = timing_parameters(telescope.camera, image, dl1.pulse_time, hillas_container)
    timing_container.prefix = ''

    # membership missing for now as it causes problems with the hdf5tablewriter
    # right now i dont need this anyway
    logging.debug(f'calculating num_islands for event {array_event_id, telescope_id}')
    num_islands, membership = number_of_islands(telescope.camera, mask)
    island_container = IslandContainer(num_islands=num_islands)
    island_container.prefix = ''
    num_pixel_in_shower = mask.sum()

    logging.debug(f'getting pointing container for event {array_event_id, telescope_id}')

    # ctapipe requires this to be rad
    pointing_container = TelescopePointingContainer(
        azimuth=event.mc.tel[telescope_id].azimuth_raw * u.rad,
        altitude=event.mc.tel[telescope_id].altitude_raw * u.rad,
        prefix='pointing',
    )

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
        telescope_type_id=config.types_to_id[telescope.type],
        camera_type_id=config.names_to_id[telescope.camera.cam_id],
        focal_length=telescope.optics.equivalent_focal_length,
        mirror_area=telescope.optics.mirror_area,
        num_pixel_in_shower=num_pixel_in_shower,
    )


def calculate_distance_to_core(tel_params, event, reconstruction_result):
    'Calculates distance to reconstructed core position for one event filling the provided container'
    for tel_id, container in tel_params.items():
        try:
            pos = event.inst.subarray.positions[tel_id]
            x, y = pos[0], pos[1]
            core_x = reconstruction_result.core_x
            core_y = reconstruction_result.core_y
            d = np.sqrt((core_x - x) ** 2 + (core_y - y) ** 2)

            container.distance_to_reconstructed_core_position = d
        except Exception:
            container.distance_to_reconstructed_core_position = np.nan
            logging.info('No properly reconstructed distance for tel_id %s', tel_id)


def process_event(event, calibrator, config):
    '''Processes one event.
    Calls calculate_image_features and performs a stereo hillas reconstruction.
    ReconstructedShowerContainer will be filled with nans (+units) if hillas failed.

    Returns:
    --------
    ArrayEventContainer
    list(telescope_event_containers.values())
    '''

    logging.info(f'processing event {event.dl0.event_id}')
    calibrator(event)

    telescope_types = []
    hillas_reconstructor = HillasReconstructor()
    telescope_event_containers = {}
    horizon_frame = AltAz()
    telescope_pointings = {}
    array_pointing = SkyCoord(
        az=event.mcheader.run_array_direction[0],
        alt=event.mcheader.run_array_direction[1],
        frame=horizon_frame,
    )

    for telescope_id, dl1 in event.dl1.tel.items():
        cam_type = event.inst.subarray.tels[telescope_id].camera.cam_id
        if cam_type not in config.cleaning_level.keys():
            logging.info(f"No cleaning levels for camera {cam_type}. Skipping event")
            continue
        telescope_types.append(str(event.inst.subarray.tels[telescope_id].optics))
        try:
            telescope_event_containers[telescope_id] = calculate_image_features(
                telescope_id, event, dl1, config
            )
        except HillasParameterizationError:
            logging.info(
                f'Error calculating hillas features for event {event.dl0.event_id, telescope_id}',
                exc_info=True,
            )
            continue
        except Exception:
            logging.info(
                f'Error calculating image features for event {event.dl0.event_id, telescope_id}',
                exc_info=True,
            )
            continue
        telescope_pointings[telescope_id] = SkyCoord(
            alt=telescope_event_containers[telescope_id].pointing.altitude,
            az=telescope_event_containers[telescope_id].pointing.azimuth,
            frame=horizon_frame,
        )

    if len(telescope_event_containers) < 1:
        raise Exception('None of the allowed telescopes triggered for event %s', event.dl0.event_id)

    parameters = {tel_id: telescope_event_containers[tel_id].hillas for tel_id in telescope_event_containers}

    try:
        reconstruction_container = hillas_reconstructor.predict(
            parameters, event.inst, array_pointing, telescopes_pointings=telescope_pointings
        )
        reconstruction_container.prefix = ''
    except Exception:
        logging.info(
            'Not enough telescopes for which Hillas parameters could be reconstructed', exc_info=True
        )
        reconstruction_container = ReconstructedShowerContainer()
        reconstruction_container.alt = u.Quantity(np.nan, u.rad)
        reconstruction_container.alt_uncert = u.Quantity(np.nan, u.rad)
        reconstruction_container.az = u.Quantity(np.nan, u.rad)
        reconstruction_container.az_uncert = np.nan
        reconstruction_container.core_x = u.Quantity(np.nan, u.m)
        reconstruction_container.core_y = u.Quantity(np.nan, u.m)
        reconstruction_container.core_uncert = np.nan
        reconstruction_container.h_max = u.Quantity(np.nan, u.m)
        reconstruction_container.h_max_uncert = np.nan
        reconstruction_container.is_valid = False
        reconstruction_container.tel_ids = []
        reconstruction_container.average_intensity = np.nan
        reconstruction_container.goodness_of_fit = np.nan
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


def get_mc_header(event_source):
    'Returns a mc header from a simtel event source taking the header of the last event'

    # yes, this is dirty
    for last_event in event_source:
        pass
    mc_header_container = MCHeaderContainer()
    mc_header_container.update(**last_event.mcheader)

    return mc_header_container


if __name__ == '__main__':
    main()
