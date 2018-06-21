from ctapipe.io.eventsourcefactory import EventSourceFactory
from ctapipe.calib import CameraCalibrator
from ctapipe.image.hillas import hillas_parameters_5, HillasParameterizationError
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.reco import HillasReconstructor, HillasIntersection
from joblib import Parallel, delayed

import pandas as pd
import fact.io
import click
import os
import pyhessio
import numpy as np
from collections import namedtuple, Counter
from tqdm import tqdm
import astropy.units as u
from astropy.coordinates import SkyCoord

# do some horrible things to silence warnings in ctapipe
import warnings
from astropy.utils.exceptions import AstropyDeprecationWarning
warnings.filterwarnings('ignore', category=AstropyDeprecationWarning, append=True)
warnings.filterwarnings('ignore', category=FutureWarning, append=True)
np.warnings.filterwarnings('ignore')


names_to_id = {'LSTCam': 1, 'NectarCam': 2, 'FlashCam': 3, 'DigiCam': 4, 'CHEC': 5}
types_to_id = {'LST': 1, 'MST': 2, 'SST': 3}
allowed_cameras = ['LSTCam', 'NectarCam', 'DigiCam']


cleaning_level = {
    # 'ASTRICam': (5, 7, 2),  # (5, 10)?
    # 'FlashCam': (12, 15, 2),
    'LSTCam': (3.5, 7.5, 2),  # ?? (3, 6) for Abelardo...
    # ASWG Zeuthen talk by Abelardo Moralejo:
    'NectarCam': (3, 5.5, 2),
    # "FlashCam": (4, 8),  # there is some scaling missing?
    'DigiCam': (2, 4.5, 2),
    # 'CHEC': (2, 4, 2),
    # 'SCTCam': (1.5, 3, 2)
}


@click.command()
@click.argument(
    'input_files', type=click.Path(
        exists=True,
        dir_okay=False,
    ), nargs=-1)
@click.argument(
    'output_file', type=click.Path(
        dir_okay=False,
    ))
@click.option('-n', '--n_events', default=-1, help='number of events to process in each file.')
@click.option('-j', '--n_jobs', default=1, help='number of jobs to start. this is usefull when passing more than one simtel file.')
@click.option('-r', '--reco_algorithm', default='planes', type=click.Choice(['intersection', 'planes']), help='Reco Algorithm to use')
def main(input_files, output_file, n_events, n_jobs, reco_algorithm):
    '''
    process multiple simtel files gievn as INPUT_FILES into one hdf5 file saved in OUTPUT_FILE.
    The hdf5 file will contain three groups. 'runs', 'array_events', 'telescope_events'.

    These files can be put into the classifier tools for learning.
    See https://github.com/fact-project/classifier-tools

    '''

    if os.path.exists(output_file):
        click.confirm('File {} exists. Overwrite?'.format(output_file), default=False, abort=True)
        os.remove(output_file)

    if n_jobs > 1:
        results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(process_file)(f, reco_algorithm=reco_algorithm, n_events=n_events, silent=True,) for f in input_files)
        for r in results:
            runs, array_events, telescope_events = r

            if runs is None or array_events is None or telescope_events is None:
                continue

            fact.io.write_data(runs, output_file, key='runs', mode='a')
            fact.io.write_data(array_events, output_file, key='array_events', mode='a')
            fact.io.write_data(telescope_events, output_file, key='telescope_events', mode='a')
    else:
        for input_file in input_files:
            print('processing file {}'.format(input_file))
            runs, array_events, telescope_events = process_file(input_file, reco_algorithm=reco_algorithm, n_events=n_events)

            if runs is None or array_events is None or telescope_events is None:
                print('file contained no information.')
                continue

            fact.io.write_data(runs, output_file, key='runs', mode='a')
            fact.io.write_data(array_events, output_file, key='array_events', mode='a')
            fact.io.write_data(telescope_events, output_file, key='telescope_events', mode='a')

    verify_file(output_file)


def process_file(input_file, reco_algorithm, n_events=-1, silent=False):
    event_source = EventSourceFactory.produce(
        input_url=input_file,
        max_events=n_events if n_events > 1 else None,
        product='HESSIOEventSource',
    )
    calibrator = CameraCalibrator(
        eventsource=event_source,
    )


    telescope_event_information = []
    array_event_information = []
    for event in tqdm(event_source, disable=silent):
        if number_of_valid_triggerd_cameras(event) < 2:
            continue

        calibrator.calibrate(event)
        try:
            image_features, reconstruction, _, _ = process_event(event, reco_algorithm=reco_algorithm)
            if len(image_features) > 1:  # check whtehr at least two telescopes returned hillas features
                event_features = event_information(event, image_features, reconstruction)
                array_event_information.append(event_features)
                telescope_event_information.append(image_features)
        except HillasParameterizationError:
            continue  # no signal in event or whatever kind of shit can happen here.

    if (len(telescope_event_information) == 0):
        return None, None, None

    telescope_events = pd.concat(telescope_event_information)
    telescope_events.set_index(['run_id', 'array_event_id', 'telescope_id'], drop=True, verify_integrity=True, inplace=True)


    array_events = pd.DataFrame(array_event_information)
    array_events.set_index(['run_id', 'array_event_id'], drop=True, verify_integrity=True, inplace=True)


    run_information = read_simtel_mc_information(input_file)
    df_runs = pd.DataFrame([run_information])
    df_runs.set_index('run_id', drop=True, verify_integrity=True, inplace=True)

    return df_runs, array_events, telescope_events



def verify_file(input_file_path):
    runs = fact.io.read_data(input_file_path, key='runs')
    runs.set_index('run_id', drop=True, verify_integrity=True, inplace=True)

    telescope_events = fact.io.read_data(input_file_path, key='telescope_events')
    telescope_events.set_index(['run_id', 'array_event_id', 'telescope_id'], drop=True, verify_integrity=True, inplace=True)

    array_events = fact.io.read_data(input_file_path, key='array_events')
    array_events.set_index(['run_id', 'array_event_id'], drop=True, verify_integrity=True, inplace=True)

    print('Processed {} runs, {} single telescope events and {} array events.'.format(len(runs), len(telescope_events), len(array_events)))


def read_simtel_mc_information(simtel_file):
    with pyhessio.open_hessio(simtel_file) as f:
        # do some weird hessio fuckup
        eventstream = f.move_to_next_event()
        _ = next(eventstream)

        d = {
            'mc_spectral_index': f.get_spectral_index(),
            'mc_num_reuse': f.get_mc_num_use(),
            'mc_num_showers': f.get_mc_num_showers(),
            'mc_max_energy': f.get_mc_E_range_Max(),
            'mc_min_energy': f.get_mc_E_range_Min(),
            'mc_max_scatter_range': f.get_mc_core_range_Y(),  # range_X is always 0 in simtel files
            'mc_max_viewcone_radius': f.get_mc_viewcone_Max(),
            'mc_min_viewcone_radius': f.get_mc_viewcone_Min(),
            'run_id': f.get_run_number(),
            'mc_max_altitude': f.get_mc_alt_range_Max(),
            'mc_max_azimuth': f.get_mc_az_range_Max(),
            'mc_min_altitude': f.get_mc_alt_range_Min(),
            'mc_min_azimuth': f.get_mc_az_range_Min(),
        }


        return d


def event_information(event, image_features, reconstruction):
    counter = Counter(image_features.telescope_type_name)
    d = {
        'mc_alt': event.mc.alt,
        'mc_az': event.mc.az,
        'mc_core_x': event.mc.core_x,
        'mc_core_y': event.mc.core_y,
        'mc_x_max': event.mc.x_max.to(u.g / u.cm**2).value,
        'num_triggered_telescopes': number_of_valid_triggerd_cameras(event),
        'mc_height_first_interaction': event.mc.h_first_int,
        'mc_energy': event.mc.energy.to('TeV').value,
        'mc_corsika_primary_id': event.mc.shower_primary_id,
        'run_id': event.r0.obs_id,
        'array_event_id': event.dl0.event_id,
        'alt_prediction': reconstruction.alt.si.value,
        'az_prediction': reconstruction.az.si.value,
        'core_x_prediction': reconstruction.core_x,
        'core_y_prediction': reconstruction.core_y,
        'h_max_prediction': reconstruction.h_max,
        'total_intensity': image_features.intensity.sum(),
        'num_triggered_lst': counter['LST'],
        'num_triggered_mst': counter['MST'],
        'num_triggered_sst': counter['SST'],
    }

    return {k: strip_unit(v) for k, v in d.items()}


def process_event(event, reco_algorithm='planes'):
    '''
    Processes
    '''

    features = {}
    params = {}
    cleaning_mask = {}
    pointing_azimuth = {}
    pointing_altitude = {}
    tel_x = {}
    tel_y = {}
    tel_focal_lengths = {}
    for telescope_id, dl1 in event.dl1.tel.items():
        camera = event.inst.subarray.tels[telescope_id].camera
        if camera.cam_id not in allowed_cameras:
            continue

        telescope_type_name = event.inst.subarray.tels[telescope_id].optics.tel_type
        boundary_thresh, picture_thresh, min_number_picture_neighbors = cleaning_level[camera.cam_id]
        mask = tailcuts_clean(
            camera,
            dl1.image[0],
            boundary_thresh=boundary_thresh,
            picture_thresh=picture_thresh,
            min_number_picture_neighbors=min_number_picture_neighbors
        )
        cleaning_mask[telescope_id] = mask
        try:
            h = hillas_parameters_5(
                camera[mask],
                dl1.image[0, mask],
            )
            params[telescope_id] = h
        except HillasParameterizationError:
            continue

        pointing_azimuth[telescope_id] = event.mc.tel[telescope_id].azimuth_raw * u.rad
        pointing_altitude[telescope_id] = event.mc.tel[telescope_id].altitude_raw * u.rad
        tel_x[telescope_id] = event.inst.subarray.positions[telescope_id][0]
        tel_y[telescope_id] = event.inst.subarray.positions[telescope_id][1]

        telescope_description = event.inst.subarray.tel[telescope_id]
        tel_focal_lengths[telescope_id] = telescope_description.optics.equivalent_focal_length

        d = {
            'array_event_id': event.dl0.event_id,
            'telescope_id': int(telescope_id),
            'camera_name': camera.cam_id,
            'camera_id': names_to_id[camera.cam_id],
            'run_id': event.r0.obs_id,
            'telescope_type_name': telescope_type_name,
            'telescope_type_id': types_to_id[telescope_type_name],
            'pointing_azimuth': event.mc.tel[telescope_id].azimuth_raw,
            'pointing_altitude': event.mc.tel[telescope_id].altitude_raw,
            'mirror_area': telescope_description.optics.mirror_area,
            'focal_length': telescope_description.optics.equivalent_focal_length,
        }

        d.update(h.as_dict())
        features[telescope_id] = ({k: strip_unit(v) for k, v in d.items()})

    if reco_algorithm == 'intersection':
        reco = HillasIntersection()
        array_direction = SkyCoord(alt=event.mcheader.run_array_direction[1], az=event.mcheader.run_array_direction[0], frame='altaz')
        reconstruction = reco.predict(params, tel_x, tel_y, tel_focal_lengths, array_direction)


    elif reco_algorithm == 'planes':
        reco = HillasReconstructor()
        reco.estimate_h_max = lambda a, b, c, d: np.nan
        reconstruction = reco.predict(params, event.inst, pointing_altitude, pointing_azimuth)

    for telescope_id in event.dl1.tel.keys():
        if telescope_id not in params:
            continue
        camera = event.inst.subarray.tels[telescope_id].camera
        if camera.cam_id not in allowed_cameras:
            continue

        pos = event.inst.subarray.positions[telescope_id]
        x, y = pos[0], pos[1]
        core_x = reconstruction.core_x
        core_y = reconstruction.core_y
        d = np.sqrt((core_x - x)**2 + (core_y - y)**2)
        features[telescope_id]['distance_to_core'] = d.value

    return pd.DataFrame(list(features.values())), reconstruction, params, cleaning_mask



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
