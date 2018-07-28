
from ctapipe.io.eventsourcefactory import EventSourceFactory
from ctapipe.calib import CameraCalibrator
from ctapipe.image.hillas import hillas_parameters
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.reco import HillasReconstructor
from ctapipe.reco.HillasReconstructor import TooFewTelescopesException
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt
from ctapipe.visualization import CameraDisplay
from astropy.coordinates.angle_utilities import angular_separation
from tqdm import tqdm
import click
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from collections import namedtuple
from ctapipe.coordinates import GroundFrame
from ctapipe.coordinates import TiltedGroundFrame


SubMomentParameters = namedtuple('SubMomentParameters', 'size,cen_x,cen_y,length,width,psi')


names_to_id = {'LSTCam': 1, 'NectarCam': 2, 'FlashCam': 3, 'DigiCam': 4, 'CHEC': 5}
types_to_id = {'LST': 1, 'MST': 2, 'SST': 3}
allowed_cameras = ['LSTCam', 'NectarCam', 'DigiCam']


cleaning_level = {
                    'ASTRICam': (5, 7),  # (5, 10)?
                    'FlashCam': (12, 15),
                    'LSTCam': (3.5, 6),  # ?? (3, 6) for Abelardo...
                    # ASWG Zeuthen talk by Abelardo Moralejo:
                    'NectarCam': (4, 8),
                    # "FlashCam": (4, 8),  # there is some scaling missing?
                    'DigiCam': (3, 6),
                    'CHEC': (2, 4),
                    'SCTCam': (1.5, 3)
                    }




@u.quantity_input
def calculate_distance_theta(alt_prediction: u.rad, az_prediction: u.rad, source_alt: u.rad=70 * u.deg, source_az: u.rad=0 * u.deg):
    source_az = Angle(source_az)
    source_alt = Angle(source_alt)

    az = Angle(az_prediction)
    alt = Angle(alt_prediction)

    distance = angular_separation(source_az, source_alt, az, alt).to(u.deg)

    return distance


@click.command()
@click.argument('simtel_file', type=click.Path(exists=True))
@click.argument('output_pdf', type=click.Path(exists=False))
@click.option('-n', '--num_events', default=50)
def main(simtel_file, output_pdf, num_events):

    event_source = EventSourceFactory.produce(
        input_url=simtel_file,
        max_events=num_events,
    )

    calibrator = CameraCalibrator(
        eventsource=event_source,
    )
    with PdfPages(output_pdf) as pdf:
        for event in tqdm(event_source, total=num_events):
            if event.mc.energy > 10 *u.TeV:
                calibrator.calibrate(event)
                reco = HillasReconstructor()
                plt.figure(figsize=(20, 20))
                plt.suptitle(f'EVENT {event.r0.event_id} \n Energy: {event.mc.energy} \n Type: {event.mc.shower_primary_id}')
                try:
                    result = plot_event(event, reco, pdf)
                except TooFewTelescopesException:
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close()
                    continue
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()

                d = calculate_distance_theta(result.alt, result.az, event.mc.alt, event.mc.az)
                plt.figure(figsize=(18, 18))
                plot_array_birdsview(event, result, reco)
                plt.suptitle(f'EVENT {event.r0.event_id} \n Energy: {event.mc.energy} \n Type: {event.mc.shower_primary_id} \n  \
                Alt: {event.mc.alt.to(u.deg)},   Az: {event.mc.az.to(u.deg)} \n \
                Predicted Alt: {result.alt.to(u.deg)},  Predicted Az: {result.az.to(u.deg)} \n \
                Distance {d}')
                plt.legend()
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()


                plt.figure(figsize=(18, 18))
                plot_array_sideview(event, result, reco)
                plt.suptitle(f'EVENT {event.r0.event_id} \n Energy: {event.mc.energy} \n Type: {event.mc.shower_primary_id} \n  \
                Alt: {event.mc.alt.to(u.deg)},   Az: {event.mc.az.to(u.deg)} \n \
                Predicted Alt: {result.alt.to(u.deg)},   Predicted Az: {result.az.to(u.deg)} \n \
                Distance: {d}')
                plt.legend()
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()


def plot_array_birdsview(event, result, reco):
    cams = [event.inst.subarray.tels[i].camera for i in event.r0.tels_with_data]
    cams = [c for c in cams if c.cam_id in allowed_cameras]

    pos = []
    ids = event.inst.subarray.tel.keys()
    for i in list(ids):
        if event.inst.subarray.tels[i].camera.cam_id in allowed_cameras:
            pos.append(event.inst.subarray.positions[i])

    pos = np.array(pos)
    plt.plot(pos[:, 0], pos[:, 1], '.', c='gray')

    sst = []
    mst = []
    lst = []
    ids = event.r0.tels_with_data
    for i in list(ids):
        plot_kwargs = {'scale':1.5, 'width':0.0015, 'alpha':0.2}
        if event.inst.subarray.tels[i].camera.cam_id in allowed_cameras:
            az = event.mc.tel[i].azimuth_raw
            direction = [1 * np.cos(az), 1 * np.sin(az)]
            pos = event.inst.subarray.positions[i]
            if event.inst.subarray.tels[i].camera.cam_id == 'DigiCam':
                sst.append(pos)
                plt.quiver(*pos[0:2], *direction, color='blue', **plot_kwargs)

            if event.inst.subarray.tels[i].camera.cam_id == 'NectarCam':
                mst.append(pos)
                plt.quiver(*pos[0:2], *direction, color='red', **plot_kwargs)
            if event.inst.subarray.tels[i].camera.cam_id == 'LSTCam':
                lst.append(pos)
                plt.quiver(*pos[0:2], *direction, color='green', **plot_kwargs)


    if sst:
        sst = np.array(sst)
        plt.scatter(sst[:, 0], sst[:, 1], label='SST', color='blue', s=60)

    if mst:
        mst = np.array(mst)
        plt.scatter(mst[:, 0], mst[:, 1], label='MST', color='red', s=60)

    if lst:
        lst = np.array(lst)
        plt.scatter(lst[:, 0], lst[:, 1], label='LST', color='green', s=60)

    point_dir = SkyCoord(
        *(event.mcheader.run_array_direction),
        frame='altaz'
    )
    tiltedframe = TiltedGroundFrame(pointing_direction=point_dir)
    core_coord = SkyCoord(
        x=event.mc.core_x,
        y=event.mc.core_y,
        frame=tiltedframe
    ).transform_to(GroundFrame())

    plt.scatter(core_coord.x.value, core_coord.x.value, s=150, marker='+', label='impact point', color='black')
    plt.scatter(result.core_x.value, result.core_y.value, s=150, marker='+', label='impact point estimated', color='orange')

    for c in reco.hillas_planes.values():
        plt.quiver(*c.pos.value[0:2], *c.a[0:2], scale=1.5, width=0.002, color='gray')
        plt.quiver(*c.pos.value[0:2], *c.b[0:2], scale=1.5, width=0.002, color='silver')

    direction = [1 * np.cos(event.mc.az.value), 1 * np.sin(event.mc.az.value)]
    plt.quiver(event.mc.core_x.value, event.mc.core_y.value, *direction, scale=1.5, width=0.002, color='black', alpha=0.5, label='true direction')
    direction = [1 * np.cos(result.az.to('rad').value), 1 * np.sin(result.az.to('rad').value)]
    plt.quiver(result.core_x.value, result.core_y.value, *direction, scale=1.5, width=0.002, color='orange', label='estimated direction')


def plot_array_sideview(event, result, reco):
    cams = [event.inst.subarray.tels[i].camera for i in event.r0.tels_with_data]
    cams = [c for c in cams if c.cam_id in allowed_cameras]

    pos = []
    ids = event.inst.subarray.tel.keys()
    for i in list(ids):
        if event.inst.subarray.tels[i].camera.cam_id in allowed_cameras:
            pos.append(event.inst.subarray.positions[i])

    pos = np.array(pos)
    plt.plot(pos[:, 0], pos[:, 2], '.', c='gray')

    sst = []
    mst = []
    lst = []
    ids = event.r0.tels_with_data
    for i in list(ids):
        if event.inst.subarray.tels[i].camera.cam_id in allowed_cameras:
            alt = event.mc.tel[i].altitude_raw
            direction = [1 * np.cos(alt), 1 * np.sin(alt)]
            pos = event.inst.subarray.positions[i]
            if event.inst.subarray.tels[i].camera.cam_id == 'DigiCam':
                sst.append(pos)
                plt.quiver(*pos[[0, 2]], *direction, color='blue', scale=1.5, width=0.0015, alpha=0.2)
            if event.inst.subarray.tels[i].camera.cam_id == 'NectarCam':
                mst.append(pos)
                plt.quiver(*pos[[0, 2]], *direction, color='red', scale=1.5, width=0.0015, alpha=0.2)
            if event.inst.subarray.tels[i].camera.cam_id == 'LSTCam':
                lst.append(pos)
                plt.quiver(*pos[[0, 2]], *direction, color='green', scale=1.5, width=0.0015, alpha=0.2)

    if sst:
        sst = np.array(sst)
        plt.scatter(sst[:, 0], sst[:, 2], label='SST', color='blue', s=60)

    if mst:
        mst = np.array(mst)
        plt.scatter(mst[:, 0], mst[:, 2], label='MST', color='red', s=60)

    if lst:
        lst = np.array(lst)
        plt.scatter(lst[:, 0], lst[:, 2], label='LST', color='green', s=60)

    point_dir = SkyCoord(
        *(event.mcheader.run_array_direction),
        frame='altaz'
    )
    tiltedframe = TiltedGroundFrame(pointing_direction=point_dir)
    core_coord = SkyCoord(
        x=event.mc.core_x,
        y=event.mc.core_y,
        frame=tiltedframe
    ).transform_to(GroundFrame())

    plt.scatter(core_coord.x.value, 0, s=150, marker='+', label='impact point', color='black')
    plt.scatter(result.core_x.value, 0, s=150, marker='+', label='impact point estimated', color='orange')

    for c in reco.hillas_planes.values():
        plt.quiver(*c.pos.value[[0, 2]], *c.a[[0, 2]], scale=1.5, width=0.002, color='gray')
        plt.quiver(*c.pos.value[[0, 2]], *c.b[[0, 2]], scale=1.5, width=0.002, color='silver')


    direction = [1 * np.cos(event.mc.alt.value), 1 * np.sin(event.mc.alt.value)]
    plt.quiver(event.mc.core_x.value, 0, *direction, scale=1.5, width=0.002, color='black', alpha=0.5, label='true direction')
    direction = [1 * np.cos(result.alt.to('rad').value), 1 * np.sin(result.alt.to('rad').value)]
    plt.quiver(result.core_x.value, 0, *direction, scale=1.5, width=0.002, color='orange', label='estimated direction')

    plt.plot([-1500, 1500], [0, 0], color='#987a48', lw=3, ls='--')
    plt.ylim(-5, 40)
    plt.xlim(-1500, 1500)


def plot_event(event, reco, pdf):
    cams = [event.inst.subarray.tels[i].camera for i in event.r0.tels_with_data]
    cams = [c for c in cams if c.cam_id in allowed_cameras]
    n_tels = len(cams)

    p = 1
    params = {}
    pointing_azimuth = {}
    pointing_altitude = {}


    for telescope_id, dl1 in event.dl1.tel.items():
        camera = event.inst.subarray.tels[telescope_id].camera
        if camera.cam_id not in allowed_cameras:
            continue

        nn = int(np.ceil(np.sqrt(n_tels)))
        ax = plt.subplot(nn, nn, p)
        p += 1

        boundary_thresh, picture_thresh = cleaning_level[camera.cam_id]
        mask = tailcuts_clean(camera, dl1.image[0], boundary_thresh=boundary_thresh, picture_thresh=picture_thresh, min_number_picture_neighbors=1)
        #
        if mask.sum() < 3:  # only two pixel remaining. No luck anyways.
            continue

        h = hillas_parameters(
            camera[mask],
            dl1.image[0, mask],
        )

        disp = CameraDisplay(camera, ax=ax, title="CT{0}".format(telescope_id))
        disp.pixels.set_antialiaseds(False)
        disp.autoupdate = False
        disp.add_colorbar()

        # Show the camera image and overlay Hillas ellipse and clean pixels
        disp.image = dl1.image[0]
        disp.cmap = 'viridis'
        disp.highlight_pixels(mask, color='white')
        disp.overlay_moments(h, color='red', linewidth=5)

        pointing_azimuth[telescope_id] = event.mc.tel[telescope_id].azimuth_raw * u.rad
        pointing_altitude[telescope_id] = event.mc.tel[telescope_id].altitude_raw * u.rad
        params[telescope_id] = h

    return reco.predict(params, event.inst, pointing_altitude, pointing_azimuth)


if __name__ == '__main__':
    main()
