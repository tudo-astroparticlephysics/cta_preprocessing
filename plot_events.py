from ctapipe.io.eventsourcefactory import EventSourceFactory
from ctapipe.calib import CameraCalibrator
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt
from ctapipe.visualization import CameraDisplay
from ctapipe.reco.HillasReconstructor import TooFewTelescopesException


from process_simtel import process_event

from tqdm import tqdm
import click


names_to_id = {'LSTCam': 1, 'NectarCam': 2, 'FlashCam': 3, 'DigiCam': 4, 'CHEC': 5}
types_to_id = {'LST': 1, 'MST': 2, 'SST': 3}
allowed_cameras = ['LSTCam', 'NectarCam', 'DigiCam']


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
            calibrator.calibrate(event)

            try:
                _, _, hillas_container, cleaning_mask = process_event(event)
            except TooFewTelescopesException:
                continue

            plt.figure(figsize=(20, 20))
            plt.suptitle(f'EVENT {event.r0.event_id} \n Energy: {event.mc.energy} \n Type: {event.mc.shower_primary_id}')
            plot_event(event, hillas_container, cleaning_mask)
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()


def plot_event(event, hillas_container, cleaning_mask):
    cams = [event.inst.subarray.tels[i].camera for i in event.r0.tels_with_data]
    cams = [c for c in cams if c.cam_id in allowed_cameras]
    n_tels = len(cams)
    p = 1

    for telescope_id, dl1 in event.dl1.tel.items():
        camera = event.inst.subarray.tels[telescope_id].camera
        if camera.cam_id not in allowed_cameras:
            continue

        nn = int(np.ceil(np.sqrt(n_tels)))
        ax = plt.subplot(nn, nn, p)
        p += 1


        disp = CameraDisplay(camera, ax=ax, title=f'CT{telescope_id}, {camera}')
        disp.pixels.set_antialiaseds(False)
        disp.autoupdate = False
        disp.add_colorbar()

        # Show the camera image and overlay Hillas ellipse and clean pixels
        disp.image = dl1.image[0]
        disp.cmap = 'viridis'

        if telescope_id not in hillas_container:
            ax.text(0, 0, 'No Signal!', color='white')
            continue

        disp.highlight_pixels(cleaning_mask[telescope_id], color='white')
        disp.overlay_moments(hillas_container[telescope_id], color='crimson', linewidth=3)


if __name__ == '__main__':
    main()
