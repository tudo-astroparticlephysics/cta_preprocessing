import astropy.units as u
import numpy as np

from ctapipe.io.containers import (
    TelescopePointingContainer, MCEventContainer,
    Container, ReconstructedShowerContainer,
    LeakageContainer, HillasParametersContainer,
    ConcentrationContainer, MCHeaderContainer, TimingParametersContainer)
from ctapipe.core import Field


class IslandContainer(Container):
    container_prefix = 'islands'
    num_islands = Field(np.nan, 'number of islands')
    # island_membership = Field(np.nan, 'membership of each pixel, compressed matrix')


class TelescopeParameterContainer(Container):

    container_prefix = ''

    telescope_id = Field(-1, 'telescope id')
    run_id = Field(-1, 'run id')
    array_event_id = Field(-1, 'array event id')

    leakage = Field(LeakageContainer(), 'Leakage container')
    hillas = Field(HillasParametersContainer(), 'HillasParametersContainer')
    concentration = Field(ConcentrationContainer(), 'ConcentrationContainer')
    timing = Field(TimingParametersContainer(), 'TimingParametersContainer')
    islands = Field(IslandContainer(), 'IslandContainer')
    num_pixel_in_shower = Field(np.nan, 'number of pixels after cleaning')

    pointing = Field(TelescopePointingContainer(), 'pointing information')

    distance_to_reconstructed_core_position = Field(
        np.nan,
        'Distance from telescope to reconstructed impact position',
        unit=u.m)

    camera_type_id = Field(
        np.nan,
        'An ID encoding the camera type (SCT, ASTRI, CHEC, ...)')
    telescope_type_id = Field(
        np.nan,
        'An ID encoding the telescope type (MST, SST, LST)')

    mirror_area = Field(np.nan, 'Mirror Area', unit=u.m**2)
    focal_length = Field(np.nan, 'focal length', unit=u.m)


class RunInfoContainer(Container):
    container_prefix = ''
    run_id = Field(-1, 'run id')
    mc = Field(MCHeaderContainer(), 'array wide MC information')


class ArrayEventContainer(Container):
    container_prefix = ''
    run_id = Field(-1, 'run id')
    array_event_id = Field(-1, 'array event id')
    reco = Field(ReconstructedShowerContainer(),
                 'reconstructed shower container')
    mc = Field(MCEventContainer(), 'array wide MC information')

    total_intensity = Field(np.nan, 'sum of all intensities')

    num_triggered_telescopes = Field(np.nan, 'Number of triggered Telescopes')

    num_triggered_lst = Field(np.nan, 'Number of triggered LSTs')
    num_triggered_mst = Field(np.nan, 'Number of triggered MSTs')
    num_triggered_sst = Field(np.nan, 'Number of triggered SSTs')
