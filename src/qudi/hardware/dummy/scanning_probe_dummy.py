# -*- coding: utf-8 -*-
"""
This file contains the Qudi dummy module for the confocal scanner.

Copyright (c) 2021, the qudi developers. See the AUTHORS.md file at the top-level directory of this
distribution and on <https://github.com/Ulm-IQO/qudi-iqo-modules/>

This file is part of qudi.

Qudi is free software: you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

Qudi is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with qudi.
If not, see <https://www.gnu.org/licenses/>.
"""

from logging import getLogger
import os
import re
import time
from typing import Optional, Dict, Tuple, Any, List
import numpy as np
from PySide2 import QtCore
from fysom import FysomError
from qudi.core.configoption import ConfigOption
from qudi.util.mutex import RecursiveMutex
from qudi.util.constraints import ScalarConstraint
from qudi.interface.scanning_probe_interface import (
    ScanningProbeInterface,
    ScanData,
    ScanConstraints,
    ScannerAxis,
    ScannerChannel,
    ScanSettings,
    BackScanCapability,
    CoordinateTransformMixin,
)
from dataclasses import dataclass

logger = getLogger(__name__)


@dataclass(frozen=True)
class DummyScanConstraints(ScanConstraints):
    spot_number: ScalarConstraint


class ImageGenerator:
    """Generate 1D and 2D images with random Gaussian spots."""

    def __init__(
        self,
        position_ranges: Dict[str, List[float]],
        spot_density: float,
        spot_size_dist: List[float],
        spot_amplitude_dist: List[float],
        spot_view_distance_factor: float,
        chunk_size: int,
        image_generation_max_calculations: int,
        indices_to_axis_mapper: dict,
    ) -> None:
        self.position_ranges = position_ranges
        self.spot_density = spot_density
        self.spot_size_dist = tuple(spot_size_dist)
        self.spot_amplitude_dist = tuple(spot_amplitude_dist)
        self.spot_view_distance_factor = spot_view_distance_factor
        self._chunk_size = chunk_size
        self._image_generation_max_calculations = image_generation_max_calculations
        self._indices_to_axes_mapper = indices_to_axis_mapper

        # random spots for each 2D axes pair
        self._spots: Dict[Tuple[str, str], Any] = {}
        self.randomize_new_spots()

    def randomize_new_spots(self):
        """Create a random set of Gaussian 2D peaks."""
        self._spots = dict()
        number_of_axes = len(list(self.position_ranges.keys()))
        axis_lengths = [abs(value[1] - value[0]) for value in self.position_ranges.values()]
        volume = 0
        if len(axis_lengths) > 0:
            volume = 1
            for value in axis_lengths:
                volume *= value

        spot_count = int(round(volume * self.spot_density**number_of_axes))
        # Have at least 1 spot
        if not spot_count:
            spot_count = 1

        spot_amplitudes = np.random.normal(self.spot_amplitude_dist[0], self.spot_amplitude_dist[1], spot_count)

        # scan bounds per axis.
        position_ranges = np.array(list(self.position_ranges.values()))
        ax_mins = position_ranges[:, 0]
        ax_maxs = position_ranges[:, 1]

        # vectorized generation of random spot positions and sigmas. Each row is a spot.
        spot_positions = np.random.uniform(ax_mins, ax_maxs, (spot_count, len(self.position_ranges)))
        spot_sigmas = np.random.normal(
            self.spot_size_dist[0], self.spot_size_dist[1], (spot_count, len(self.position_ranges))
        )

        self._spots["count"] = spot_count
        self._spots["pos"] = spot_positions
        self._spots["sigma"] = spot_sigmas
        self._spots["amp"] = spot_amplitudes
        logger.debug(f"Generated {spot_count} spots.")

    def generate_image(self, scan_vectors: Dict[str, np.ndarray], scan_resolution: Tuple[int, ...]) -> np.ndarray:
        sim_data = self._spots
        positions = sim_data["pos"]
        amplitudes = sim_data["amp"]
        sigmas = sim_data["sigma"]

        t_start = time.perf_counter()

        include_dist = max(self.spot_size_dist) * self.spot_view_distance_factor
        grid_array = self._scan_vectors_2_array(scan_vectors)

        positions_in_detection_volume, indices = self._points_in_detection_volume(positions, grid_array, include_dist)

        scan_image = np.random.uniform(0, min(self.spot_amplitude_dist) * 0.2, scan_resolution)

        if len(indices) > 0:
            gauss_image = self._resolve_grid_processed_sum_m_gaussian_n_dim_return_method(
                self._process_in_grid_chunks(
                    method=self._sum_m_gaussian_n_dim,
                    positions=positions_in_detection_volume,
                    grid_points=grid_array,
                    include_dist=include_dist,
                    method_params={
                        "grid_points": grid_array,
                        "mus": positions_in_detection_volume,
                        "sigmas": sigmas[indices],
                        "amplitudes": amplitudes[indices],
                    },
                ),
                image_dimension=scan_resolution,
            )

            scan_image += gauss_image

        logger.debug(
            f"Image took {time.perf_counter()-t_start:.3f} s for {positions_in_detection_volume.shape[0]} spots on"
            f" {len(grid_array)} grid points."
        )

        return scan_image

    def _scan_vectors_2_array(self, axes_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate coordinates for calculating a Gaussian distribution over specified axes.

        Parameters
        ----------
        axes_dict : dict of {str: ndarray}
            A dictionary where each key represents an axis name, and the corresponding value is a 1D array
            of values along that axis for which the Gaussian should be evaluated.

        Returns
        -------
        ndarray
            A 2D NumPy array of coordinates to evaluate the Gaussian at. Each row contains the coordinates
            for a single scan point, with columns arranged in alphabetical order based on the axis names.
        """

        sorted_axes = [
            axes_dict[self._indices_to_axes_mapper[key]] for key in sorted(self._indices_to_axes_mapper.keys())
        ]
        return np.asarray(sorted_axes).T

    def _process_in_grid_chunks(
        self, method, positions: np.ndarray, grid_points: np.ndarray, include_dist: float, method_params: dict
    ) -> list:
        if len(positions) * len(grid_points) <= self._image_generation_max_calculations:
            return [method(**method_params)]

        logger.warning(
            f"number of grid_points * number of spot positions exceeds {self._image_generation_max_calculations} values. "
            f"Processing in grid point chunks, this may take a while. "
            f"Consider reducing the number of scan points, the view distance of spots or the spot density to regain performance.\n "
            f"number of spots: {len(positions)}\n "
            f"number of grid points: {len(grid_points)}\n "
            f"view distance: {include_dist} m"
        )

        all_results = []

        # Process grid points in chunks
        for i in range(0, grid_points.shape[0], self._chunk_size):
            grid_chunk = grid_points[i : i + self._chunk_size]
            method_params.update({"grid_points": grid_chunk})
            all_results.append(method(**method_params))

        return all_results

    @staticmethod
    def _calc_plane_normal_vector(vectors):
        """
        Calculate the normal vector in n-dimensional space of a plane defined by a set of vectors.

        Parameters
        ----------
        vectors : ndarray
            A 2D array where each row represents one of the m vectors that define the plane,
            and each column corresponds to one of the n dimensions of the space.

        Returns
        -------
        ndarray
            The normal vector of the plane, represented as a 1D array of length n.
        """
        # TODO: move to qudi-core.math
        vectors = vectors - np.mean(vectors, axis=0)

        # Find the orthogonal complement of the space spanned by the vectors
        _, _, v = np.linalg.svd(vectors)
        normal = v[-1]

        normal /= np.linalg.norm(normal)

        return normal

    @staticmethod
    def _distance_to_plane(point, point_in_plane, normal_vector):
        # projection of the connection (point-point_in_plane) onto the normal
        distance = np.abs(np.dot(point - point_in_plane, normal_vector))

        return distance

    @staticmethod
    def _points_in_detection_volume(
        positions: np.ndarray, grid_points: np.ndarray, include_dist: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_scan_vecs = grid_points.T.shape[1]
        n_emitters = positions.shape[0]
        n_dim = grid_points.T.shape[0]

        # need dim(plane) vectors to define plane. Some reserve if unlucky.
        idxs_rand = []
        for i in range(2 * n_dim):
            idxs_rand.append(np.random.randint(0, n_scan_vecs))
        plane_vecs_rand = grid_points[idxs_rand, :]  # todo: check whether vectors really span 2d plane in n dim

        plane_normal_vec = ImageGenerator._calc_plane_normal_vector(plane_vecs_rand)
        distances_svd = np.asarray(
            [
                ImageGenerator._distance_to_plane(positions[i, :], plane_vecs_rand[0, :], plane_normal_vec)
                for i in range(0, n_emitters)
            ]
        )

        idxs = np.where(distances_svd <= include_dist)[0]
        # TODO: Remove in scan plane out of bounds spots
        positions_svd = positions[idxs, :]
        indices_svd = idxs

        return positions_svd, indices_svd

    @staticmethod
    def _sum_m_gaussian_n_dim(grid_points, mus, sigmas, amplitudes=None) -> np.ndarray:
        """
        Calculate the Gaussian values for each point in an n-dimensional grid with customizable amplitudes.

        Parameters
        ----------
        grid_points : ndarray
            A 1D NumPy array representing the coordinates at which the Gaussian should be evaluated.
        mus : ndarray
            A 1D NumPy array representing the mean values of the Gaussians for each dimension.
        sigmas : ndarray
            A 1D NumPy array representing the variances of each Gaussian along each axis.
        amplitudes : float, optional
            A scalar value representing the amplitude of the Gaussian, default is 1.0.

        Returns
        -------
        ndarray
            A NumPy array of Gaussian values evaluated at each point in `grid_points`.
        """
        if amplitudes is None:
            amplitudes = np.ones(mus.shape[0])

        grid_points = grid_points[:, np.newaxis, :]
        mus = mus[np.newaxis, :, :]
        sigmas = sigmas[np.newaxis, :, :]

        diff = grid_points - mus
        exponent = -0.5 * np.sum((diff / sigmas) ** 2, axis=2)

        gaussians = amplitudes[:, np.newaxis] * np.exp(exponent.T)

        return np.sum(gaussians, axis=0)

    @staticmethod
    def _resolve_grid_processed_sum_m_gaussian_n_dim_return_method(
        gauss_data: List[np.ndarray], image_dimension: Tuple[int, ...]
    ) -> np.ndarray:
        return np.vstack(gauss_data).reshape(image_dimension)

    @staticmethod
    def _create_coordinates(axes_dict: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Generate coordinates for calculating a Gaussian distribution from the specified axes.

        Parameters
        ----------
        axes_dict : dict of {int: ndarray}
            A dictionary where each key is an axis index, and each value is a 1D array of values along that axis
            for which the Gaussian should be evaluated.

        Returns
        -------
        ndarray
            A 2D NumPy array of coordinates to evaluate the Gaussian at. Each row contains the coordinates
            of a single scan point.
        """
        axes_coords = [axes_dict[coords] for coords in sorted(axes_dict.keys())]
        return np.column_stack(axes_coords)


class ScanningProbeDummyBare(ScanningProbeInterface):
    """
    Dummy scanning probe microscope. Produces a picture with several gaussian spots.

    Example config for copy-paste:

    scanning_probe_dummy:
        module.Class: 'dummy.scanning_probe_dummy.ScanningProbeDummy'
        options:
            position_ranges:
                x: [0, 200e-6]
                y: [0, 200e-6]
                z: [-100e-6, 100e-6]
            frequency_ranges:
                x: [1, 5000]
                y: [1, 5000]
                z: [1, 1000]
            resolution_ranges:
                x: [1, 10000]
                y: [1, 10000]
                z: [2, 1000]
            position_accuracy:
                x: 10e-9
                y: 10e-9
                z: 50e-9
            # max_spot_number: 80e3 # optional
            # spot_density: 5e4 # optional
            # spot_view_distance_factor: 2 # optional
            # spot_size_dist: [400e-9, 100e-9] # optional
            # spot_amplitude_dist: [2e5, 4e4] # optional
            # require_square_pixels: False # optional
            # back_scan_available: True # optional
            # back_scan_frequency_configurable: True # optional
            # back_scan_resolution_configurable: True # optional
            # image_generation_max_calculations: 100e6 # optional
            # image_generation_chunk_size: 1000 # optional
    """

    _threaded = True

    # config options
    _position_ranges: Dict[str, List[float]] = ConfigOption(name='position_ranges', missing='error')
    _frequency_ranges: Dict[str, List[float]] = ConfigOption(name='frequency_ranges', missing='error')
    _resolution_ranges: Dict[str, List[float]] = ConfigOption(name='resolution_ranges', missing='error')
    _position_accuracy: Dict[str, float] = ConfigOption(name='position_accuracy', missing='error')
    _max_spot_number: int = ConfigOption(name="max_spot_number", default=int(80e3), constructor=lambda x: int(x))
    _spot_density: float = ConfigOption(name="spot_density", default=1e5)  # in 1/m
    _spot_view_distance_factor: float = ConfigOption(
        name="spot_view_distance_factor", default=2, constructor=lambda x: float(x)
    )  # spots are visible by this factor times the maximum spot size from each scan point away
    _spot_size_dist: List[float] = ConfigOption(name="spot_size_dist", default=(400e-9, 100e-9))
    _spot_amplitude_dist: List[float] = ConfigOption(name="spot_amplitude_dist", default=(2e5, 4e4))
    _require_square_pixels: bool = ConfigOption(name='require_square_pixels', default=False)
    _back_scan_available: bool = ConfigOption(name='back_scan_available', default=True)
    _back_scan_frequency_configurable: bool = ConfigOption(name='back_scan_frequency_configurable', default=True)
    _back_scan_resolution_configurable: bool = ConfigOption(name='back_scan_resolution_configurable', default=True)
    _image_generation_max_calculations: int = ConfigOption(
        name="image_generation_max_calculations", default=int(100e6), constructor=lambda x: int(x)
    )  # number of points that can be calculated at once during image generation
    _image_generation_chunk_size: int = ConfigOption(
        name="image_generation_chunk_size", default=1000, constructor=lambda x: int(x)
    )  # if too many points are being calculated at once during image generation, this gives the size of the chunks it should be broken up into
    _use_recorded_scan: bool = ConfigOption(name='use_recorded_scan', default=False)
    _recorded_scan_file: str = ConfigOption(name='recorded_scan_file', default='')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scan process parameters
        self._scan_settings: Optional[ScanSettings] = None
        self._back_scan_settings: Optional[ScanSettings] = None
        self._current_position = dict()
        self._scan_image = None
        self._back_scan_image = None
        self._scan_data = None
        self._back_scan_data = None

        self._image_generator: Optional[ImageGenerator] = None
        self._recorded_scan: Optional[dict] = None  
        # "Hardware" constraints
        self._constraints: Optional[DummyScanConstraints] = None
        # Mutex for access serialization
        self._thread_lock = RecursiveMutex()

        self.__scan_start = 0
        self.__last_forward_pixel = 0
        self.__last_backward_pixel = 0
        self.__update_timer = None

    def on_activate(self):
        """Initialisation performed during activation of the module."""
        # Generate static constraints
        axes = list()
        indices_to_axis_mapper = {}
        for ii, (axis, ax_range) in enumerate(self._position_ranges.items()):
            indices_to_axis_mapper[ii] = axis
            dist = max(ax_range) - min(ax_range)
            resolution_range = tuple(self._resolution_ranges[axis])
            res_default = min(resolution_range[1], 100)
            frequency_range = tuple(self._frequency_ranges[axis])
            f_default = min(frequency_range[1], 1e3)

            position = ScalarConstraint(default=min(ax_range), bounds=tuple(ax_range))
            resolution = ScalarConstraint(default=res_default, bounds=resolution_range, enforce_int=True)
            frequency = ScalarConstraint(default=f_default, bounds=frequency_range)
            step = ScalarConstraint(default=0, bounds=(0, dist))
            spot_number = ScalarConstraint(default=self._max_spot_number, bounds=(0, self._max_spot_number))
            axes.append(
                ScannerAxis(
                    name=axis, unit='m', position=position, step=step, resolution=resolution, frequency=frequency
                )
            )
        channels = [
            ScannerChannel(name='fluorescence', unit='c/s', dtype='float64'),
            ScannerChannel(name='APD events', unit='count', dtype='float64'),
        ]

        if not self._back_scan_available:
            back_scan_capability = BackScanCapability(0)
        else:
            back_scan_capability = BackScanCapability.AVAILABLE
            if self._back_scan_resolution_configurable:
                back_scan_capability = back_scan_capability | BackScanCapability.RESOLUTION_CONFIGURABLE
            if self._back_scan_frequency_configurable:
                back_scan_capability = back_scan_capability | BackScanCapability.FREQUENCY_CONFIGURABLE
        self._constraints = DummyScanConstraints(
            axis_objects=tuple(axes),
            channel_objects=tuple(channels),
            back_scan_capability=back_scan_capability,
            has_position_feedback=False,
            square_px_only=False,
            spot_number=spot_number,
        )
        self._spot_density = self._spot_density_constructor(self._spot_density)

        # Set default process values
        self._current_position = {ax.name: np.mean(ax.position.bounds) for ax in self.constraints.axes.values()}
        self._scan_image = None
        self._back_scan_image = None
        self._scan_data = None
        self._back_scan_data = None

        # Create fixed maps of spots for each scan axes configuration
        self._image_generator = ImageGenerator(
            self._position_ranges,
            self._spot_density,
            self._spot_size_dist,
            self._spot_amplitude_dist,
            self._spot_view_distance_factor,
            self._image_generation_chunk_size,
            self._image_generation_max_calculations,
            indices_to_axis_mapper,
        )
        if self._use_recorded_scan:
            try:
                self._recorded_scan = self._load_recorded_scan(self._recorded_scan_file)
                if self._recorded_scan is None:
                    self.log.warning(
                        f"Recorded scan could not be loaded from '{self._recorded_scan_file}'. Falling back to generator."
                    )
            except Exception as exc:
                self._recorded_scan = None
                self.log.warning(
                    f"Failed to load recorded scan from '{self._recorded_scan_file}'. Using generator instead.",
                    exc_info=exc,
                )

        self.__scan_start = 0
        self.__last_forward_pixel = 0
        self.__last_backward_pixel = 0
        self.__update_timer = QtCore.QTimer()
        self.__update_timer.setSingleShot(True)
        self.__update_timer.timeout.connect(self._handle_timer, QtCore.Qt.QueuedConnection)

    def on_deactivate(self):
        """Deactivate properly the confocal scanner dummy."""
        self.reset()
        # free memory
        del self._image_generator
        self._scan_image = None
        self._back_scan_image = None
        try:
            self.__update_timer.stop()
        except:
            pass
        self.__update_timer.timeout.disconnect()

    @property
    def scan_settings(self) -> Optional[ScanSettings]:
        """Property returning all parameters needed for a 1D or 2D scan. Returns None if not configured."""
        with self._thread_lock:
            return self._scan_settings

    @property
    def back_scan_settings(self) -> Optional[ScanSettings]:
        with self._thread_lock:
            return self._back_scan_settings

    def reset(self):
        """Hard reset of the hardware."""
        with self._thread_lock:
            if self.module_state() == 'locked':
                self.module_state.unlock()
            self.log.debug('Scanning probe dummy has been reset.')

    @property
    def constraints(self) -> ScanConstraints:
        """Read-only property returning the constraints of this scanning probe hardware."""
        # self.log.debug('Scanning probe dummy "get_constraints" called.')
        return self._constraints

    def configure_scan(self, settings: ScanSettings) -> None:
        """
        Configure the hardware with all parameters required for a 1D or 2D scan.

        Raises an exception if the provided settings are invalid or do not comply with hardware constraints.

        Parameters
        ----------
        settings : ScanSettings
            An instance of `ScanSettings` containing all necessary scan parameters.

        Raises
        ------
        ValueError
            If the settings are invalid or incompatible with hardware constraints.
        """
        with self._thread_lock:
            self.log.debug('Scanning probe dummy "configure_scan" called.')
            # Sanity checking
            if self.module_state() != 'idle':
                raise RuntimeError(
                    'Unable to configure scan parameters while scan is running. ' 'Stop scanning and try again.'
                )

            # check settings - will raise appropriate exceptions if something is not right
            self.constraints.check_settings(settings)

            self._scan_settings = settings
            # reset back scan configuration
            self._back_scan_settings = None

    def configure_back_scan(self, settings: ScanSettings) -> None:
        """Configure the hardware with all parameters of the backwards scan.
        Raise an exception if the settings are invalid and do not comply with the hardware constraints.

        @param ScanSettings settings: ScanSettings instance holding all parameters for the back scan
        """
        with self._thread_lock:
            if self.module_state() != 'idle':
                raise RuntimeError(
                    'Unable to configure scan parameters while scan is running. ' 'Stop scanning and try again.'
                )
            if self._scan_settings is None:
                raise RuntimeError('Configure forward scan settings first.')

            # check settings - will raise appropriate exceptions if something is not right
            self.constraints.check_back_scan_settings(backward_settings=settings, forward_settings=self._scan_settings)
            self._back_scan_settings = settings

    def move_absolute(self, position, velocity=None, blocking=False):
        """Move the scanning probe to an absolute position as fast as possible or with a defined
        velocity.

        Log error and return current target position if something fails or a 1D/2D scan is in
        progress.
        """
        with self._thread_lock:
            # self.log.debug('Scanning probe dummy "move_absolute" called.')
            if self.module_state() != 'idle':
                raise RuntimeError('Scanning in progress. Unable to move to position.')
            elif not set(position).issubset(self._position_ranges):
                raise ValueError(
                    f'Invalid axes encountered in position dict. ' f'Valid axes are: {set(self._position_ranges)}'
                )
            else:
                move_distance = {ax: np.abs(pos - self._current_position[ax]) for ax, pos in position.items()}
                if velocity is None:
                    move_time = 0.01
                else:
                    move_time = max(0.01, np.sqrt(np.sum(dist**2 for dist in move_distance.values())) / velocity)
                time.sleep(move_time)
                self._current_position.update(position)
            return self._current_position

    def move_relative(self, distance, velocity=None, blocking=False):
        """Move the scanning probe by a relative distance from the current target position as fast
        as possible or with a defined velocity.

        Log error and return current target position if something fails or a 1D/2D scan is in
        progress.
        """
        with self._thread_lock:
            self.log.debug('Scanning probe dummy "move_relative" called.')
            if self.module_state() != 'idle':
                raise RuntimeError('Scanning in progress. Unable to move relative.')
            elif not set(distance).issubset(self._position_ranges):
                raise ValueError(
                    'Invalid axes encountered in distance dict. ' f'Valid axes are: {set(self._position_ranges)}'
                )
            else:
                new_pos = {ax: self._current_position[ax] + dist for ax, dist in distance.items()}
                if velocity is None:
                    move_time = 0.01
                else:
                    move_time = max(0.01, np.sqrt(np.sum(dist**2 for dist in distance.values())) / velocity)
                time.sleep(move_time)
                self._current_position.update(new_pos)
            return self._current_position

    def get_target(self):
        """
        Retrieve the current target position of the scanner hardware.

        Returns
        -------
        dict
            A dictionary representing the current target position for each axis.
        """
        with self._thread_lock:
            return self._current_position.copy()

    def get_position(self):
        """
        Retrieve a snapshot of the actual scanner position from position feedback sensors.

        Returns
        -------
        dict
            A dictionary representing the current actual position for each axis.
        """
        with self._thread_lock:
            self.log.debug('Scanning probe dummy "get_position" called.')
            position = {
                ax: pos + np.random.normal(0, self._position_accuracy[ax]) for ax, pos in self._current_position.items()
            }
            return position

    def start_scan(self):
        """Start a scan as configured beforehand.
        Log an error if something fails or a 1D/2D scan is in progress.
        """
        with self._thread_lock:
            self.log.debug('Scanning probe dummy "start_scan" called.')
            if self.module_state() != 'idle':
                raise RuntimeError('Cannot start scan. Scan already in progress.')
            if not self.scan_settings:
                raise RuntimeError('No scan settings configured. Cannot start scan.')
            self.module_state.lock()

            scan_vectors = self._init_scan_vectors()

            if self._use_recorded_scan and self._recorded_scan is not None:
                if self.scan_settings.scan_dimension == 2:
                    self._scan_image = self._resample_recorded_to_settings(self._recorded_scan, self.scan_settings)
                elif self.scan_settings.scan_dimension == 1:
                    self._scan_image = self._resample_recorded_to_settings_1d(self._recorded_scan, self.scan_settings)
                else:
                    self._scan_image = self._image_generator.generate_image(scan_vectors, self.scan_settings.resolution)
            else:
                self._scan_image = self._image_generator.generate_image(scan_vectors, self.scan_settings.resolution)
            self._scan_data = ScanData.from_constraints(
                settings=self.scan_settings, constraints=self.constraints, scanner_target_at_start=self.get_target()
            )
            self._scan_data.new_scan()

            if self._back_scan_settings is not None:
                if self._use_recorded_scan and self._recorded_scan is not None:
                    if self.scan_settings.scan_dimension == 2:
                        self._back_scan_image = self._resample_recorded_to_settings(
                            self._recorded_scan, self._back_scan_settings
                        )
                    elif self.scan_settings.scan_dimension == 1:
                        self._back_scan_image = self._resample_recorded_to_settings_1d(
                            self._recorded_scan, self._back_scan_settings
                        )
                    else:
                        self._back_scan_image = self._image_generator.generate_image(
                            scan_vectors, self.scan_settings.resolution
                        )
                else:
                    self._back_scan_image = self._image_generator.generate_image(
                        scan_vectors, self.scan_settings.resolution
                    )
                self._back_scan_data = ScanData.from_constraints(
                    settings=self.back_scan_settings,
                    constraints=self.constraints,
                    scanner_target_at_start=self.get_target(),
                )
                self._back_scan_data.new_scan()

            self.__scan_start = time.time()
            self.__last_forward_pixel = 0
            self.__last_backward_pixel = 0
            line_time = self.scan_settings.resolution[0] / self.scan_settings.frequency
            timer_interval_ms = int(0.5 * line_time * 1000)  # update twice every line
            self.__update_timer.setInterval(timer_interval_ms)

        self.__start_timer()

    def stop_scan(self):
        """Stop the currently running scan.
        Log an error if something fails or no 1D/2D scan is in progress.
        """

        self.log.debug('Scanning probe dummy "stop_scan" called.')
        if self.module_state() == 'locked':
            with self._thread_lock:
                self._scan_image = None
                self._back_scan_image = None

            self.__stop_timer()
            self.module_state.unlock()
        else:
            raise RuntimeError('No scan in progress. Cannot stop scan.')

    def emergency_stop(self):
        """ """
        try:
            self.module_state.unlock()
        except FysomError:
            pass
        self._scan_image = None
        self._back_scan_image = None
        self.log.warning('Scanner has been emergency stopped.')

    def _handle_timer(self):
        """Update during a running scan."""
        try:
            with self._thread_lock:
                self.__update_scan_data()
        except Exception as e:
            self.log.error("Could not update scan data.", exc_info=e)

    def __update_scan_data(self) -> None:
        """Update scan data."""
        if self.module_state() == 'idle':
            raise RuntimeError("Scan is not running.")

        t_elapsed = time.time() - self.__scan_start
        t_forward = self.scan_settings.resolution[0] / self.scan_settings.frequency
        if self.back_scan_settings is not None:
            back_resolution = self.back_scan_settings.resolution[0]
            t_backward = back_resolution / self.back_scan_settings.frequency
        else:
            back_resolution = 0
            t_backward = 0
        t_complete_line = t_forward + t_backward

        aq_lines = int(t_elapsed / t_complete_line)
        t_current_line = t_elapsed % t_complete_line
        if t_current_line < t_forward:
            # currently in forwards scan
            aq_px_backward = back_resolution * aq_lines
            aq_lines_forward = aq_lines + (t_current_line / t_forward)
            aq_px_forward = int(self.scan_settings.resolution[0] * aq_lines_forward)
        else:
            # currently in backwards scan
            aq_px_forward = self.scan_settings.resolution[0] * (aq_lines + 1)
            aq_lines_backward = aq_lines + (t_current_line - t_forward) / t_backward
            aq_px_backward = int(back_resolution * aq_lines_backward)

        # transposing the arrays is necessary to fill along the fast axis first
        new_forward_data = self._scan_image.T.flat[self.__last_forward_pixel : aq_px_forward]
        for ch in self.constraints.channels:
            self._scan_data.data[ch].T.flat[self.__last_forward_pixel : aq_px_forward] = new_forward_data
        self.__last_forward_pixel = aq_px_forward

        # back scan image is not fully accurate: last line is filled the same direction as the forward axis
        if self._back_scan_settings is not None:
            new_backward_data = self._back_scan_image.T.flat[self.__last_backward_pixel : aq_px_backward]
            for ch in self.constraints.channels:
                self._back_scan_data.data[ch].T.flat[self.__last_backward_pixel : aq_px_backward] = new_backward_data
            self.__last_backward_pixel = aq_px_backward

        if self.scan_settings.scan_dimension == 1:
            is_finished = aq_lines >= 1
        else:
            is_finished = aq_lines >= self.scan_settings.resolution[1]
        if is_finished:
            self.module_state.unlock()
            self.log.debug("Scan finished.")
        else:
            self.__start_timer()

    def get_scan_data(self) -> Optional[ScanData]:
        """Retrieve the ScanData instance used in the scan."""
        with self._thread_lock:
            if self._scan_data is None:
                return None
            else:
                return self._scan_data.copy()

    def get_back_scan_data(self) -> Optional[ScanData]:
        """Retrieve the ScanData instance used in the backwards scan."""
        with self._thread_lock:
            if self._back_scan_data is None:
                return None
            return self._back_scan_data.copy()

    def __start_timer(self):
        """
        Offload __update_timer.start() from the caller to the module's thread.
        ATTENTION: Do not call this from within thread lock protected code to avoid deadlock (PR #178).
        :return:
        """
        if self.thread() is not QtCore.QThread.currentThread():
            QtCore.QMetaObject.invokeMethod(self.__update_timer, 'start', QtCore.Qt.BlockingQueuedConnection)
        else:
            self.__update_timer.start()

    def __stop_timer(self):
        """
        Offload __update_timer.stop() from the caller to the module's thread.
        ATTENTION: Do not call this from within thread lock protected code to avoid deadlock (PR #178).
        :return:
        """
        if self.thread() is not QtCore.QThread.currentThread():
            QtCore.QMetaObject.invokeMethod(self.__update_timer, 'stop', QtCore.Qt.BlockingQueuedConnection)
        else:
            self.__update_timer.stop()

    def _init_scan_vectors_from_scan_settings(self) -> Dict[str, np.ndarray]:
        axes_scan_values = [
            np.linspace(
                self.scan_settings.range[ii][0], self.scan_settings.range[ii][1], self.scan_settings.resolution[ii]
            )
            for ii, _ in enumerate(self.scan_settings.axes)
        ]
        # generate all combinations of points
        meshgrids = np.meshgrid(*axes_scan_values, indexing="ij")
        # create position vector dictionary by raveling the grids
        scan_vectors = {axis: grid.ravel() for axis, grid in zip(self.scan_settings.axes, meshgrids)}
        return scan_vectors

    def _init_scan_vectors(self) -> Dict[str, np.ndarray]:
        scan_vectors = self._init_scan_vectors_from_scan_settings()
        scan_vectors = self._expand_coordinate(scan_vectors)  # always expand to all scan dims

        return scan_vectors

    def _load_recorded_scan(self, file_path: str) -> Optional[dict]:
        """Load a previously recorded XY scan from a file.

        Expected format:
        - Header lines start with '# ' and end with a line containing '---- END HEADER ----'.
        - Header contains keys like 'x axis resolution', 'y axis resolution', 'x axis min', 'x axis max', etc.
        - Data block after header is numeric, representing an image with shape (y_res, x_res),
          rows correspond to Y, columns to X. If flattened, it will be reshaped accordingly.
        """
        if not file_path:
            return None
        if not os.path.isfile(file_path):
            self.log.warning(f"Recorded scan file does not exist: {file_path}")
            return None

        header_lines = []
        data_start = 0
        with open(file_path, 'r') as fh:
            for idx, line in enumerate(fh):
                if line.startswith('#'):
                    header_lines.append(line.rstrip('\n'))
                    if 'END HEADER' in line:
                        data_start = idx + 1
                        break
                else:
                    # no header marker, assume no header
                    data_start = idx
                    break

        # Parse useful header fields
        header_text = '\n'.join(header_lines)
        def _search_float(pattern: str, default: Optional[float] = None) -> Optional[float]:
            m = re.search(pattern, header_text)
            return float(m.group(1)) if m else default
        def _search_int(pattern: str, default: Optional[int] = None) -> Optional[int]:
            m = re.search(pattern, header_text)
            return int(m.group(1)) if m else default

        x_res = _search_int(r"x axis resolution\s*=\s*([0-9]+)")
        y_res = _search_int(r"y axis resolution\s*=\s*([0-9]+)")
        x_min = _search_float(r"x axis min\s*=\s*([0-9eE\.\-\+]+)")
        x_max = _search_float(r"x axis max\s*=\s*([0-9eE\.\-\+]+)")
        y_min = _search_float(r"y axis min\s*=\s*([0-9eE\.\-\+]+)")
        y_max = _search_float(r"y axis max\s*=\s*([0-9eE\.\-\+]+)")

        delimiter = None
        mdel = re.search(r"delimiter\s*=\s*'([^']+)'", header_text)
        if mdel:
            # interpret escaped sequences like '\t'
            delim_str = mdel.group(1)
            delimiter = delim_str.encode('utf-8').decode('unicode_escape')

        # Load numeric data
        try:
            data = np.loadtxt(file_path, delimiter=delimiter, comments='#', skiprows=data_start)
        except Exception:
            # Fallback: generic whitespace
            data = np.loadtxt(file_path, comments='#', skiprows=data_start)

        # Ensure 2D image shape (rows = Y, cols = X)
        if x_res is not None and y_res is not None:
            total = int(x_res) * int(y_res)
            if data.ndim == 1 and data.size == total:
                data = data.reshape((y_res, x_res))
            elif data.ndim == 2 and data.shape != (y_res, x_res):
                # If transposed, try to fix
                if data.shape == (x_res, y_res):
                    data = data.T
        else:
            # If header missing, try to infer: prefer wide images (more columns)
            if data.ndim == 1:
                # cannot infer robustly; return as 1D
                pass

        result = {
            'image': data.astype(float),
            'x_res': x_res,
            'y_res': y_res,
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
        }
        return result

    def _resample_recorded_to_settings(self, recorded: dict, settings: ScanSettings) -> np.ndarray:
        """Resample the recorded image to the requested scan settings using separable linear interpolation.

        Assumes axes are ['x', 'y'] or contain 'x' and 'y'. If metadata missing, falls back to nearest indexing.
        """
        img = recorded.get('image')
        if img is None or img.ndim != 2:
            # fall back to generator path by forcing caller to handle
            raise ValueError('Recorded image invalid')

        # Determine requested axes order (fast axis first in settings.resolution[0])
        try:
            x_index = settings.axes.index('x')
            y_index = settings.axes.index('y')
        except ValueError:
            # Axes not matching, just return original image (or simple resize by nearest)
            return img

        req_x_points = settings.resolution[x_index]
        req_y_points = settings.resolution[y_index]

        # Build source axes
        src_x_res = recorded.get('x_res') or img.shape[1]
        src_y_res = recorded.get('y_res') or img.shape[0]
        src_x_min = recorded.get('x_min', 0.0)
        src_x_max = recorded.get('x_max', float(src_x_res - 1))
        src_y_min = recorded.get('y_min', 0.0)
        src_y_max = recorded.get('y_max', float(src_y_res - 1))

        src_x = np.linspace(src_x_min, src_x_max, src_x_res)
        src_y = np.linspace(src_y_min, src_y_max, src_y_res)

        # Requested axes from settings
        req_x_min, req_x_max = settings.range[x_index]
        req_y_min, req_y_max = settings.range[y_index]
        dst_x = np.linspace(req_x_min, req_x_max, req_x_points)
        dst_y = np.linspace(req_y_min, req_y_max, req_y_points)

        # First interpolate along X for each Y row
        # Vectorized 1D interpolation per row using list comprehension
        interp_x = np.vstack([np.interp(dst_x, src_x, img_row) for img_row in img])

        # Then interpolate along Y for each X column
        interp_y_cols = []
        for col_idx in range(interp_x.shape[1]):
            col = interp_x[:, col_idx]
            interp_col = np.interp(dst_y, src_y, col)
            interp_y_cols.append(interp_col)
        resampled = np.column_stack(interp_y_cols)

        # Ensure shape (req_y, req_x)
        resampled = resampled.reshape((req_y_points, req_x_points))
        return resampled

    def _spot_density_constructor(self, spot_density: float) -> float:
        volume_edges = [abs(pos_range[1] - pos_range[0]) for pos_range in self._position_ranges.values()]
        volume = 1
        for edge in volume_edges:
            volume *= edge
        spot_number = volume * spot_density ** len(self._position_ranges.keys())
        if not self._constraints.spot_number.is_valid(spot_number):
            spot_density = (self._constraints.spot_number.default / volume) ** (1 / len(self._position_ranges.keys()))
            self.log.warning(
                f"Specified spot density results in an out of bounds number of spots "
                f"({int(spot_number)}, allowed: {self._constraints.spot_number.bounds}). "
                f"To keep performance, reducing spot density to {spot_density} 1/m"
            )
        return spot_density

    def _resample_recorded_to_settings_1d(self, recorded: dict, settings: ScanSettings) -> np.ndarray:
        """Generate a 1D line profile from the recorded 2D image at the current target position.

        If the varying axis is 'x', we sample along X at fixed Y = current target 'y'.
        If the varying axis is 'y', we sample along Y at fixed X = current target 'x'.
        """
        img = recorded.get('image')
        if img is None or img.ndim != 2:
            raise ValueError('Recorded image invalid for 1D resampling')

        axes = settings.axes
        if len(axes) != 1:
            raise ValueError('Expected 1D settings with a single axis')

        var_ax = axes[0]
        req_points = settings.resolution[0]
        req_min, req_max = settings.range[0]
        dst = np.linspace(req_min, req_max, req_points)

        # Source axes
        src_x_res = recorded.get('x_res') or img.shape[1]
        src_y_res = recorded.get('y_res') or img.shape[0]
        src_x_min = recorded.get('x_min', 0.0)
        src_x_max = recorded.get('x_max', float(src_x_res - 1))
        src_y_min = recorded.get('y_min', 0.0)
        src_y_max = recorded.get('y_max', float(src_y_res - 1))
        src_x = np.linspace(src_x_min, src_x_max, src_x_res)
        src_y = np.linspace(src_y_min, src_y_max, src_y_res)

        target = self.get_target()

        if var_ax == 'x':
            # First interpolate along X for all rows at dst positions
            interp_x = np.vstack([np.interp(dst, src_x, row) for row in img])  # shape (src_y_res, req_points)
            # Now interpolate along Y at fixed target y for each column
            ty = target.get('y', float(np.clip(np.mean(src_y), src_y_min, src_y_max)))
            out = np.empty(req_points, dtype=float)
            for j in range(req_points):
                out[j] = np.interp(ty, src_y, interp_x[:, j])
            return out
        elif var_ax == 'y':
            # Interpolate along Y for all columns at dst positions
            interp_y_cols = []
            for col_idx in range(img.shape[1]):
                col = img[:, col_idx]
                interp_col = np.interp(dst, src_y, col)  # length req_points
                interp_y_cols.append(interp_col)
            interp_y = np.column_stack(interp_y_cols)  # shape (req_points, src_x_res)
            # Now interpolate along X at fixed target x for each row
            tx = target.get('x', float(np.clip(np.mean(src_x), src_x_min, src_x_max)))
            out = np.empty(req_points, dtype=float)
            for i in range(req_points):
                out[i] = np.interp(tx, src_x, interp_y[i, :])
            return out
        else:
            # Unknown axis; fallback: average across other dimension
            avg = img.mean(axis=0) if img.shape[1] >= img.shape[0] else img.mean(axis=1)
            src_axis = src_x if avg.shape[0] == src_x_res else src_y
            return np.interp(dst, src_axis, avg)


class ScanningProbeDummy(CoordinateTransformMixin, ScanningProbeDummyBare):
    def _init_scan_vectors(self) -> Dict[str, np.ndarray]:
        scan_vectors = super()._init_scan_vectors()

        if self.coordinate_transform_enabled:
            return self.coordinate_transform(scan_vectors)

        return scan_vectors
