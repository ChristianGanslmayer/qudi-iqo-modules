# -*- coding: utf-8 -*-
"""
This module is responsible for performing scanning probe measurements in order to find some optimal
position and move the scanner there.

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

from uuid import UUID

import numpy as np
from PySide2 import QtCore
import copy as cp
from typing import Dict, Tuple, List, Optional, Union
import itertools
from dataclasses import dataclass
from enum import Enum, auto
from lmfit.models import GaussianModel, ConstantModel

from qudi.core.module import LogicBase
from qudi.interface.scanning_probe_interface import ScanData, BackScanCapability
from qudi.logic.scanning_probe_logic import ScanningProbeLogic
from qudi.util.mutex import RecursiveMutex, Mutex
from qudi.core.connector import Connector
from qudi.core.statusvariable import StatusVar
from qudi.util.fit_models.gaussian import Gaussian2D, Gaussian
from qudi.core.configoption import ConfigOption


class FitStatus(Enum):
    OK = auto()
    TWO_PEAKS = auto()
    REJECTED_NOT_NV = auto()

@dataclass
class FitDecision:
    status: FitStatus
    reason: str
    recommended_pos: float | None      # 1D case
    diagnostics: dict                  # anything useful (snr, redchi, etc.)
    used_model: str                    # 'gaussian1' or 'gaussian2'
    peaks: list | None = None          # for two-peak fit: [{center, sigma, amp}, ...]


class ScanningOptimizeLogic(LogicBase):
    """
    This logic module makes use of the scanning probe logic to perform a sequence of
    1D and 2D spatial signal optimization steps.

    Example config for copy-paste:

    scanning_optimize_logic:
        module.Class: 'scanning_optimize_logic.ScanningOptimizeLogic'
        connect:
            scan_logic: scanning_probe_logic

    """

    # declare connectors
    _scan_logic = Connector(name='scan_logic', interface='ScanningProbeLogic')

    # status variables
    # not configuring the back scan parameters is represented by empty dictionaries

    # for all optimizer sub widgets, (2= xy, 1=z)
    _optimizer_sequence_dimensions: Tuple[int] = StatusVar(name='optimizer_sequence_dimensions', default=[2, 1])
    _scan_sequence: Tuple[Tuple[str, ...]] = StatusVar(name='scan_sequence', default=tuple())
    _data_channel = StatusVar(name='data_channel', default=None)
    _scan_range: Dict[str, float] = StatusVar(name='scan_range', default=dict())
    _scan_resolution: Dict[str, int] = StatusVar(name='scan_resolution', default=dict())
    _back_scan_resolution: Dict[str, int] = StatusVar(name='back_scan_resolution', default=dict())
    _scan_frequency: Dict[str, float] = StatusVar(name='scan_frequency', default=dict())
    _back_scan_frequency: Dict[str, float] = StatusVar(name='back_scan_frequency', default=dict())

    # signals
    sigOptimizeStateChanged = QtCore.Signal(bool, dict, object)
    sigOptimizeSettingsChanged = QtCore.Signal(dict)
    sigOptimizeSequenceDimensionsChanged = QtCore.Signal()

    _sigNextSequenceStep = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._thread_lock = RecursiveMutex()
        self._result_lock = Mutex()

        self._sequence_index = 0
        self._optimal_position = dict()
        self._last_scans = list()
        self._last_fits = list()
        self._avail_axes = tuple()
        self._stashed_settings = None


    def on_activate(self):
        """Initialisation performed during activation of the module."""
        scan_logic: ScanningProbeLogic = self._scan_logic()
        axes = scan_logic.scanner_axes
        channels = scan_logic.scanner_channels

        # check if settings in status variables are valid
        # reset to defaults if required
        try:
            self._check_scan_settings()
        except Exception as e:
            self.log.warning("Scan settings in Status Variable empty or invalid, using defaults.", exc_info=e)
            self._set_default_scan_settings()

        self._avail_axes = tuple(axes.values())
        self._set_default_scan_sequence()

        if self._data_channel is None:
            self._data_channel = tuple(channels.values())[0].name

        self._sequence_index = 0
        self._optimal_position = dict()
        self._last_scans = list()
        self._last_fits = list()

        self._sigNextSequenceStep.connect(self._next_sequence_step, QtCore.Qt.QueuedConnection)
        self._scan_logic().sigScanStateChanged.connect(self._scan_state_changed, QtCore.Qt.QueuedConnection)
        self.sigOptimizeSequenceDimensionsChanged.connect(self._set_default_scan_sequence, QtCore.Qt.QueuedConnection)

    def on_deactivate(self):
        """Reverse steps of activation"""
        self._scan_logic().sigScanStateChanged.disconnect(self._scan_state_changed)
        self._sigNextSequenceStep.disconnect()
        self.stop_optimize()
        return

    @property
    def data_channel(self) -> str:
        return self._data_channel

    @property
    def scan_range(self) -> Dict[str, float]:
        return self._scan_range.copy()

    @property
    def scan_resolution(self) -> Dict[str, int]:
        return self._scan_resolution.copy()

    @property
    def back_scan_resolution(self) -> Dict[str, int]:
        # use value of forward scan if not configured otherwise (merge dictionaries)
        return {**self._scan_resolution, **self._back_scan_resolution}

    @property
    def scan_frequency(self) -> Dict[str, float]:
        return self._scan_frequency.copy()

    @property
    def back_scan_frequency(self) -> Dict[str, float]:
        # use value of forward scan if not configured otherwise (merge dictionaries)
        return {**self._scan_frequency, **self._back_scan_frequency}

    @property
    def scan_sequence(self) -> Tuple[Tuple[str, ...]]:
        # serialization into status variable changes step type <tuple> -> <list>
        return tuple(tuple(i) for i in self._scan_sequence)

    @scan_sequence.setter
    def scan_sequence(self, sequence: Tuple[Tuple[str, ...]]):
        """
        @param sequence: list or tuple of string tuples giving the scan order, e.g. [('x','y'), ('z')]
        """
        occurring_axes = set([axis for step in sequence for axis in step])
        available_axes = [ax.name for ax in self._avail_axes]
        if not occurring_axes.issubset(available_axes):
            self.log.error(f"Optimizer sequence {sequence} must contain only" f" available axes ({available_axes}).")
        else:
            self._scan_sequence = sequence

    @property
    def allowed_scan_sequences(self) -> Dict[list, List[tuple]]:
        allowed_sequences = {}
        for dimension in self.allowed_optimizer_sequence_dimensions:
            try:
                allowed_sequences[dimension] = self._allowed_sequences(dimension)
            except NotImplementedError:
                continue

        return allowed_sequences

    def _allowed_sequences(self, sequence_dimension: List[int]) -> List[Tuple[tuple]]:
        axes_names = [ax.name for ax in self._avail_axes]
        # figure out sensible optimization sequences for user selection
        possible_optimizations_per_plot = [itertools.combinations(axes_names, n) for n in sequence_dimension]
        optimization_sequences = list(itertools.product(*possible_optimizations_per_plot))
        sequences_no_axis_twice = []
        if sum(sequence_dimension) > len(axes_names):
            raise NotImplementedError(
                f"Requested optimization sequence ({sum(sequence_dimension)}) "
                f"is greater than available scanner axes ({len(axes_names)}). "
                f"This is currently not supported. Decrease 'optimizer_sequence_dimensions' "
                f"in your config file."
            )

        for sequence in optimization_sequences:
            occurring_axes = [axis for step in sequence for axis in step]
            if len(occurring_axes) <= len(set(occurring_axes)):
                sequences_no_axis_twice.append(sequence)

        return sequences_no_axis_twice

    @property
    def optimizer_sequence_dimensions(self) -> list:
        return self._optimizer_sequence_dimensions

    @optimizer_sequence_dimensions.setter
    def optimizer_sequence_dimensions(self, dimensions: tuple) -> None:
        self._optimizer_sequence_dimensions = self.sequence_dimension_constructor(dimensions)
        self.sigOptimizeSequenceDimensionsChanged.emit()

    @property
    def allowed_optimizer_sequence_dimensions(self) -> List[tuple]:
        allowed_values = {1, 2}
        valid_combinations = []
        # TODO: Fix this constraint
        max_value = len(self._avail_axes)  # current toolchain constraint
        # Iterate over all possible lengths from 1 to the max number of axes
        for length in range(1, max_value // min(allowed_values) + 1):
            all_combinations = itertools.product(allowed_values, repeat=length)
            valid_combinations += [comb for comb in all_combinations if sum(comb) <= max_value]

        return valid_combinations

    @property
    def optimizer_running(self):
        return self.module_state() != 'idle'

    def set_optimize_settings(
        self,
        data_channel: str,
        scan_sequence: Tuple[Tuple[str, ...]],
        scan_dimension: List[int],
        range: Dict[str, float],
        resolution: Dict[str, int],
        frequency: Dict[str, float],
        back_resolution: Dict[str, int] = None,
        back_frequency: Dict[str, float] = None,
    ):
        """Set all optimizer settings."""
        if back_resolution is None:
            back_resolution = dict()
        if back_frequency is None:
            back_frequency = dict()
        with self._thread_lock:
            if self.module_state() != 'idle':
                self.log.error('Cannot change optimize settings when module is locked.')
            else:
                self._data_channel = data_channel
                self.optimizer_sequence_dimensions = scan_dimension
                self.scan_sequence = scan_sequence
                self._scan_range.update(range)
                self._scan_resolution.update(resolution)
                self._scan_frequency.update(frequency)
                self._back_scan_resolution.update(back_resolution)
                self._back_scan_frequency.update(back_frequency)

    @property
    def last_scans(self):
        with self._result_lock:
            return self._last_scans.copy()

    @property
    def last_fits(self):
        with self._result_lock:
            return self._last_fits.copy()

    @property
    def optimal_position(self):
        return self._optimal_position.copy()

    def toggle_optimize(self, start):
        if start:
            self.start_optimize()
        else:
            self.stop_optimize()

    def start_optimize(self):
        with self._thread_lock:
            scan_logic: ScanningProbeLogic = self._scan_logic()

            if self.module_state() != 'idle' or scan_logic.module_state() != 'idle':
                self.sigOptimizeStateChanged.emit(True, dict(), None)
                return

            self.module_state.lock()

            self._stashed_settings = scan_logic.get_scan_settings_per_ax()

            curr_pos = scan_logic.scanner_target
            constraints = scan_logic.scanner_constraints
            for ax, rel_rng in self.scan_range.items():
                rng_start = curr_pos[ax] - rel_rng / 2
                rng_stop = curr_pos[ax] + rel_rng / 2
                # range needs to be clipped if optimizing at the very edge
                rng_start = constraints.axes[ax].position.clip(rng_start)
                rng_stop = constraints.axes[ax].position.clip(rng_stop)
                scan_logic.set_scan_range(ax, (rng_start, rng_stop))

            for ax, res in self.scan_resolution.items():
                scan_logic.set_scan_resolution(ax, res)
            for ax, res in self.back_scan_resolution.items():
                scan_logic.set_back_scan_resolution(ax, res)
            for ax, res in self.scan_frequency.items():
                scan_logic.set_scan_frequency(ax, res)
            for ax, res in self.back_scan_frequency.items():
                scan_logic.set_back_scan_frequency(ax, res)

            # optimizer scans always explicitly configure the backwards scan settings
            scan_logic.set_use_back_scan_settings(True)

            with self._result_lock:
                self._last_scans = list()
                self._last_fits = list()
            self._scan_logic().save_to_history = False  # optimizer scans not saved
            self._sequence_index = 0
            self._optimal_position = dict()
            self.sigOptimizeStateChanged.emit(True, self.optimal_position, None)
            self._sigNextSequenceStep.emit()

    def _next_sequence_step(self):
        with self._thread_lock:
            if self.module_state() == 'idle':
                return
            self._scan_logic().toggle_scan(True, self._scan_sequence[self._sequence_index], self.module_uuid)

    def _scan_state_changed(self, is_running: bool,
                            data: Optional[ScanData], back_scan_data: Optional[ScanData],
                            caller_id: UUID):
        with self._thread_lock:
            if is_running or self.module_state() == 'idle' or caller_id != self.module_uuid:
                return
            elif not is_running and data is None:
                # scan could not be started due to some error
                self.stop_optimize()
            elif data is not None:
                # self.log.debug(f"Trying to fit on data after scan of dim {data.scan_dimension}")

                try:
                    if data.settings.scan_dimension == 1:
                        x = np.linspace(*data.settings.range[0], data.settings.resolution[0])

                        # axis tuple like ('z',) or ('x',); take the single axis name
                        axis = data.settings.axes[0]
                        opt_pos, fit_data, fit_res = self._get_pos_from_1d_fit_decide(axis, x,
                                                                                      data.data[self._data_channel])
                    else:
                        x = np.linspace(*data.settings.range[0], data.settings.resolution[0])
                        y = np.linspace(*data.settings.range[1], data.settings.resolution[1])
                        xy = np.meshgrid(x, y, indexing='ij')
                        opt_pos, fit_data, fit_res = self._get_pos_from_2d_gauss_fit(
                            xy, data.data[self._data_channel].ravel()
                        )

                    position_update = {ax: opt_pos[ii] for ii, ax in enumerate(data.settings.axes)}
                    # self.log.debug(f"Optimizer issuing position update: {position_update}")
                    if fit_data is not None:
                        new_pos = self._scan_logic().set_target_position(position_update, move_blocking=True)
                        for ax in tuple(position_update):
                            position_update[ax] = new_pos[ax]

                        fit_data = {'fit_data': fit_data, 'full_fit_res': fit_res}

                    self._optimal_position.update(position_update)
                    with self._result_lock:
                        self._last_scans.append(cp.copy(data))
                        self._last_fits.append(fit_res)
                    self.sigOptimizeStateChanged.emit(True, position_update, fit_data)

                    # Abort optimize if fit failed
                    if fit_data is None:
                        self.log.warning("Stopping optimization due to failed fit.")
                        self.stop_optimize()
                        return

                except:
                    self.log.exception("")

            self._sequence_index += 1

            # Terminate optimize sequence if finished; continue with next sequence step otherwise
            if self._sequence_index >= len(self._scan_sequence):
                self.stop_optimize()
            else:
                self._sigNextSequenceStep.emit()
            return

    def stop_optimize(self):
        with self._thread_lock:
            if self.module_state() == 'idle':
                self.sigOptimizeStateChanged.emit(False, dict(), None)
                return

            try:
                if self._scan_logic().module_state() != 'idle':
                    # optimizer scans are never saved in scanning history
                    self._scan_logic().stop_scan()
            finally:

                for setting, back_setting in self._stashed_settings:
                    # self.log.debug(f"Recovering scan settings: {setting}")
                    self._scan_logic().set_scan_settings(setting)
                    self._scan_logic().set_back_scan_settings(back_setting)

                self._stashed_settings = None

                self._scan_logic().save_to_history = True
                self.module_state.unlock()
                self.sigOptimizeStateChanged.emit(False, dict(), None)

    def _get_pos_from_2d_gauss_fit(self, xy, data):
        model = Gaussian2D()

        try:
            fit_result = model.fit(data, x=xy, **model.estimate_peak(data, xy))
        except:
            x_min, x_max = xy[0].min(), xy[0].max()
            y_min, y_max = xy[1].min(), xy[1].max()
            x_middle = (x_max - x_min) / 2 + x_min
            y_middle = (y_max - y_min) / 2 + y_min
            self.log.exception('2D Gaussian fit unsuccessful.')
            return (x_middle, y_middle), None, None

        return (
            (fit_result.best_values['center_x'], fit_result.best_values['center_y']),
            fit_result.best_fit.reshape(xy[0].shape),
            fit_result,
        )

    def _get_quality_cfg(self) -> dict:
        """
        Return quality configuration dict. Reads from module config.
        Falls back to built-in defaults if not provided.
        """
        # Built-in fallback defaults
        defaults = {
            'min_snr': 3.0,
            'max_sigma_um': 2.0,
            'edge_margin_frac': 0.08,
            'min_amplitude': 0.0,
            'try_two_gaussian': True,
            'two_peak_min_sep_um': 0.5,
            'two_peak_amp_ratio_max': 3.0,
            'fallback_no_move': True,
        }

        logic_cfg = getattr(self, 'config', None)
        if isinstance(logic_cfg, dict):
            cfg = dict(defaults)
            cfg.update(logic_cfg.get('optimizer_quality', {}))
        else:
            cfg = defaults

        self._quality_cfg = cfg
        return cfg

    def _get_pos_from_1d_gauss_fit(self, x, data):
        model = Gaussian()

        try:
            fit_result = model.fit(data, x=x, **model.estimate_peak(data, x))
        except:
            x_min, x_max = x.min(), x.max()
            middle = (x_max - x_min) / 2 + x_min
            self.log.exception('1D Gaussian fit unsuccessful.')
            return (middle,), None, None

        return (fit_result.best_values['center'],), fit_result.best_fit, fit_result

    def _get_pos_from_1d_fit_decide(self, axis: str, x, data):
        """
        Decide best position for a 1D scan along `axis`:
          1) try 1-Gaussian; if OK -> use it
          2) if bad OR 1-Gaussian fails -> try 2-Gaussian; if OK -> use peak nearer to current pos
          3) else -> no move (current position)
        Always returns ((pos,), best_fit_array, fit_result_like) with best_fit_array != None.
        Attaches `assessment` dict to the returned fit_result object.
        """
        x = np.asarray(x).ravel()
        y = np.asarray(data).ravel()

        # ----------------
        # 1) Single-Gaussian
        # ----------------
        pos1, bestfit1, res1 = self._get_pos_from_1d_gauss_fit(x, y)

        # If 1-Gaussian produced a result, assess it
        if res1 is not None:
            dec1 = self._assess_1d_fit(x, y, res1, model='gaussian1')
            if dec1.status.name == 'OK':
                assessment = {
                    'status': 'OK',
                    'reason': 'ok',
                    'used_model': 'gaussian1',
                    'recommended_pos': dec1.recommended_pos,
                    'diagnostics': dec1.diagnostics,
                    'peaks': None,
                }
                setattr(res1, 'assessment', assessment)

                if hasattr(self, 'log'):
                    self.log.info(
                        "Optimizer(1D %s): using single-Gaussian fit @ %.6g (redchi=%s)",
                        axis, dec1.recommended_pos,
                        getattr(res1, 'redchi', None)
                    )

                return (dec1.recommended_pos,), bestfit1, res1
        else:
            # Ensure bestfit1 is a valid array so callers don't skip movement logic
            bestfit1 = y.copy()

        # ----------------
        # 2) Two-Gaussian fallback (even if 1-Gaussian failed)
        # ----------------
        q = self._get_quality_cfg()
        if q.get('try_two_gaussian', True):
            try:
                pos2, bestfit2, res2 = self._get_pos_from_1d_two_gauss_fit(axis, x, y)
                # Assess 2-G result (basic checks reused)
                dec2 = self._assess_1d_fit(x, y, res2, model='gaussian2')
                if dec2.status.name == 'OK':
                    chosen = pos2[0]
                    # Try to collect both peaks
                    peaks = None
                    try:
                        p = res2.params
                        peaks = [
                            {'center': p['g1_center'].value, 'sigma': p['g1_sigma'].value,
                             'amp_area': p['g1_amplitude'].value},
                            {'center': p['g2_center'].value, 'sigma': p['g2_sigma'].value,
                             'amp_area': p['g2_amplitude'].value},
                        ]
                    except Exception:
                        pass
                    assessment = {
                        'status': 'OK',
                        'reason': 'two_gauss_ok',
                        'used_model': 'gaussian2',
                        'recommended_pos': chosen,
                        'diagnostics': dec2.diagnostics,
                        'peaks': peaks,
                    }
                    setattr(res2, 'assessment', assessment)

                    if hasattr(self, 'log'):
                        self.log.info(
                            "Optimizer(1D %s): using two-Gaussian fit @ %.6g (redchi=%s)",
                            axis, chosen, getattr(res2, 'redchi', None)
                        )

                    return (chosen,), bestfit2, res2
                else:
                    # keep res2/bestfit2 for plotting/diagnostics
                    bad_res = res2
                    bad_bestfit = bestfit2
                    bad_reason = dec2.reason
                    bad_diag = dec2.diagnostics
            except Exception:
                bad_res = None
                bad_bestfit = None
                bad_reason = 'fit_failed_two_gauss'
                bad_diag = {}
        else:
            bad_res = None
            bad_bestfit = None
            bad_reason = 'two_gauss_disabled'
            bad_diag = {}

        # ----------------
        # 3) Both bad → no move; return current position (or scan midpoint if unknown)
        # ----------------
        curr = self._get_current_axis_position(axis)
        if curr is None:
            # Fallback so we never crash; also log if useful
            x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
            curr = 0.5 * (x_min + x_max)
            if hasattr(self, 'log'):
                self.log.warning("Optimizer: current position for axis '%s' unavailable; "
                                 "falling back to scan midpoint.", axis)

        # prefer some non-None bestfit for the GUI
        bestfit = bad_bestfit if bad_bestfit is not None else bestfit1
        if bestfit is None:
            bestfit = y.copy()

        # attach assessment to whichever result object we have
        res = bad_res if bad_res is not None else (res1 if res1 is not None else type('Dummy', (), {})())
        assessment = {
            'status': 'REJECTED_NOT_NV',
            'reason': bad_reason if bad_res is not None else (
                dec1.reason if (res1 is not None) else 'fit_failed_single'),
            'used_model': 'gaussian2' if bad_res is not None else ('gaussian1' if res1 is not None else 'none'),
            'recommended_pos': None,
            'diagnostics': bad_diag if bad_res is not None else (dec1.diagnostics if (res1 is not None) else {}),
            'peaks': None,
        }
        setattr(res, 'assessment', assessment)

        if hasattr(self, 'log'):
            self.log.warning("Optimizer(1D %s): rejecting move (reason=%s) -> staying at %.6g",
                             axis, assessment['reason'], curr)

        return (curr,), bestfit, res

    def _get_current_axis_position(self, axis: str):
        """
        Try to read current scanner position along `axis` from several likely providers.
        Returns float position if available, otherwise None (no exception).
        """
        # Common attributes used across setups
        provider_names = (
            '_scanner', 'scanner',  # hardware/logic
            '_scan_logic', 'scan_logic',  # other logic
            '_scanner_logic', 'scanner_logic',
            '_positioner', '_stage', '_confocal',
        )
        for name in provider_names:
            obj = getattr(self, name, None)
            if obj is None or not hasattr(obj, 'get_position'):
                continue
            try:
                pos = obj.get_position()
            except Exception:
                continue
            if isinstance(pos, dict) and axis in pos:
                try:
                    return float(pos[axis])
                except Exception:
                    pass

        # Optional: fall back to a cached last known pos if you keep one
        for cache_name in ('_last_optimal_pos', '_last_position', 'last_position'):
            pos = getattr(self, cache_name, None)
            if isinstance(pos, dict) and axis in pos:
                try:
                    return float(pos[axis])
                except Exception:
                    pass

        # Nothing found
        return None


    def _get_pos_from_1d_two_gauss_fit(self, axis: str, x, data):
        """
        Fit a sum of two 1D Gaussians plus a constant background.
        Returns: ((best_center,), best_fit, fit_result)

        - Chooses initial centers from the two highest points (after background estimate).
        - Picks the component with the larger fitted amplitude as the 'best_center'.
        """
        # --- prepare data ---
        x = np.asarray(x).ravel()
        y = np.asarray(data).ravel()
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size < 3:
            raise ValueError("Not enough finite points for two-Gaussian fit")

        xmin, xmax = float(x.min()), float(x.max())
        span = xmax - xmin if xmax > xmin else 1.0

        # --- crude background + peak seeds ---
        # robust baseline: lower decile
        bkg0 = float(np.nanpercentile(y, 10))
        y0 = y - bkg0

        # seed centers: indices of two largest values after baseline subtraction
        # (works without scipy)
        if y0.size >= 2:
            top2_idx = np.argpartition(y0, -2)[-2:]
        else:
            top2_idx = np.array([np.argmax(y0), np.argmax(y0)])  # degenerate case

        # sort by x so g1 is left, g2 is right (stable parameterization)
        top2_idx = top2_idx[np.argsort(x[top2_idx])]
        i1, i2 = int(top2_idx[0]), int(top2_idx[1])

        c1_0, c2_0 = float(x[i1]), float(x[i2])
        a1_0, a2_0 = max(float(y0[i1]), 1.0), max(float(y0[i2]), 1.0)
        # heuristic sigma: 1/10 of span (clamped)
        sig0 = max(0.05 * span, 1e-12)

        # ensure distinct initial centers (tiny nudge if identical)
        if abs(c2_0 - c1_0) < 1e-15:
            c2_0 = min(xmax, c1_0 + 0.01 * span)

        # --- build model: two Gaussians + constant background ---
        g1 = GaussianModel(prefix='g1_')
        g2 = GaussianModel(prefix='g2_')
        bkg = ConstantModel(prefix='bkg_')
        model = g1 + g2 + bkg

        params = model.make_params()

        # background
        params['bkg_c'].set(value=bkg0)

        # g1 seeds and bounds
        params['g1_center'].set(value=c1_0, min=xmin, max=xmax)
        params['g1_sigma'].set(value=sig0, min=span * 1e-6, max=span)  # positive, sane upper
        params['g1_amplitude'].set(value=a1_0 * sig0 * np.sqrt(2 * np.pi), min=0.0)  # area ~ amp*sig*sqrt(2π)

        # g2 seeds and bounds
        params['g2_center'].set(value=c2_0, min=xmin, max=xmax)
        params['g2_sigma'].set(value=sig0, min=span * 1e-6, max=span)
        params['g2_amplitude'].set(value=a2_0 * sig0 * np.sqrt(2 * np.pi), min=0.0)

        # optional soft constraint: centers at least a tiny fraction apart (kept heuristic)
        # If you later want a hard constraint, you can reparameterize centers.
        # For now we leave it to the data + bounds.

        # --- fit ---
        try:
            fit_result = model.fit(y, params, x=x, nan_policy='omit', max_nfev=5000)
        except Exception as exc:
            # If the fit fails, return a degenerate result similar to the 1-Gauss API
            # so the caller can decide what to do.
            # Here we just raise; or you could fall back to single-Gaussian.
            raise

        curr = self._get_current_axis_position(axis)

        c1 = fit_result.params['g1_center'].value
        c2 = fit_result.params['g2_center'].value

        if (curr is not None) and (c1 is not None) and (c2 is not None):
            chosen_center = c1 if abs(c1 - curr) <= abs(c2 - curr) else c2
        else:
            # Fallback: choose the higher peak if we don't know current position
            def peak_height(prefix):
                area = fit_result.params[f'{prefix}amplitude'].value
                sigma = fit_result.params[f'{prefix}sigma'].value
                if sigma is None or sigma <= 0:
                    return -np.inf
                return area / (sigma * np.sqrt(2 * np.pi))

            chosen_center = c1 if peak_height('g1_') >= peak_height('g2_') else c2

        chosen_center = float(chosen_center) if chosen_center is not None else None
        return (chosen_center,), fit_result.best_fit, fit_result

    def _assess_1d_fit(self, x, y, fit_result, model='gaussian1') -> FitDecision:
        """Assess quality of a 1D Gaussian fit and decide if the result is usable."""
        params = fit_result.params

        # Get core parameters
        center = params['center'].value if 'center' in params else None
        sigma_val = params['sigma'].value if 'sigma' in params else None

        # Robust amplitude/offset lookup (handles 1D and 2D naming variations)
        amplitude_names = ['amplitude', 'amp', 'A', 'height']
        offset_names = ['offset', 'bkg', 'bkg_c', 'const', 'c']
        amp_val = self._get_param_value(params, amplitude_names)
        off_val = self._get_param_value(params, offset_names)

        # Load thresholds
        q = self._get_quality_cfg()
        x_min, x_max = float(x.min()), float(x.max())
        margin = (x_max - x_min) * q['edge_margin_frac']

        # Compute contrast (SNR proxy)
        snr = None
        if amp_val is not None and off_val is not None:
            denom = abs(off_val)
            if denom > 1e-12:
                snr = abs(amp_val) / denom

        # Fallback to crude metric if no valid contrast
        if snr is None or not np.isfinite(snr):
            snr = np.ptp(y) / (np.std(y) + 1e-9)

        # Decide
        reason = None
        if amp_val is not None and amp_val < q['min_amplitude']:
            reason = 'low_amplitude'
        elif sigma_val is not None and sigma_val * 1e6 > q['max_sigma_um']:
            reason = 'sigma_out_of_bounds'
        elif center is not None and (center < x_min + margin or center > x_max - margin):
            reason = 'edge_peak'
        elif hasattr(fit_result, 'redchi') and fit_result.redchi > 2.0:
            reason = 'high_redchi'
        elif snr < q['min_snr']:
            reason = 'low_snr'

        if reason is None:
            return FitDecision(FitStatus.OK, 'ok', center, {
                'snr': snr, 'redchi': getattr(fit_result, 'redchi', None)
            }, model)

        # If single-Gauss rejected and allowed, try two-Gaussian fallback
        if model == 'gaussian1' and q['try_two_gaussian']:
            return FitDecision(FitStatus.TWO_PEAKS, reason, None, {
                'snr': snr, 'redchi': getattr(fit_result, 'redchi', None)
            }, model)

        # Otherwise reject
        amp_str = f"{amp_val:.3g}" if amp_val is not None else "n/a"
        sigma_str = f"{sigma_val * 1e6:.3g}" if sigma_val is not None else "n/a"
        redchi_val = getattr(fit_result, 'redchi', None)
        redchi_str = f"{redchi_val:.3g}" if redchi_val is not None else "n/a"
        snr_str = f"{snr:.3g}" if snr is not None and np.isfinite(snr) else "n/a"

        self.log.warning(
            f"Fit rejected on model={model}: reason={reason}, "
            f"amp={amp_str}, sigma={sigma_str} µm, redchi={redchi_str}, contrast={snr_str}"
        )
        return FitDecision(FitStatus.REJECTED_NOT_NV, reason, None, {
            'snr': snr, 'redchi': getattr(fit_result, 'redchi', None)
        }, model)

    def _get_param_value(self, params, candidate_names):
        """Return first available lmfit Parameter.value from candidate_names, else None."""
        for name in candidate_names:
            if name in params:
                try:
                    return params[name].value
                except Exception:
                    pass
        return None

    def _check_scan_settings(self):
        """Basic check of scan settings for all axes."""
        scan_logic: ScanningProbeLogic = self._scan_logic()

        for stg in [self.scan_range, self.scan_resolution, self.scan_frequency]:
            axs = stg.keys()
            for ax in axs:
                if ax not in scan_logic.scanner_axes.keys():
                    self.log.debug(f"Axis {ax} from optimizer scan settings not available on scanner" )
                    raise ValueError

        capability = scan_logic.back_scan_capability
        if self._back_scan_resolution and (BackScanCapability.RESOLUTION_CONFIGURABLE not in capability):
            raise AssertionError('Back scan resolution cannot be configured for this scanner hardware.')
        if self._back_scan_frequency and (BackScanCapability.FREQUENCY_CONFIGURABLE not in capability):
            raise AssertionError('Back scan frequency cannot be configured for this scanner hardware.')
        for name, ax in scan_logic.scanner_axes.items():
            ax.position.check(self.scan_range[name])
            ax.resolution.check(self.scan_resolution[name])
            ax.resolution.check(self.back_scan_resolution[name])
            ax.frequency.check(self.scan_frequency[name])
            ax.frequency.check(self.back_scan_frequency[name])

    def _set_default_scan_settings(self):
        """Set range, resolution and frequency to default values."""
        scan_logic: ScanningProbeLogic = self._scan_logic()
        axes = scan_logic.scanner_axes
        self._scan_range = {ax.name: abs(ax.position.maximum - ax.position.minimum) / 100 for ax in axes.values()}
        self._scan_resolution = {ax.name: max(16, ax.resolution.minimum) for ax in axes.values()}
        self._scan_frequency = {ax.name: max(ax.frequency.minimum, ax.frequency.maximum / 100) for ax in axes.values()}
        self._back_scan_resolution = {}
        self._back_scan_frequency = {}

    def _set_default_scan_sequence(self):

        if self._optimizer_sequence_dimensions not in self.allowed_optimizer_sequence_dimensions:
            fallback_dimension = self.allowed_optimizer_sequence_dimensions[0]
            self.log.info(f"Selected optimization dimensions ({self._optimizer_sequence_dimensions}) "
                          f"are not in the allowed optimizer dimensions ({self.allowed_optimizer_sequence_dimensions}),"
                          f" choosing fallback dimension {fallback_dimension}. ")
            self._optimizer_sequence_dimensions = fallback_dimension

        possible_scan_sequences = self._allowed_sequences(self._optimizer_sequence_dimensions)

        if self._scan_sequence is None or self._scan_sequence not in possible_scan_sequences:

            fallback_scan_sequence = possible_scan_sequences[0]
            self.log.info(f"No valid scan sequence existing ({self._scan_sequence=}),"
                          f" setting scan sequence to {fallback_scan_sequence}.")

            self._scan_sequence = fallback_scan_sequence

    @_optimizer_sequence_dimensions.constructor
    def sequence_dimension_constructor(self, dimensions: Union[list, tuple]) -> tuple:
        if set(dimensions) <= {1, 2}:
            return tuple(dimensions)
        raise ValueError(f"Dimensions must be in {set([1,2])}, received {dimensions=}.")

    @_scan_sequence.constructor
    def sequence_constructor(self, sequence: Union[list, tuple]) -> tuple:
        return tuple(tuple(value) for value in sequence)
