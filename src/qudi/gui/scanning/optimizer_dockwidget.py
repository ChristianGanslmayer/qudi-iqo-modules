# -*- coding: utf-8 -*-

"""
This file contains a QDockWidget subclass to display the scanner optimizer results.

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

__all__ = ('OptimizerDockWidget',)

import numpy as np
from PySide2 import QtCore, QtWidgets
from pyqtgraph import PlotDataItem, mkPen
import copy as cp

from qudi.util.widgets.plotting.plot_widget import DataSelectionPlotWidget
from qudi.util.widgets.plotting.plot_item import DataImageItem, XYPlotItem
from qudi.util.colordefs import QudiPalette


class OptimizerDockWidget(QtWidgets.QDockWidget):
    """
    """

    def __init__(self, axes, plot_dims, sequence, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle('Optimizer')
        self.setObjectName('optimizer_dockWidget')

        self._last_optimal_pos = {}
        self._last_peakcnt = None
        self._last_optimal_sigma = {}
        self._scanner_sequence = sequence
        self._plot_widgets = []

        self.pos_ax_label = QtWidgets.QLabel(f'({", ".join(axes)}):')
        self.pos_ax_label.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.result_label = QtWidgets.QLabel(f'({", ".join(["?"]*len(axes))}):')
        self.result_label.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        label_layout = QtWidgets.QHBoxLayout()
        label_layout.addWidget(self.pos_ax_label)
        label_layout.addWidget(self.result_label)
        label_layout.setStretch(1, 1)

        layout = QtWidgets.QGridLayout()
        # fill list of all optimizer subplot widgets
        for i_col, n_dim in enumerate(plot_dims):
            if n_dim == 1:
                plot_item = XYPlotItem(pen=mkPen(QudiPalette.c1, style=QtCore.Qt.DotLine),
                                       symbol='o',
                                       symbolPen=QudiPalette.c1,
                                       symbolBrush=QudiPalette.c1,
                                       symbolSize=7)
                # Fit overlay (solid line)
                fit_plot_item = XYPlotItem(pen=mkPen(QudiPalette.c2))

                plot1d_widget = DataSelectionPlotWidget()
                plot1d_widget.set_selection_mutable(False)
                plot1d_widget.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
                # Add the fit first so it renders under the markers
                plot1d_widget.addItem(fit_plot_item)
                plot1d_widget.addItem(plot_item)
                plot1d_widget.add_marker_selection((0, 0),
                                                   mode=DataSelectionPlotWidget.SelectionMode.X)
                self._plot_widgets.append({'widget': plot1d_widget, 'plot_1d': plot_item,
                                           'fit_1d': fit_plot_item, 'dim': 1})
            elif n_dim == 2:
                plot2d_widget = DataSelectionPlotWidget()
                plot2d_widget.setAspectLocked(lock=True, ratio=1)
                plot2d_widget.set_selection_mutable(False)
                plot2d_widget.add_marker_selection((0, 0),
                                                   mode=DataSelectionPlotWidget.SelectionMode.XY)
                image_item = DataImageItem()
                plot2d_widget.addItem(image_item)
                self._plot_widgets.append({'widget': plot2d_widget,
                                           'image_2d': image_item,
                                           'dim': 2})
            else:
                raise ValueError(f"Optimizer widget can have axis dim= 1 or 2, not {n_dim}")

            layout.addWidget(self._plot_widgets[-1]['widget'], 0, i_col)

        layout.addLayout(label_layout, 1, 0, 1, 2)
        layout.setRowStretch(0, 1)

        # --- Fit parameter table (param, value, ±1σ) ---
        self._fit_table = QtWidgets.QTableWidget(0, 3, self)
        self._fit_table.setHorizontalHeaderLabels(['param', 'value', '±1σ'])
        self._fit_table.horizontalHeader().setStretchLastSection(True)
        self._fit_table.verticalHeader().setVisible(False)
        self._fit_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._fit_table.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._fit_table.setAlternatingRowColors(True)
        layout.addWidget(self._fit_table, 2, 0, 1, 2)

        # Accumulate rows per axis so (1,1,1) shows x, y, z
        self._fit_param_rows = {}

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setWidget(widget)

    @property
    def scan_sequence(self):
        return cp.copy(self._scanner_sequence)

    @scan_sequence.setter
    def scan_sequence(self, sequence):
        self._scanner_sequence = sequence

    def _get_all_widgets_part(self, part='widget', dim=None):
        widgets_1d = [wid for wid in self._plot_widgets if wid['dim'] == 1]
        widgets_2d = [wid for wid in self._plot_widgets if wid['dim'] == 2]

        if dim is None:
            return [wid[part] for wid in self._plot_widgets]
        elif dim == 1:
            return [wid[part] for wid in widgets_1d]
        elif dim == 2:
            return [wid[part] for wid in widgets_2d]
        else:
            raise ValueError

    def _get_widget_part(self, axs, part='widget'):
        """
        Based on the given axes, return the corresponding widget.
        Will keep the order given by self._scan_sequence. Eg. axs=('x','y') will
        give the second 2d widget for the scan order [('phi', 'z'), ('x','y')]
        """

        seqs_1d = [tuple(seq) for seq in self._scanner_sequence if len(seq) == 1]
        seqs_2d = [tuple(seq) for seq in self._scanner_sequence if len(seq) == 2]
        widgets_1d = [wid for wid in self._plot_widgets if wid['dim'] == 1]
        widgets_2d = [wid for wid in self._plot_widgets if wid['dim'] == 2]

        widget = None
        axs = tuple(axs)

        try:
            if len(axs) == 1:
                idx = seqs_1d.index(axs)
                widget = widgets_1d[idx]
            elif len(axs) == 2:
                idx = seqs_2d.index(axs)
                widget = widgets_2d[idx]
            else:
                raise ValueError
        except ValueError:
            raise ValueError(f"Given axs {axs} not in scanner sequence. Couldn't find widget.")

        return widget[part]

    def get_plot_widget(self, axs):
        return self._get_widget_part(axs, part='widget')

    def toogle_crosshair(self, axs=None, enabled=False):
        """
        Toggle all or specified crosshairds of 2d widgets.
        """
        if axs:
            plot2d_widgets = [self._get_widget_part(axs, part='widget')]
        else:
            plot2d_widgets = self._get_all_widgets_part(dim=2)

        for wid in plot2d_widgets:
            if enabled:
                wid.show_marker_selections()
            else:
                wid.hide_marker_selections()

    def toogle_marker(self, axs=None, enabled=False):
        """
        Toggle all or specified markers of 1d widgets.
        """
        if axs:
            plot1d_widgets = [self._get_widget_part(axs, part='widget')]
        else:
            plot1d_widgets = self._get_all_widgets_part(dim=1)

        for wid in plot1d_widgets:
            if enabled:
                wid.show_marker_selections()
            else:
                wid.hide_marker_selections()

    def set_2d_position(self, pos, axs, sigma=None, peakcnt=None):
        widget = self.get_plot_widget(axs)
        widget.move_marker_selection(pos, index=0)

        self._last_optimal_pos[axs[0]] = pos[0]
        self._last_optimal_pos[axs[1]] = pos[1]
        if sigma:
            self._last_optimal_sigma[axs[0]] = sigma[0]
            self._last_optimal_sigma[axs[1]] = sigma[1]
        self._last_peakcnt = peakcnt

        self.update_result_label()

    def set_1d_position(self, pos, axs, sigma=None, peakcnt=None):
        widget = self.get_plot_widget(axs)
        widget.move_marker_selection((pos, 0), index=0)

        self._last_optimal_pos[axs[0]] = pos
        if sigma:
            self._last_optimal_sigma[axs[0]] = sigma
        self._last_peakcnt = peakcnt

        self.update_result_label()

    def update_result_label(self):
        def _dict_2_str(in_dict, print_only_key=False):
            out_str = "("

            for key, val in dict(sorted(in_dict.items())).items():
                if print_only_key:
                    out_str += f"{key}, "
                else:
                    if val:
                        out_str += f"{val*1e6:.3f}, "
                    else:
                        out_str += "?, "

            out_str = out_str.rstrip(', ')
            out_str += ")"
            return out_str

        axis_str = _dict_2_str(self._last_optimal_pos, True) + "= "
        pos_str = _dict_2_str(self._last_optimal_pos)
        sigma_str = _dict_2_str(self._last_optimal_sigma)
        self.pos_ax_label.setText(axis_str)
        peakcnt_label = "" if self._last_peakcnt is None else f", peak cnts: {self._last_peakcnt:.2E}"
        self.result_label.setText(pos_str + " µm,  σ= " + sigma_str + " µm" + peakcnt_label)

    def set_image(self, image, axs, extent=None):

        image_item = self._get_widget_part(axs, 'image_2d')

        image_item.set_image(image=image)
        if extent is not None:
            image_item.set_image_extent(extent)

    def get_plot_item(self, axs):
        return self._get_widget_part(axs, 'plot_1d')

    def get_plot_fit_item(self, axs):
        return self._get_widget_part(axs, 'fit_1d')

    def set_plot_data(self, axs, x=None, y=None):

        plot_item = self.get_plot_item(axs)

        if x is None and y is None:
            plot_item.clear()
            return
        elif x is None:
            x = plot_item.xData
            if x is None or len(x) != len(y):
                x = np.arange(len(y))
        elif y is None:
            y = plot_item.yData
            if y is None or len(x) != len(y):
                y = np.zeros(len(x))

        nan_mask = np.isnan(y)
        if nan_mask.all():
            plot_item.clear()
        else:
            plot_item.setData(x=x[~nan_mask], y=y[~nan_mask])
        return

    def set_fit_data(self, axs, x=None, y=None):
        """
        Plot the fit curve for the given 1D axes.
        If x is not provided, inherit the x-array from the data points plot so the
        fit overlays perfectly. We avoid falling back to indices because that breaks
        units (e.g., shows 0..N-1 "meters").
        """
        fit_plot_item = self.get_plot_fit_item(axs)

        # 1) Clear if both are None
        if x is None and y is None:
            fit_plot_item.clear()
            return

        # 2) Ensure we have y
        if y is None:
            y = fit_plot_item.yData
            if y is None:
                # Nothing sensible to draw
                fit_plot_item.clear()
                return

        # 3) Fill x if missing: inherit from the data points plot item
        if x is None:
            data_plot_item = self.get_plot_item(axs)
            x = data_plot_item.xData

        # 4) Final sanity: if x still unusable, do not draw (better than wrong units)
        if x is None or len(x) != len(y):
            # nothing to draw correctly; keep previous fit (if any) and exit
            return

        # 5) Draw
        fit_plot_item.setData(x=x, y=y)
        return

    def set_image_label(self, axis, axs=None, text=None, units=None):
        if len(axs) != 2:
            raise ValueError(f"For setting a image label, must be a 2d axes, not {axs}")

        widget = self._get_widget_part(axs, 'widget')

        widget.setLabel(axis=axis, text=text, units=units)

    def set_plot_label(self, axis, axs=None, text=None, units=None):
        if len(axs) != 1:
            raise ValueError(f"For setting a image label, must be a 1d axes, not {axs}")

        widget = self._get_widget_part(axs, 'widget')

        widget.setLabel(axis=axis, text=text, units=units)

    def clear_fit_params(self):
        """Clear stored fit parameters and the table."""
        self._fit_param_rows.clear()
        self._fit_table.setRowCount(0)

    def set_fit_params_from_result(self, fit_result, axs):
        """
        Store and display fit parameters for the given 1D/2D axes.
        Handles both single-Gaussian (center/sigma/amplitude) and two-Gaussian
        fits with prefixes 'g1_' and 'g2_' from lmfit GaussianModel.
        Accumulates per-axis rows so (1,1,1) shows x, y, z together.
        """
        axs = tuple(axs)
        params = getattr(fit_result, 'params', {}) or {}

        rows = []

        # --- detect two-Gaussian (lmfit) ---
        is_two_g = any(name.startswith('g1_') for name in params.keys()) or \
                   any(name.startswith('g2_') for name in params.keys())

        if is_two_g:
            def grab(prefix, key):
                p = params.get(f'{prefix}{key}')
                return (getattr(p, 'value', None), getattr(p, 'stderr', None)) if p is not None else (None, None)

            # We show peak *height* as well (area converted), which is often more intuitive.
            def height(prefix):
                area, _ = grab(prefix, 'amplitude')  # lmfit Gaussian amplitude is area
                sigma, _ = grab(prefix, 'sigma')
                import math
                if sigma is None or sigma <= 0 or area is None:
                    return None
                return area / (sigma * math.sqrt(2 * math.pi))

            for label, prefix in (('g1', 'g1_'), ('g2', 'g2_')):
                c, c_err = grab(prefix, 'center')
                s, s_err = grab(prefix, 'sigma')
                a_area, a_err = grab(prefix, 'amplitude')
                h = height(prefix)
                rows.extend([
                    (f'{label}: center', c, c_err),
                    (f'{label}: sigma', s, s_err),
                    (f'{label}: area', a_area, a_err),
                    (f'{label}: height', h, None),
                ])

            # background, if present
            if 'bkg_c' in params:
                rows.append(('background', params['bkg_c'].value, getattr(params['bkg_c'], 'stderr', None)))

        else:
            # --- single Gaussian (your original Gaussian with center/sigma/amplitude) ---
            def getv(name):
                p = params.get(name)
                return (getattr(p, 'value', None), getattr(p, 'stderr', None)) if p is not None else (None, None)

            a, a_e = getv('amplitude')
            c, c_e = getv('center')
            s, s_e = getv('sigma')
            off, off_e = getv('offset')
            # show what exists
            for label, val, err in (('amplitude', a, a_e), ('center', c, c_e), ('sigma', s, s_e),
                                    ('offset', off, off_e)):
                if val is not None or err is not None:
                    rows.append((label, val, err))

        # Always add some fit quality info so the section isn't empty
        for k in ('redchi', 'aic', 'bic'):
            if hasattr(fit_result, k):
                rows.append((k, getattr(fit_result, k), None))

        # Fallback
        if not rows:
            rows = [('(no parameters found)', None, None)]

        # Store per-axis and rebuild table (keep your existing accumulation logic)
        self._fit_param_rows[axs] = rows

        # Preferred order from scanner sequence with fallback to received order
        seq_1d = [tuple(seq) for seq in self._scanner_sequence if len(seq) == 1]
        seq_2d = [tuple(seq) for seq in self._scanner_sequence if len(seq) == 2]
        preferred = [*seq_1d, *seq_2d]
        groups = [g for g in preferred if g in self._fit_param_rows]
        for g in self._fit_param_rows.keys():
            if g not in groups:
                groups.append(g)

        if not groups:
            self._fit_table.setRowCount(0)
            self._fit_table.viewport().update()
            return

        total_rows = sum(1 + len(self._fit_param_rows[g]) for g in groups)
        self._fit_table.clearContents()
        self._fit_table.setRowCount(total_rows)

        r = 0
        for g in groups:
            header = f"{g[0]} fit parameters" if len(g) == 1 else f"{g} fit parameters"
            header_item = QtWidgets.QTableWidgetItem(header)
            header_item.setFlags(QtCore.Qt.ItemIsEnabled)
            font = header_item.font();
            font.setBold(True);
            header_item.setFont(font)
            self._fit_table.setItem(r, 0, header_item)
            self._fit_table.setSpan(r, 0, 1, 3)
            r += 1

            for name, val, err in self._fit_param_rows[g]:
                self._fit_table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(name)))
                self._fit_table.setItem(r, 1, QtWidgets.QTableWidgetItem('—' if val is None else f'{val:.6g}'))
                self._fit_table.setItem(r, 2, QtWidgets.QTableWidgetItem('n/a' if err is None else f'{err:.2g}'))
                r += 1

        self._fit_table.viewport().update()