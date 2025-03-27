# -*- coding: utf-8 -*-

"""
This file contains the Qudi Predefined Methods for dynamical decoupling sequences

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

import numpy as np
from qudi.logic.pulsed.pulse_objects import PulseBlock, PulseBlockEnsemble, PulseSequence
from qudi.logic.pulsed.pulse_objects import PredefinedGeneratorBase


class DDPredefinedGenerator(PredefinedGeneratorBase):
    """

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_xy8_tau(self, name='xy8_tau', tau_start=0.5e-6, tau_step=0.01e-6, num_of_points=50,
                         xy8_order=4, alternating=True, laser_on=20.0e-9, laser_off=60.0e-9):
        """

        """
        created_blocks = list()
        created_ensembles = list()
        created_sequences = list()

        # get tau array for measurement ticks
        tau_array = tau_start + np.arange(num_of_points) * tau_step
        # calculate "real" start length of tau due to finite pi-pulse length
        tau_pspacing_start = self.tau_2_pulse_spacing(tau_start)

        # create the elements
        waiting_element = self._get_idle_element(length=self.wait_time, increment=0)

        laser_block = []
        laser_reps = int(self.laser_length / (laser_on + laser_off))
        for n in range(laser_reps):
            laser_block.append(self._get_laser_element(length=laser_on, increment=0))
            laser_block.append(self._get_idle_element(length=laser_off, increment=0))
        delay_element = self._get_delay_gate_element()
        initpihalf_element = self._get_mw_element(length=self.rabi_period / 4,
                                              increment=0,
                                              amp=self.microwave_amplitude,
                                              freq=self.microwave_frequency,
                                              phase=init_phase)
        readpihalf_element = self._get_mw_element(length=self.rabi_period / 4,
                                              increment=0,
                                              amp=self.microwave_amplitude,
                                              freq=self.microwave_frequency,
                                              phase=read_phase)
        readaltpihalf_element = self._get_mw_element(length=self.rabi_period / 4,
                                              increment=0,
                                              amp=self.microwave_amplitude,
                                              freq=self.microwave_frequency,
                                              phase=read_phase+180)

        pix_element = self._get_mw_element(length=self.rabi_period / 2,
                                           increment=0,
                                           amp=self.microwave_amplitude,
                                           freq=self.microwave_frequency,
                                           phase=0)
        piy_element = self._get_mw_element(length=self.rabi_period / 2,
                                           increment=0,
                                           amp=self.microwave_amplitude,
                                           freq=self.microwave_frequency,
                                           phase=90)
        tauhalf_element = self._get_idle_element(length=tau_pspacing_start / 2, increment=tau_step / 2)
        tau_element = self._get_idle_element(length=tau_pspacing_start, increment=tau_step)

        # Create block and append to created_blocks list
        xy8_block = PulseBlock(name=name)
        xy8_block.append(initpihalf_element)
        xy8_block.append(tauhalf_element)
        for n in range(xy8_order):
            xy8_block.append(pix_element)
            xy8_block.append(tau_element)
            xy8_block.append(piy_element)
            xy8_block.append(tau_element)
            xy8_block.append(pix_element)
            xy8_block.append(tau_element)
            xy8_block.append(piy_element)
            xy8_block.append(tau_element)
            xy8_block.append(piy_element)
            xy8_block.append(tau_element)
            xy8_block.append(pix_element)
            xy8_block.append(tau_element)
            xy8_block.append(piy_element)
            xy8_block.append(tau_element)
            xy8_block.append(pix_element)
            if n != xy8_order - 1:
                xy8_block.append(tau_element)
        xy8_block.append(tauhalf_element)
        xy8_block.append(readpihalf_element)
        for i, laser_trig in enumerate(laser_block):
            xy8_block.append(laser_trig)
        xy8_block.append(delay_element)
        xy8_block.append(waiting_element)
        if alternating:
            xy8_block.append(initpihalf_element)
            xy8_block.append(tauhalf_element)
            for n in range(xy8_order):
                xy8_block.append(pix_element)
                xy8_block.append(tau_element)
                xy8_block.append(piy_element)
                xy8_block.append(tau_element)
                xy8_block.append(pix_element)
                xy8_block.append(tau_element)
                xy8_block.append(piy_element)
                xy8_block.append(tau_element)
                xy8_block.append(piy_element)
                xy8_block.append(tau_element)
                xy8_block.append(pix_element)
                xy8_block.append(tau_element)
                xy8_block.append(piy_element)
                xy8_block.append(tau_element)
                xy8_block.append(pix_element)
                if n != xy8_order - 1:
                    xy8_block.append(tau_element)
            xy8_block.append(tauhalf_element)
            xy8_block.append(readaltpihalf_element)
            for i, laser_trig in enumerate(laser_block):
                xy8_block.append(laser_trig)
            xy8_block.append(delay_element)
            xy8_block.append(waiting_element)
        created_blocks.append(xy8_block)

        # Create block ensemble
        block_ensemble = PulseBlockEnsemble(name=name, rotating_frame=True)
        block_ensemble.append((xy8_block.name, num_of_points - 1))

        # Create and append sync trigger block if needed
        self._add_trigger(created_blocks=created_blocks, block_ensemble=block_ensemble)

        # add metadata to invoke settings later on
        number_of_lasers = num_of_points * 2 if alternating else num_of_points
        block_ensemble.measurement_information['alternating'] = alternating
        block_ensemble.measurement_information['laser_ignore_list'] = list()
        block_ensemble.measurement_information['controlled_variable'] = tau_array
        block_ensemble.measurement_information['units'] = ('s', '')
        block_ensemble.measurement_information['labels'] = ('Tau', 'Signal')
        block_ensemble.measurement_information['number_of_lasers'] = number_of_lasers
        block_ensemble.measurement_information['counting_length'] = self._get_ensemble_count_length(
            ensemble=block_ensemble, created_blocks=created_blocks)

        # append ensemble to created ensembles
        created_ensembles.append(block_ensemble)
        return created_blocks, created_ensembles, created_sequences

    def generate_xy8_freq(self, name='xy8_freq', freq_start=0.1e6, freq_step=0.01e6,
                          num_of_points=50, xy8_order=4, alternating=True, laser_on=20.0e-9, laser_off=60.0e-9):
        """

        """
        created_blocks = list()
        created_ensembles = list()
        created_sequences = list()

        # get frequency array for measurement ticks
        freq_array = freq_start + np.arange(num_of_points) * freq_step
        # get tau array from freq array
        tau_array = 1 / (2 * freq_array)
        tau_pspacing_array = self.tau_2_pulse_spacing(tau_array)

        # Convert back to frequency in order to account for clipped values
        freq_array = 1 / (2 * (self.tau_2_pulse_spacing(tau_pspacing_array, inverse=True)))

        # create the elements
        waiting_element = self._get_idle_element(length=self.wait_time, increment=0)
        laser_block = []
        laser_reps = int(self.laser_length / (laser_on + laser_off))
        for n in range(laser_reps):
            laser_block.append(self._get_laser_element(length=laser_on, increment=0))
            laser_block.append(self._get_idle_element(length=laser_off, increment=0))
        delay_element = self._get_delay_gate_element()
        pihalf_element = self._get_mw_element(length=self.rabi_period / 4,
                                              increment=0,
                                              amp=self.microwave_amplitude,
                                              freq=self.microwave_frequency,
                                              phase=0)
        # Use a 180 deg phase shifted pulse as 3pihalf pulse if microwave channel is analog
        if self.microwave_channel.startswith('a'):
            pi3half_element = self._get_mw_element(length=self.rabi_period / 4,
                                                   increment=0,
                                                   amp=self.microwave_amplitude,
                                                   freq=self.microwave_frequency,
                                                   phase=180)
        else:
            pi3half_element = self._get_mw_element(length=3 * self.rabi_period / 4,
                                                   increment=0,
                                                   amp=self.microwave_amplitude,
                                                   freq=self.microwave_frequency,
                                                   phase=0)
        pix_element = self._get_mw_element(length=self.rabi_period / 2,
                                           increment=0,
                                           amp=self.microwave_amplitude,
                                           freq=self.microwave_frequency,
                                           phase=0)
        piy_element = self._get_mw_element(length=self.rabi_period / 2,
                                           increment=0,
                                           amp=self.microwave_amplitude,
                                           freq=self.microwave_frequency,
                                           phase=90)

        # Create block and append to created_blocks list
        xy8_block = PulseBlock(name=name)
        for ii, tau in enumerate(tau_pspacing_array):
            tauhalf_element = self._get_idle_element(length=tau / 2, increment=0)
            tau_element = self._get_idle_element(length=tau, increment=0)
            xy8_block.append(pihalf_element)
            xy8_block.append(tauhalf_element)
            for n in range(xy8_order):
                xy8_block.append(pix_element)
                xy8_block.append(tau_element)
                xy8_block.append(piy_element)
                xy8_block.append(tau_element)
                xy8_block.append(pix_element)
                xy8_block.append(tau_element)
                xy8_block.append(piy_element)
                xy8_block.append(tau_element)
                xy8_block.append(piy_element)
                xy8_block.append(tau_element)
                xy8_block.append(pix_element)
                xy8_block.append(tau_element)
                xy8_block.append(piy_element)
                xy8_block.append(tau_element)
                xy8_block.append(pix_element)
                if n != xy8_order - 1:
                    xy8_block.append(tau_element)
            xy8_block.append(tauhalf_element)
            xy8_block.append(pihalf_element)
            for i, laser_trig in enumerate(laser_block):
                xy8_block.append(laser_trig)
            xy8_block.append(delay_element)
            xy8_block.append(waiting_element)
            if alternating:
                xy8_block.append(pihalf_element)
                xy8_block.append(tauhalf_element)
                for n in range(xy8_order):
                    xy8_block.append(pix_element)
                    xy8_block.append(tau_element)
                    xy8_block.append(piy_element)
                    xy8_block.append(tau_element)
                    xy8_block.append(pix_element)
                    xy8_block.append(tau_element)
                    xy8_block.append(piy_element)
                    xy8_block.append(tau_element)
                    xy8_block.append(piy_element)
                    xy8_block.append(tau_element)
                    xy8_block.append(pix_element)
                    xy8_block.append(tau_element)
                    xy8_block.append(piy_element)
                    xy8_block.append(tau_element)
                    xy8_block.append(pix_element)
                    if n != xy8_order - 1:
                        xy8_block.append(tau_element)
                xy8_block.append(tauhalf_element)
                xy8_block.append(pi3half_element)
                for i, laser_trig in enumerate(laser_block):
                    xy8_block.append(laser_trig)
                xy8_block.append(delay_element)
                xy8_block.append(waiting_element)
        created_blocks.append(xy8_block)

        # Create block ensemble
        block_ensemble = PulseBlockEnsemble(name=name, rotating_frame=True)
        block_ensemble.append((xy8_block.name, 0))

        # Create and append sync trigger block if needed
        self._add_trigger(created_blocks=created_blocks, block_ensemble=block_ensemble)

        # add metadata to invoke settings later on
        number_of_lasers = num_of_points * 2 if alternating else num_of_points
        block_ensemble.measurement_information['alternating'] = alternating
        block_ensemble.measurement_information['laser_ignore_list'] = list()
        block_ensemble.measurement_information['controlled_variable'] = freq_array
        block_ensemble.measurement_information['units'] = ('Hz', '')
        block_ensemble.measurement_information['labels'] = ('Frequency', 'Signal')
        block_ensemble.measurement_information['number_of_lasers'] = number_of_lasers
        block_ensemble.measurement_information['counting_length'] = self._get_ensemble_count_length(
            ensemble=block_ensemble, created_blocks=created_blocks)

        # append ensemble to created ensembles
        created_ensembles.append(block_ensemble)
        return created_blocks, created_ensembles, created_sequences


    def generate_dd_deer(self, name='dd_deer', tau1=400e-9, tau2_start=0e-9, tau2_step=5e-9, num_of_points=32,
                         mw_freq2=2.664e9, mw_amp2=225e-3, rabi_period2=75e-9, xy8_order=1, read_phase=0,
                         alternating=True, initFlipNV1notNV2=False):
        """

        """
        MW_elements = {
            'MWpix1': self._get_mw_element(length=self.rabi_period / 2,
                                           increment=0,
                                           amp=self.microwave_amplitude,
                                           freq=self.microwave_frequency,
                                           phase=0),
            'MWpiy1': self._get_mw_element(length=self.rabi_period / 2,
                                           increment=0,
                                           amp=self.microwave_amplitude,
                                           freq=self.microwave_frequency,
                                           phase=90),
            'MWpix2': self._get_mw_element(length=rabi_period2 / 2,
                                           increment=0,
                                           amp=mw_amp2,
                                           freq=mw_freq2,
                                           phase=0),
            'MWpiy2': self._get_mw_element(length=rabi_period2 / 2,
                                           increment=0,
                                           amp=mw_amp2,
                                           freq=mw_freq2,
                                           phase=90),
        }
        laser_block = self._get_xq_laser_block()
        tau2_array = tau2_start + np.arange(num_of_points) * tau2_step
        #initpulse_lengths = [self.rabi_period / 4, 3 * self.rabi_period / 4] if alternating else [self.rabi_period / 4]
        flipNV2_list = [False, True] if alternating else [False]
        rotAxes = ['x', 'y', 'x', 'y', 'y', 'x', 'y', 'x']

        #print( type(tau1), type(self.rabi_period), type(rabi_period2), type(tau2_array[0]) )

        pulse_block = PulseBlock(name=name)
        for tau2 in tau2_array:
            #for initpulse_length in initpulse_lengths:
            for flipNV2 in flipNV2_list:
                if flipNV2:
                    if not initFlipNV1notNV2:
                        pulse_block.append(MW_elements['MWpix2'])
                    else:
                        pulse_block.append(MW_elements['MWpix1'])
                pulse_block.append(self._get_mw_element(length=self.rabi_period / 4, increment=0,
                                                        amp=self.microwave_amplitude, freq=self.microwave_frequency, phase=0))
                delayDelta = - self.rabi_period / 8
                for n in range(xy8_order):
                    for rotAx in rotAxes:
                        pulse_block.append(self._get_idle_element(length=tau1 / 2 - self.rabi_period / 4 + delayDelta , increment=0))
                        pulse_block.append(MW_elements['MWpi'+rotAx+'1'])
                        pulse_block.append(self._get_idle_element(length=tau1 / 2 - self.rabi_period / 4 - tau2 - rabi_period2 / 4, increment=0))
                        pulse_block.append(MW_elements['MWpi'+ rotAx+'2'])
                        delayDelta = tau2 - rabi_period2 / 4
                pulse_block.append(self._get_idle_element(length=delayDelta - self.rabi_period / 8, increment=0))
                pulse_block.append(self._get_mw_element(length=self.rabi_period / 4, increment=0, amp=self.microwave_amplitude,
                                                        freq=self.microwave_frequency, phase=read_phase))
                if delayDelta - self.rabi_period / 8 < 0:
                    pulse_block.append(self._get_mw_element(length=min(-(delayDelta - self.rabi_period / 8), rabi_period2 / 2),
                                                            increment=0, amp=mw_amp2, freq=mw_freq2, phase=0))
                if not flipNV2:
                    pulse_block.append(MW_elements['MWpix2'])
                else:
                    if initFlipNV1notNV2:
                        pulse_block.append(MW_elements['MWpix2'])
                pulse_block.extend(laser_block)
                pulse_block.append(self._get_idle_element(length=self.laser_delay, increment=0))
                pulse_block.append(self._get_idle_element(length=self.wait_time, increment=0))

        # Create block ensemble
        created_blocks = list()
        created_ensembles = list()
        created_sequences = list()
        created_blocks.append(pulse_block)
        block_ensemble = PulseBlockEnsemble(name=name, rotating_frame=True)
        block_ensemble.append((pulse_block.name, 0))

        # Create and append sync trigger block if needed
        self._add_trigger(created_blocks=created_blocks, block_ensemble=block_ensemble)

        # add metadata to invoke settings later on
        number_of_lasers = num_of_points * 2 if alternating else num_of_points
        block_ensemble.measurement_information['alternating'] = alternating
        block_ensemble.measurement_information['laser_ignore_list'] = list()
        block_ensemble.measurement_information['controlled_variable'] = tau2_array * 8 * xy8_order
        block_ensemble.measurement_information['units'] = ('s', '')
        block_ensemble.measurement_information['labels'] = ('t_evol', 'Signal')
        block_ensemble.measurement_information['number_of_lasers'] = number_of_lasers
        block_ensemble.measurement_information['counting_length'] = self._get_ensemble_count_length(
            ensemble=block_ensemble, created_blocks=created_blocks)

        # append ensemble to created ensembles
        created_ensembles.append(block_ensemble)
        return created_blocks, created_ensembles, created_sequences

    ################################################################################################
    #                             Generation methods for sequences                                 #
    ################################################################################################
