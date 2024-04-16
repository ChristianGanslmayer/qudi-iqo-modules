# -*- coding: utf-8 -*-

"""
This file contains the Qudi Predefined Methods for sequence generator

Qudi is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Qudi is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Qudi. If not, see <http://www.gnu.org/licenses/>.use_DD

Copyright (c) the Qudi Developers. See the COPYRIGHT.txt file at the
top-level directory of this distribution and at <https://github.com/Ulm-IQO/qudi/>
"""

import numpy as np
from qudi.logic.pulsed.pulse_objects import PulseBlock, PulseBlockEnsemble, PulseSequence
from qudi.logic.pulsed.pulse_objects import PredefinedGeneratorBase
from enum import Enum
import re


class TQstates(Enum):
    State00 = '00'
    State01 = '01'
    State10 = '10'
    State11 = '11'


class xq1iGate(PredefinedGeneratorBase):
    def __init__(self, sequencegeneratorlogic, **kwargs):
        super().__init__(sequencegeneratorlogic)
        self.name = 'noop'
        self.qubit = 0
        self.param = 0
        if 'name' in kwargs:
            self.name = kwargs['name']
        if 'qubit' in kwargs:
            self.qubit = kwargs['qubit']
        if 'param' in kwargs:
            self.param = kwargs['param']


    def get_pulse(self, QCQB12_params, accumRotZAng=0):
        match self.name:
            case 'noop':
                pulse = [ self._get_idle_element(length=self.param,
                                                 increment=0) ]
            case 'rz':
                pulse = [ self._get_idle_element(length=0.0e-9,
                                                 increment=0) ]
            case 'sx':
                if self.qubit == 1:
                    pulse = [ self._get_mw_element(length=self.rabi_period / 4,
                                                   increment=0,
                                                   amp=self.microwave_amplitude,
                                                   freq=self.microwave_frequency,
                                                   phase=accumRotZAng) ]
                elif self.qubit == 2:
                    # pulse = [ self._get_multiple_rf_element(length=QCQB12_params['RF_pi'] / 2,
                    #                                         increment=0,
                    #                                         amps=[QCQB12_params['RF_amp0'], QCQB12_params['RF_amp1']],
                    #                                         freqs=[QCQB12_params['RF_freq0'], QCQB12_params['RF_freq1']],
                    #                                         phases=[accumRotZAng, accumRotZAng]) ]
                    pulse = ( self._ddrf_pulse_block(QCQB12_params, type='uc', tau_count=1, rot_phase=accumRotZAng) +
                              self._ddrf_pulse_block(QCQB12_params, type='uc', tau_count=2*QCQB12_params['DD_N']+1,
                                                     rot_phase=accumRotZAng) )
            case 'c0x':
                if self.qubit == 1:
                    pulse = [ self._get_mw_element(length=QCQB12_params['NV_Cpi_len'],
                                                   increment=0,
                                                   amp=QCQB12_params['NV_Cpi_amp'],
                                                   freq=QCQB12_params['NV_Cpi_freq1'],
                                                   phase=accumRotZAng) ]
                elif self.qubit == 2:
                    # pulse = [ self._get_rf_element(length=QCQB12_params['RF_pi'],
                    #                                increment=0,
                    #                                amp=QCQB12_params['RF_amp0'],
                    #                                freq=QCQB12_params['RF_freq0'],
                    #                                phase=accumRotZAng) ]
                    pulse = ( self._ddrf_pulse_block(QCQB12_params, type='c0', tau_count=1, rot_phase=accumRotZAng) +
                              self._ddrf_pulse_block(QCQB12_params, type='uc', tau_count=2*QCQB12_params['DD_N']+1,
                                                     rot_phase=accumRotZAng) )
            case 'c1x':
                if self.qubit == 2:
                    # pulse = [ self._get_rf_element(length=QCQB12_params['RF_pi'],
                    #                                increment=0,
                    #                                amp=QCQB12_params['RF_amp1'],
                    #                                freq=QCQB12_params['RF_freq1'],
                    #                                phase=accumRotZAng) ]
                    pulse = ( self._ddrf_pulse_block(QCQB12_params, type='c1', tau_count=1, rot_phase=accumRotZAng) +
                              self._ddrf_pulse_block(QCQB12_params, type='uc', tau_count=2*QCQB12_params['DD_N']+1,
                                                     rot_phase=accumRotZAng) )
        return pulse


    def _ddrf_pulse_block(self, QCQB12_params, type, tau_count=1, rot_phase=0):
        match type:
            case 'c0':
                phaseOffsets = [0, 180]
            case 'c1':
                phaseOffsets = [180, 0]
            case 'uc':
                phaseOffsets = [0, 0]
        tau = QCQB12_params['cyclesf'] * (1 / QCQB12_params['RF_freq1']) + 1.0e-9
        cycles = ((2 * np.pi * QCQB12_params['RF_freq1']) * (tau)) // (2 * np.pi)
        tau_pulse = (2 * np.pi * cycles) / (2 * np.pi * QCQB12_params['RF_freq1'])
        tau_idle = (tau - tau_pulse) / 2
        phase = self._inst_phase(QCQB12_params['RF_freq1'],
                                 QCQB12_params['RF_freq0'],
                                 0.0,
                                 tau,
                                 0)
        MW_elements = {
            'MWpix': self._get_mw_element(length=self.rabi_period / 2,
                                          increment=0,
                                          amp=self.microwave_amplitude,
                                          freq=self.microwave_frequency,
                                          phase=0),
            'MWpiy': self._get_mw_element(length=self.rabi_period / 2,
                                          increment=0,
                                          amp=self.microwave_amplitude,
                                          freq=self.microwave_frequency,
                                          phase=90)
        }
        pulse_block = []
        for n in range(1, QCQB12_params['DD_N']+1):
            if n != 1:
                del pulse_block[len(pulse_block) - 3:len(pulse_block)]
                numTau = 2
            else:
                numTau = 1
            #k=1,4,7,10
            RF_phase = np.mod( (tau_count-1)*phase + phaseOffsets[n%2] + rot_phase, 360 )
            tauidle_element = self._get_idle_element(length=numTau*tau_idle, increment=0)
            RFtau_element = self._get_rf_element(length=numTau*tau_pulse,
                                                 increment=0,
                                                 amp=QCQB12_params['RF_amp1'],
                                                 freq=QCQB12_params['RF_freq1'],
                                                 phase=RF_phase)
            pulse_block.append(tauidle_element)
            pulse_block.append(RFtau_element)
            pulse_block.append(tauidle_element)
            rotAx = 'x' if n%4 in (1,2) else 'y'
            pulse_block.append(MW_elements['MWpi'+rotAx])
            tau_count += 1
            #k=2,5,8,11
            RF_phase = np.mod( (tau_count-1)*phase + phaseOffsets[n%2] + rot_phase, 360 )
            tauidle_element = self._get_idle_element(length=2*tau_idle, increment=0)
            RFtau_element = self._get_rf_element(length=2*tau_pulse,
                                                 increment=0,
                                                 amp=QCQB12_params['RF_amp1'],
                                                 freq=QCQB12_params['RF_freq1'],
                                                 phase=RF_phase)
            pulse_block.append(tauidle_element)
            pulse_block.append(RFtau_element)
            pulse_block.append(tauidle_element)
            rotAx = 'y' if n%4 in (1,2) else 'x'
            pulse_block.append(MW_elements['MWpi'+rotAx])
            tau_count += 1
            #k=3,6,9,12
            RF_phase = np.mod( (tau_count - 1)*phase + phaseOffsets[n%2] + rot_phase, 360 )
            tauidle_element = self._get_idle_element(length=tau_idle, increment=0)
            RFtau_element = self._get_rf_element(length=tau_pulse,
                                                 increment=0,
                                                 amp=QCQB12_params['RF_amp1'],
                                                 freq=QCQB12_params['RF_freq1'],
                                                 phase=RF_phase)
            pulse_block.append(tauidle_element)
            pulse_block.append(RFtau_element)
            pulse_block.append(tauidle_element)
        return pulse_block


    @staticmethod
    def _inst_phase(freq1, freq2, tauT, tau, phase):
        instantaneous_phase = (2 * np.pi * freq1 * tauT) * (180.0 / np.pi)
        phase_diff = (2 * np.pi * (freq2 - freq1) * tau) * (180.0 / np.pi)
        return (instantaneous_phase + phase_diff + phase)%360



class QB12ControlPredefinedGenerator(PredefinedGeneratorBase):
    """

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sgl = args[0]

    ################################################################################################
    #                             Generation methods for waveforms                                 #
    ################################################################################################

    def generate_QuantumCircuitQB12(self, name='quantumcircuitQB12', Initial_state=TQstates.State00,
                                    NV_Cpi_len=1.0e-6, NV_Cpi_amp=0.05, NV_Cpi_freq1=1.432e9,
                                    RF_freq0=5.1e6, RF_amp0=0.02,  RF_freq1=5.1e6, RF_amp1=0.02,
                                    cyclesf=9, DD_N=2, RF_pi=20.0e-6,
                                    gate_operations = "sx(0)[1], rz(90)[2], sx(0)[2]",
                                    laser_on=20.0e-9, laser_off=60.0e-9):
        """

        """
        for methodParams in self._sgl.generate_method_params.values():
            if methodParams['name'] == name:
                QCQB12_params = methodParams
                break

        gate_noop = xq1iGate(self._sgl, name='noop', param=0.0e-9)
        gate_c0q2x = xq1iGate(self._sgl, name="c0x", qubit=2)
        gate_c1q2x = xq1iGate(self._sgl, name="c1x", qubit=2)
        gate_q1sx = xq1iGate(self._sgl, name="sx", qubit=1)

        gate_list = []
        regex_pattern = re.compile("([^(]*)\(([0-9\.eE\+\-]*)\)\[([0-9]*)\]")
        for gate in gate_operations.replace(" ", "").split(","):
            regex_match = regex_pattern.match(gate)
            gate_name = regex_match.group(1)
            param = float(regex_match.group(2))
            qubit = int(regex_match.group(3))
            gate_list.append( xq1iGate(self._sgl, name=gate_name, qubit=qubit, param=param) )

        # create blocks for initialization
        initialblock_list=[]
        match Initial_state.value:
            case '00':
                initialblock_list += gate_noop.get_pulse(QCQB12_params)
            case '01':
                initialblock_list += gate_c0q2x.get_pulse(QCQB12_params)
            case '10':
                initialblock_list += 2*gate_q1sx.get_pulse(QCQB12_params)
            case '11':
                initialblock_list += 2*gate_q1sx.get_pulse(QCQB12_params) + gate_c1q2x.get_pulse(QCQB12_params)

        # create blocks for executing circuit operations
        opersblock_list=[]
        accum_rotZAng = [0, 0]
        for gate in gate_list:
            if gate.name == 'rz':
                accum_rotZAng[gate.qubit-1] += gate.param
            opersblock_list += gate.get_pulse(QCQB12_params, accum_rotZAng[gate.qubit-1])

        # create blocks for readout of populations of computational basis states
        laser_reps = int(self.laser_length / (laser_on + laser_off))
        laser_block = laser_reps*[ self._get_laser_element(length=laser_on, increment=0),
                                   self._get_idle_element(length=laser_off, increment=0) ]
        readblock00_list = gate_noop.get_pulse(QCQB12_params)
        readblock01_list = gate_c0q2x.get_pulse(QCQB12_params)
        readblock10_list = 2*gate_q1sx.get_pulse(QCQB12_params)
        readblock11_list = gate_c1q2x.get_pulse(QCQB12_params) + 2*gate_q1sx.get_pulse(QCQB12_params)

        # combine blocks for init, operations and readout into one state tomography block (sequentially reading the population of each basis state)
        statetomo_block = PulseBlock(name=name)
        for readout_block in [readblock00_list, readblock01_list, readblock10_list, readblock11_list]:
            for pulse in initialblock_list:
                statetomo_block.append(pulse)
            for pulse in opersblock_list:
                statetomo_block.append(pulse)
            for pulse in readout_block:
                statetomo_block.append(pulse)
            for laser_trig in laser_block:
                statetomo_block.append(laser_trig)
            statetomo_block.append( self._get_idle_element(length=self.wait_time, increment=0) )
            statetomo_block.append( self._get_idle_element(length=self.laser_delay, increment=0) )

        # Create block ensemble
        created_blocks = list()
        created_ensembles = list()
        created_sequences = list()
        created_blocks.append(statetomo_block)
        block_ensemble = PulseBlockEnsemble(name=name, rotating_frame=True)
        block_ensemble.append((statetomo_block.name, 0))

        # Create and append sync trigger block if needed
        self._add_trigger(created_blocks=created_blocks, block_ensemble=block_ensemble)

        # get tau array for measurement ticks
        tau_array = (0) + np.arange(4) * (1)
        # add metadata to invoke settings later on
        number_of_lasers = 4
        block_ensemble.measurement_information['alternating'] = False
        block_ensemble.measurement_information['laser_ignore_list'] = list()
        block_ensemble.measurement_information['controlled_variable'] = tau_array
        block_ensemble.measurement_information['units'] = ('s', '')
        block_ensemble.measurement_information['number_of_lasers'] = number_of_lasers
        block_ensemble.measurement_information['counting_length'] = self._get_ensemble_count_length(
                            ensemble=block_ensemble, created_blocks=created_blocks)

        # append ensemble to created ensembles
        created_ensembles.append(block_ensemble)
        return created_blocks, created_ensembles, created_sequences
