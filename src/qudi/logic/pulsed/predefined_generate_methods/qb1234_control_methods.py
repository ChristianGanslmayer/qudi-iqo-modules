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
import scipy.optimize
from qudi.logic.pulsed.pulse_objects import PulseBlock, PulseBlockEnsemble, PulseSequence
from qudi.logic.pulsed.pulse_objects import PredefinedGeneratorBase
from enum import Enum
import re


class TQstates(Enum):
    State00 = '00'
    State01 = '01'
    State10 = '10'
    State11 = '11'

class TQReadoutCircs(Enum):
    P00 = 1
    P01 = 2
    P10 = 3

class TQQSTReadoutCircs(Enum):
    IX = 1
    IY = 2
    IZ = 3
    XI = 4
    XX = 5
    XY = 6
    XZ = 7
    YI = 8
    YX = 9
    YY = 10
    YZ = 11
    ZI = 12
    ZX = 13
    ZY = 14
    ZZ = 15


class ThrQstates(Enum):
    State000 = '000'
    State001 = '001'
    State010 = '010'
    State011 = '011'
    State100 = '100'
    State101 = '101'
    State110 = '110'
    State111 = '111'

class ThrQReadoutCircs(Enum):
    IIZ1 = 1
    IIZ2 = 2
    IZI1 = 3
    IZI2 = 4
    IZZ2 = 5
    ZII2 = 6
    ZIZ1 = 7
    ZIZ2 = 8
    ZZI2 = 9
    ZZZ2 = 10


class FourQstates(Enum):
    State0000 = '0000'
    State0001 = '0001'
    State0010 = '0010'
    State0011 = '0011'
    State0100 = '0100'
    State0101 = '0101'
    State0110 = '0110'
    State0111 = '0111'
    State1000 = '1000'
    State1001 = '1001'
    State1010 = '1010'
    State1011 = '1011'
    State1100 = '1100'
    State1101 = '1101'
    State1110 = '1110'
    State1111 = '1111'

class FourQReadoutCircs(Enum):
    IIZI1 = 1
    IIZI2 = 2
    IZII1 = 3
    IZII2 = 4
    IZZI2 = 5
    ZIII2 = 6
    ZIZI1 = 7
    ZIZI2 = 8
    ZZII2 = 9
    ZZZI2 = 10
    IIIZ1 = 11
    IIIZ2 = 12
    IIZZ1 = 13
    IIZZ2 = 14
    IZIZ2 = 15
    IZZZ2 = 16
    ZIIZ1 = 17
    ZIIZ2 = 18
    ZIZZ1 = 19
    ZIZZ2 = 20
    ZZIZ2 = 21
    ZZZZ2 = 22


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


    def get_pulse(self, QC_params, accumRotZAng=0):
        match self.name:
            case 'noop':
                pulse = [ self._get_idle_element(length=self.param,
                                                 increment=0) ]
            case 'p00':
                laser_reps = int(self.laser_length / (QC_params['laser_on'] + QC_params['laser_off']))
                pulse = laser_reps * [self._get_laser_element(length=QC_params['laser_on'], increment=0),
                                      self._get_idle_element(length=QC_params['laser_off'], increment=0)]
                pulse += [self._get_idle_element(length=self.laser_delay, increment=0),
                          self._get_idle_element(length=self.wait_time, increment=0)]
            case 'rz':
                if self.qubit == 3:
                    pulse = self._xy8_pulse_block( QC_params['order_z'], QC_params['tau_z']/2 )
                elif self.qubit == 4:
                    pulse = []
                else:
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
                    # pulse = [ self._get_multiple_rf_element(length=QC_params['RF_pi'] / 2,
                    #                                         increment=0,
                    #                                         amps=[QC_params['RF_amp0'], QC_params['RF_amp1']],
                    #                                         freqs=[QC_params['RF_freq0'], QC_params['RF_freq1']],
                    #                                         phases=[accumRotZAng, accumRotZAng]) ]
                    pulse = self._ddrf_pulse_block(QC_params, type='uc', tau_count=1, rot_phase=accumRotZAng)
                elif self.qubit == 3:
                    pulse = self._axy8_pulse_block(QC_params['order_uc'], QC_params['tau_uc'], QC_params['f1_uc'])
                elif self.qubit == 4:
                    pulse = []
            case 'sy':
                if self.qubit == 1:
                    pulse = [ self._get_mw_element(length=self.rabi_period / 4,
                                                   increment=0,
                                                   amp=self.microwave_amplitude,
                                                   freq=self.microwave_frequency,
                                                   phase=90+accumRotZAng) ]
            case 'c0x':
                if self.qubit == 1:
                    pulse = [ self._get_mw_element(length=QC_params['NV_Cpi_len'],
                                                   increment=0,
                                                   amp=QC_params['NV_Cpi_amp'],
                                                   freq=QC_params['NV_Cpi_freq1'],
                                                   phase=accumRotZAng) ]
                elif self.qubit == 2:
                    pulse = [ self._get_rf_element(length=QC_params['RF_pi'],
                                                   increment=0,
                                                   amp=QC_params['RF_amp0'],
                                                   freq=QC_params['RF_freq0'],
                                                   phase=accumRotZAng) ]
                    # pulse = ( self._ddrf_pulse_block(QC_params, type='c0', tau_count=1, rot_phase=accumRotZAng) +
                              # self._ddrf_pulse_block(QC_params, type='uc', tau_count=2*QC_params['DD_N']+1,
                                                     # rot_phase=accumRotZAng) )
            case 'c1x':
                if self.qubit == 2:
                    pulse = [ self._get_rf_element(length=QC_params['RF_pi'],
                                                   increment=0,
                                                   amp=QC_params['RF_amp1'],
                                                   freq=QC_params['RF_freq1'],
                                                   phase=accumRotZAng) ]
                    # pulse = ( self._ddrf_pulse_block(QC_params, type='c1', tau_count=1, rot_phase=accumRotZAng) +
                              # self._ddrf_pulse_block(QC_params, type='uc', tau_count=2*QC_params['DD_N']+1,
                                                     # rot_phase=accumRotZAng) )
            case 'crotx':
                if self.qubit == 3:
                    #pulse = self._axy8_pulse_block(QC_params['order_c'], QC_params['tau_c'], QC_params['f1_c'])
                    pulse = self._xy8_pulse_block( QC_params['order_c'], QC_params['tau_c']/2 )
                elif self.qubit == 4:
                    pulse = []
        return pulse


    def _xy8_pulse_block(self, order, tau_half):
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
        tauhalf_element = self._get_idle_element(length=tau_half, increment=0)
        pulse_block = []
        for n in range(1, order+1):
            rotAxes = ['x','y'] if n%4 in (1,2) else ['y','x']
            pulse_block += [tauhalf_element]
            pulse_block += [ MW_elements['MWpi'+rotAxes[0]] ]
            pulse_block += 2*[tauhalf_element]
            pulse_block += [ MW_elements['MWpi'+rotAxes[1]] ]
            pulse_block += [tauhalf_element]
        return pulse_block


    def _axy8_pulse_block(self, order, tau, f1):
        spacings = self._get_axy_spacing( fe_vals=[f1, 0, 0, 0] )
        # determine a scale factor for each tau
        tau_factors = np.zeros(6, dtype='float64')
        tau_factors[0] = spacings[0]
        tau_factors[1] = spacings[1] - spacings[0]
        tau_factors[2] = spacings[2] - spacings[1]
        tau_factors[3] = tau_factors[2]
        tau_factors[4] = tau_factors[1]
        tau_factors[5] = tau_factors[0]
        first_tau = self._get_idle_element(length=1*tau_factors[0] * 2*tau - (self.rabi_period / 4), increment=0)
        last_tau = self._get_idle_element(length=1*tau_factors[5] * 2*tau - (self.rabi_period / 4), increment=0)
        tau1_raw_element = self._get_idle_element(length=tau_factors[0] * 2*tau, increment=0)
        tau6_raw_element = self._get_idle_element(length=tau_factors[5] * 2*tau, increment=0)
        tau_elements = [
            self._get_idle_element(length=tau_factors[k] * 2*tau - (self.rabi_period / 2), increment=0) for k in range(len(tau_factors))
        ]

        mw_elements = {
            'pix_0':  self._get_mw_element(length=self.rabi_period/2,
                                                  increment=0,
                                                  amp=self.microwave_amplitude,
                                                  freq=self.microwave_frequency,
                                                  phase=0),
            'pix_30': self._get_mw_element(length=self.rabi_period/2,
                                                  increment=0,
                                                  amp=self.microwave_amplitude,
                                                  freq=self.microwave_frequency,
                                                  phase=30),
            'pix_90': self._get_mw_element(length=self.rabi_period/2,
                                                  increment=0,
                                                  amp=self.microwave_amplitude,
                                                  freq=self.microwave_frequency,
                                                  phase=90),
            'piy_0':  self._get_mw_element(length=self.rabi_period/2,
                                                  increment=0,
                                                  amp=self.microwave_amplitude,
                                                  freq=self.microwave_frequency,
                                                  phase=90),
            'piy_30': self._get_mw_element(length=self.rabi_period/2,
                                                  increment=0,
                                                  amp=self.microwave_amplitude,
                                                  freq=self.microwave_frequency,
                                                  phase=120),
            'piy_90': self._get_mw_element(length=self.rabi_period/2,
                                                  increment=0,
                                                  amp=self.microwave_amplitude,
                                                  freq=self.microwave_frequency,
                                                  phase=180)
        }

        rot_angles = ['30', '0', '90', '0', '30']
        pulse_block = []
        for n in range(1, order+1):
            rotAxes = ['x', 'y'] if n % 4 in (1, 2) else ['y', 'x']
            for _ in range(2):
                #rotAxes[0]
                for angle, tau_elem in zip(rot_angles, tau_elements):
                    pulse_block += [tau_elem]
                    pulse_block += [ mw_elements['pi' + rotAxes[0] + '_' + angle] ]
                pulse_block += [tau6_raw_element]
                #rotAxes[1]
                for angle, tau_elem in zip(rot_angles, reversed(tau_elements)):
                    pulse_block += [tau_elem]
                    pulse_block += [ mw_elements['pi' + rotAxes[1] + '_' + angle] ]
                pulse_block += [tau1_raw_element]
        pulse_block[0] = first_tau
        pulse_block[-1] = last_tau
        return pulse_block


    def _ddrf_pulse_block(self, QC_params, type, tau_count=1, rot_phase=0):
        match type:
            case 'c0':
                phaseOffsets = [0, 180]
            case 'c1':
                phaseOffsets = [180, 0]
            case 'uc':
                phaseOffsets = [0, 0]
        tau = QC_params['cyclesf'] * (1 / QC_params['RF_freq1']) + 1.0e-9
        cycles = ((2 * np.pi * QC_params['RF_freq1']) * (tau)) // (2 * np.pi)
        tau_pulse = (2 * np.pi * cycles) / (2 * np.pi * QC_params['RF_freq1'])
        tau_idle = (tau - tau_pulse) / 2
        phase = self._inst_phase(QC_params['RF_freq1'],
                                 QC_params['RF_freq0'],
                                 0.0,
                                 tau,
                                 0)
        #print(f'phase: {phase:.3f}')
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
        #for n in range(1, order + 1):
        for n in range(1, QC_params['DD_N']+1):
            if n != 1:
                del pulse_block[len(pulse_block) - 3:len(pulse_block)]
                numTau = 2
            else:
                numTau = 1
            #k=1,4,7,10
            RF_phase = np.mod( (tau_count-1)*phase + phaseOffsets[tau_count%2] + rot_phase, 360 )
            #print(f'tau_count: {tau_count}, RF_phase: {RF_phase:.3f}')
            tauidle_element = self._get_idle_element(length=numTau*tau_idle, increment=0)
            RFtau_element = self._get_rf_element(length=numTau*tau_pulse,
                                                 increment=0,
                                                 amp=QC_params['RF_amp1'],
                                                 freq=QC_params['RF_freq1'],
                                                 phase=RF_phase)
            pulse_block.append(tauidle_element)
            pulse_block.append(RFtau_element)
            pulse_block.append(tauidle_element)
            rotAx = 'x' if n%4 in (1,2) else 'y'
            pulse_block.append(MW_elements['MWpi'+rotAx])
            tau_count += 1
            #k=2,5,8,11
            RF_phase = np.mod( (tau_count-1)*phase + phaseOffsets[tau_count%2] + rot_phase, 360 )
            #print(f'tau_count: {tau_count}, RF_phase: {RF_phase:.3f}')
            tauidle_element = self._get_idle_element(length=2*tau_idle, increment=0)
            RFtau_element = self._get_rf_element(length=2*tau_pulse,
                                                 increment=0,
                                                 amp=QC_params['RF_amp1'],
                                                 freq=QC_params['RF_freq1'],
                                                 phase=RF_phase)
            pulse_block.append(tauidle_element)
            pulse_block.append(RFtau_element)
            pulse_block.append(tauidle_element)
            rotAx = 'y' if n%4 in (1,2) else 'x'
            pulse_block.append(MW_elements['MWpi'+rotAx])
            tau_count += 1
            #k=3,6,9,12
            RF_phase = np.mod( (tau_count - 1)*phase + phaseOffsets[tau_count%2] + rot_phase, 360 )
            #print(f'tau_count: {tau_count}, RF_phase: {RF_phase:.3f}')
            tauidle_element = self._get_idle_element(length=tau_idle, increment=0)
            RFtau_element = self._get_rf_element(length=tau_pulse,
                                                 increment=0,
                                                 amp=QC_params['RF_amp1'],
                                                 freq=QC_params['RF_freq1'],
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


    @staticmethod
    def _get_axy_spacing(fe_vals = [1.0, 0.0, 0.0, 0.0]):
        # define function to solve
        def kdd5even(x):
            return_val = np.zeros(4, dtype='float64')
            x4 = x[1] + x[3] - (x[0] + x[2]) + np.pi/2
            for k in range(1,5):
                return_val[k-1] = fe_vals[k-1] - 4/(k * np.pi) * (
                        np.sin(k*x[0]) + np.sin(k*x[2]) + np.sin(k*x4) - np.sin(k*x[1]) - np.sin(k*x[3])
                    )
            return return_val

        # Initial angles for solver
        x0 = np.array([0.1 * np.pi, 0.3 * np.pi, 0.6 * np.pi, 0.9 * np.pi], dtype='float64')
        # Solve for kdd5even(x) = 0
        solved_x = scipy.optimize.fsolve(kdd5even, x0)
        solved_x = np.append(solved_x, solved_x[1] + solved_x[3] - (solved_x[0] + solved_x[2]) + np.pi/2)
        return solved_x / (2*np.pi)



class QB1234ControlPredefinedGenerator(PredefinedGeneratorBase):
    """

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sgl = args[0]


    ################################################################################################
    #                             Generation methods for waveforms                                 #
    ################################################################################################


    def generate_QuantumCircuitQB12(self, name='quantumcircuitQB12',
                                    Initial_state=TQstates.State00, Readout_circ=TQReadoutCircs.P00,
                                    NV_Cpi_len=1.0e-6, NV_Cpi_amp=0.05, NV_Cpi_freq1=1.432e9,
                                    RF_freq0=5.1e6, RF_amp0=0.02,  RF_freq1=5.1e6, RF_amp1=0.02,
                                    cyclesf=9, DD_N=2, RF_pi=20.0e-6,
                                    gate_operations = "sx(0)[1], rz(90)[2], sx(0)[2]",
                                    num_of_points=50,
                                    laser_on=20.0e-9, laser_off=60.0e-9):
        """

        """
        for methodParams in self._sgl.generate_method_params.values():
            if methodParams['name'] == name:
                QCQB12_params = methodParams
                break

        gate_noop = xq1iGate(self._sgl, name='noop', param=0.0e-9)
        gate_c0q1x = xq1iGate(self._sgl, name="c0x", qubit=1)
        gate_c0q2x = xq1iGate(self._sgl, name="c0x", qubit=2)
        gate_c1q2x = xq1iGate(self._sgl, name="c1x", qubit=2)
        gate_q1sx = xq1iGate(self._sgl, name="sx", qubit=1)
        #gate_q2sx = xq1iGate(self._sgl, name="sx", qubit=2)

        gate_list = []
        if not gate_operations == "":
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
            case TQstates.State00.value:
                initialblock_list += gate_noop.get_pulse(QCQB12_params)
            case TQstates.State01.value:
                initialblock_list += gate_c0q2x.get_pulse(QCQB12_params)
            case TQstates.State10.value:
                initialblock_list += gate_q1sx.get_pulse(QCQB12_params) + gate_q1sx.get_pulse(QCQB12_params)
            case TQstates.State11.value:
                initialblock_list += gate_q1sx.get_pulse(QCQB12_params) + gate_q1sx.get_pulse(QCQB12_params) + gate_c1q2x.get_pulse(QCQB12_params)

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
        readout_blocks = {
            TQReadoutCircs.P00.value: gate_noop.get_pulse(QCQB12_params),
            TQReadoutCircs.P01.value: gate_c0q2x.get_pulse(QCQB12_params),
            TQReadoutCircs.P10.value: gate_c0q1x.get_pulse(QCQB12_params),
        }

        # combine blocks for init, operations and readout into one state tomography block (sequentially reading the population of each basis state)
        statetomo_block = PulseBlock(name=name)

        # ddrf_orderscan sequence for testing/debugging the _ddrf_pulse_block function
        # num_points = 30
        # for n in range(1, num_points+1):
        #     pulse_block = []
        #     # pulse_block = [self._get_idle_element(length=self.rabi_period/2, increment=0)]
        #     # pulse_block = gate_q1sx.get_pulse(QCQB12_params)
        #     # pulse_block += gate_q1sx.get_pulse(QCQB12_params)
        #     pulse_block += gate_q2sx.get_pulse(QCQB12_params, order=n)
        #     # pulse_block += [self._get_idle_element(length=self.rabi_period / 2, increment=0)]
        #     # pulse_block += gate_q1sx.get_pulse(QCQB12_params)
        #     # pulse_block += gate_q1sx.get_pulse(QCQB12_params)
        #     for pulse in pulse_block:
        #         statetomo_block.append(pulse)
        #     for laser_trig in laser_block:
        #         statetomo_block.append(laser_trig)
        #     statetomo_block.append( self._get_idle_element(length=self.laser_delay, increment=0) )
        #     statetomo_block.append( self._get_idle_element(length=self.wait_time, increment=0) )

        # readout of all 4 populations sequentially
        # for readout_block in readout_blocks.values():
        #     for pulse in initialblock_list:
        #         statetomo_block.append(pulse)
        #     for pulse in opersblock_list:
        #         statetomo_block.append(pulse)
        #     for pulse in readout_block:
        #         statetomo_block.append(pulse)
        #     for laser_trig in laser_block:
        #         statetomo_block.append(laser_trig)
        #     statetomo_block.append( self._get_idle_element(length=self.laser_delay, increment=0) )
        #     statetomo_block.append( self._get_idle_element(length=self.wait_time, increment=0) )

        # readout of the population of a single state Readout_state (population transfer to 00 and subsequent Rabi driving)
        if Readout_circ.value in (TQReadoutCircs.P00.value, TQReadoutCircs.P10.value):
            statetomo_block.append( self._get_idle_element(length=QCQB12_params['RF_pi'], increment=0) )
        for pulse in initialblock_list:
            statetomo_block.append(pulse)
        for pulse in opersblock_list:
            statetomo_block.append(pulse)
        for pulse in readout_blocks[Readout_circ.value]:
            statetomo_block.append(pulse)
        tau_step = QCQB12_params['NV_Cpi_len']/5
        statetomo_block.append(self._get_mw_element(length=0, increment=tau_step,
                                                    amp=QCQB12_params['NV_Cpi_amp'], freq=QCQB12_params['NV_Cpi_freq1'],
                                                    phase=accum_rotZAng[0])
                               )
        for laser_trig in laser_block:
            statetomo_block.append(laser_trig)
        statetomo_block.append( self._get_idle_element(length=self.laser_delay, increment=0) )
        statetomo_block.append( self._get_idle_element(length=self.wait_time, increment=0) )

        # Create block ensemble
        created_blocks = list()
        created_ensembles = list()
        created_sequences = list()
        created_blocks.append(statetomo_block)
        block_ensemble = PulseBlockEnsemble(name=name, rotating_frame=True)
        block_ensemble.append((statetomo_block.name, num_of_points-1))

        # Create and append sync trigger block if needed
        self._add_trigger(created_blocks=created_blocks, block_ensemble=block_ensemble)

        # get tau array for measurement ticks
        #tau_array =  np.arange(1,num_points+1)
        #tau_array =  np.arange(4)
        tau_array = np.arange(num_of_points) * tau_step
        # add metadata to invoke settings later on
        number_of_lasers = num_of_points
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


    def generate_QuantumCircuitQstQB13(self, name='quantumcircuitQstQB13',
                                    Initial_state=TQstates.State00, Readout_circ=TQQSTReadoutCircs.IX,
                                    f1_uc=1.0, tau_uc=0.8e-6, order_uc=8,
                                    f1_c=1.0, tau_c=0.8e-6, order_c=8,
                                    tau_z=0.8e-6, order_z = 8,
                                    gate_operations = "sx(0)[1], rz(90)[3], sx(0)[3]",
                                    num_of_points=50,
                                    laser_on=20.0e-9, laser_off=60.0e-9):
        """

        """
        for methodParams in self._sgl.generate_method_params.values():
            if methodParams['name'] == name:
                QCQSTQB13_params = methodParams
                break

        gate_p00 = xq1iGate(self._sgl, name='p00')
        gate_q1sx = xq1iGate(self._sgl, name="sx", qubit=1)
        gate_q1sy = xq1iGate(self._sgl, name="sy", qubit=1)
        gate_q3sx = xq1iGate(self._sgl, name="sx", qubit=3)
        gate_q3rzPihalf = xq1iGate(self._sgl, name="rz", qubit=3, param=90)
        gate_crotq3x = xq1iGate(self._sgl, name="crotx", qubit=3)

        # gates for initialization
        qb3_init_block = [gate_q1sy, gate_crotq3x, gate_q1sx, gate_q3rzPihalf, gate_crotq3x, gate_p00]
        init_gates = {
            TQstates.State00.value: qb3_init_block,
            TQstates.State01.value: qb3_init_block + 2*[gate_crotq3x],
            TQstates.State10.value: qb3_init_block + 2*[gate_q1sx],
            TQstates.State11.value: qb3_init_block + 2*[gate_crotq3x] + 2*[gate_q1sx],
        }

        gate_list = []
        if not gate_operations == "":
            regex_pattern = re.compile("([^(]*)\(([0-9\.eE\+\-]*)\)\[([0-9]*)\]")
            for gate in gate_operations.replace(" ", "").split(","):
                regex_match = regex_pattern.match(gate)
                gate_name = regex_match.group(1)
                param = float(regex_match.group(2))
                qubit = int(regex_match.group(3))
                gate_list.append( xq1iGate(self._sgl, name=gate_name, qubit=qubit, param=param) )

        # gates for readout
        readout_gates = {
            TQQSTReadoutCircs.IX.value: [gate_q3rzPihalf, gate_q3sx, gate_crotq3x, gate_q3rzPihalf, gate_q1sx, gate_crotq3x, gate_q1sy],
            TQQSTReadoutCircs.IY.value: [gate_q3sx, gate_crotq3x, gate_q3rzPihalf, gate_q1sx, gate_crotq3x, gate_q1sy],
            TQQSTReadoutCircs.IZ.value: [gate_crotq3x, gate_q3rzPihalf, gate_q1sx, gate_crotq3x, gate_q1sy],
            TQQSTReadoutCircs.XI.value: [gate_q1sy],
            TQQSTReadoutCircs.XX.value: [gate_crotq3x, gate_q1sx],
            TQQSTReadoutCircs.XY.value: [gate_q3rzPihalf, gate_crotq3x, gate_q1sx],
            TQQSTReadoutCircs.XZ.value: [gate_q3sx, gate_q3rzPihalf, gate_crotq3x, gate_q1sx],
            TQQSTReadoutCircs.YI.value: [gate_q1sx],
            TQQSTReadoutCircs.YX.value: [gate_crotq3x, gate_q1sy],
            TQQSTReadoutCircs.YY.value: [gate_q3rzPihalf, gate_crotq3x, gate_q1sy],
            TQQSTReadoutCircs.YZ.value: [gate_q3sx, gate_q3rzPihalf, gate_crotq3x, gate_q1sy],
            TQQSTReadoutCircs.ZI.value: [],
            TQQSTReadoutCircs.ZX.value: [gate_q1sy, gate_crotq3x, gate_q1sx],
            TQQSTReadoutCircs.ZY.value: [gate_q3rzPihalf, gate_q1sy, gate_crotq3x, gate_q1sx],
            TQQSTReadoutCircs.ZZ.value: [gate_q3sx, gate_q3rzPihalf, gate_q1sy, gate_crotq3x, gate_q1sx]
        }

        # create pulse blocks from gates
        opersblock_list=[]
        accum_rotZAng = [0, 0, 0]
        for gate in init_gates[Initial_state.value] + gate_list + readout_gates[Readout_circ.value]:
            if gate.name == 'rz':
                accum_rotZAng[gate.qubit-1] += gate.param
            opersblock_list += gate.get_pulse(QCQSTQB13_params, accum_rotZAng[gate.qubit-1])

        # combine blocks for init, operations and readout into one state tomography block (sequentially reading the population of each basis state)
        statetomo_block = PulseBlock(name=name)

        # readout of the fluorescence rate corresponding to a single Readout_circ (population transfer to 00 and subsequent Rabi driving)
        for pulse in opersblock_list:
            statetomo_block.append(pulse)
        tau_step = self.rabi_period / 13
        statetomo_block.append(self._get_mw_element(length=0, increment=tau_step,
                                                    amp=self.microwave_amplitude, freq=self.microwave_frequency,
                                                    phase=accum_rotZAng[0])
                               )
        for laser_elem in gate_p00.get_pulse(QCQSTQB13_params):
            statetomo_block.append(laser_elem)

        # Create block ensemble
        created_blocks = list()
        created_ensembles = list()
        created_sequences = list()
        created_blocks.append(statetomo_block)
        block_ensemble = PulseBlockEnsemble(name=name, rotating_frame=True)
        block_ensemble.append((statetomo_block.name, num_of_points-1))

        # Create and append sync trigger block if needed
        self._add_trigger(created_blocks=created_blocks, block_ensemble=block_ensemble)

        # get tau array for measurement ticks
        tau_array = np.arange(num_of_points) * tau_step
        # add metadata to invoke settings later on
        number_of_lasers = 2*num_of_points
        block_ensemble.measurement_information['alternating'] = False
        # ignore the laser pulses used for resetting the NV after QB3 init
        block_ensemble.measurement_information['laser_ignore_list'] = [_ for _ in range(0, number_of_lasers, 2)]
        block_ensemble.measurement_information['controlled_variable'] = tau_array
        block_ensemble.measurement_information['units'] = ('s', '')
        block_ensemble.measurement_information['number_of_lasers'] = number_of_lasers
        block_ensemble.measurement_information['counting_length'] = self._get_ensemble_count_length(
                            ensemble=block_ensemble, created_blocks=created_blocks)

        # append ensemble to created ensembles
        created_ensembles.append(block_ensemble)
        return created_blocks, created_ensembles, created_sequences


    def generate_QuantumCircuitQB123(self, name='quantumcircuitQB123',
                                    Initial_state=ThrQstates.State000, Readout_circ=ThrQReadoutCircs.IZI1,
                                    NV_Cpi_len=1.0e-6, NV_Cpi_amp=0.05, NV_Cpi_freq1=1.432e9,
                                    RF_freq0=5.1e6, RF_amp0=0.02,  RF_freq1=5.1e6, RF_amp1=0.02,
                                    cyclesf=9, DD_N=2, RF_pi=20.0e-6,
                                    f1_uc=1.0, tau_uc=0.8e-6, order_uc=8,
                                    f1_c=1.0, tau_c=0.8e-6, order_c=8,
                                    tau_z=0.8e-6, order_z = 8,
                                    gate_operations = "sx(0)[1], rz(90)[2], sx(0)[2], sx(0)[3]",
                                    num_of_points=50,
                                    laser_on=20.0e-9, laser_off=60.0e-9):
        """

        """
        for methodParams in self._sgl.generate_method_params.values():
            if methodParams['name'] == name:
                QCQB123_params = methodParams
                break

        gate_noop = xq1iGate(self._sgl, name='noop', param=0.0e-9)
        gate_p00 = xq1iGate(self._sgl, name='p00')
        gate_c0q1x = xq1iGate(self._sgl, name="c0x", qubit=1)
        gate_c0q2x = xq1iGate(self._sgl, name="c0x", qubit=2)
        gate_c1q2x = xq1iGate(self._sgl, name="c1x", qubit=2)
        gate_q1sx = xq1iGate(self._sgl, name="sx", qubit=1)
        gate_q1sy = xq1iGate(self._sgl, name="sy", qubit=1)
        gate_q3sx = xq1iGate(self._sgl, name="sx", qubit=3)
        gate_q3rzPihalf = xq1iGate(self._sgl, name="rz", qubit=3, param=90)
        gate_crotq3x = xq1iGate(self._sgl, name="crotx", qubit=3)

        # gates for initialization
        init_gates = {
            ThrQstates.State000.value: [gate_q1sy, gate_crotq3x, gate_q1sx, gate_q3rzPihalf, gate_crotq3x, gate_p00],
            ThrQstates.State001.value: [gate_q1sx, gate_q1sx, gate_q1sy, gate_crotq3x, gate_q1sx, gate_q3rzPihalf, gate_crotq3x, gate_p00],
            ThrQstates.State010.value: [gate_q1sy, gate_crotq3x, gate_q1sx, gate_q3rzPihalf, gate_crotq3x, gate_p00, gate_c0q2x],
            ThrQstates.State011.value: [gate_q1sx, gate_q1sx, gate_q1sy, gate_crotq3x, gate_q1sx, gate_q3rzPihalf, gate_crotq3x, gate_p00, gate_c0q2x],
            ThrQstates.State100.value: [gate_q1sy, gate_crotq3x, gate_q1sx, gate_q3rzPihalf, gate_crotq3x, gate_p00, gate_q1sx, gate_q1sx],
            ThrQstates.State101.value: [gate_q1sx, gate_q1sx, gate_q1sy, gate_crotq3x, gate_q1sx, gate_q3rzPihalf, gate_crotq3x, gate_p00, gate_q1sx, gate_q1sx],
            ThrQstates.State110.value: [gate_q1sy, gate_crotq3x, gate_q1sx, gate_q3rzPihalf, gate_crotq3x, gate_p00, gate_q1sx, gate_q1sx, gate_c1q2x],
            ThrQstates.State111.value: [gate_q1sx, gate_q1sx, gate_q1sy, gate_crotq3x, gate_q1sx, gate_q3rzPihalf, gate_crotq3x, gate_p00, gate_q1sx, gate_q1sx, gate_c1q2x],
        }

        # gates in circuit
        gate_list = []
        if not gate_operations == "":
            regex_pattern = re.compile("([^(]*)\(([0-9\.eE\+\-]*)\)\[([0-9]*)\]")
            for gate in gate_operations.replace(" ", "").split(","):
                regex_match = regex_pattern.match(gate)
                gate_name = regex_match.group(1)
                param = float(regex_match.group(2))
                qubit = int(regex_match.group(3))
                gate_list.append( xq1iGate(self._sgl, name=gate_name, qubit=qubit, param=param) )

        # gates for readout
        readout_gates = {
            ThrQReadoutCircs.IIZ1.value: [gate_crotq3x, gate_q1sx, gate_q3rzPihalf, gate_crotq3x, gate_q1sy],
            ThrQReadoutCircs.IIZ2.value: [gate_crotq3x, gate_q1sx, gate_q3rzPihalf, gate_crotq3x, gate_q1sy, gate_c0q2x],
            ThrQReadoutCircs.IZI1.value: [],
            ThrQReadoutCircs.IZI2.value: [gate_c0q1x],
            ThrQReadoutCircs.IZZ2.value: [gate_crotq3x, gate_q1sx, gate_q3rzPihalf, gate_crotq3x, gate_q1sy, gate_c1q2x, gate_c0q1x],
            ThrQReadoutCircs.ZII2.value: [gate_c0q2x],
            ThrQReadoutCircs.ZIZ1.value: [gate_q3sx, gate_q3rzPihalf, gate_q1sy, gate_crotq3x, gate_q1sx],
            ThrQReadoutCircs.ZIZ2.value: [gate_q3sx, gate_q3rzPihalf, gate_q1sy, gate_crotq3x, gate_q1sx, gate_c0q2x],
            ThrQReadoutCircs.ZZI2.value: [gate_c1q2x, gate_c0q1x],
            ThrQReadoutCircs.ZZZ2.value: [gate_q3sx, gate_q3rzPihalf, gate_q1sy, gate_crotq3x, gate_q1sx, gate_c1q2x, gate_c0q1x]
        }

        # create pulse blocks from gates
        opersblock_list=[]
        accum_rotZAng = [0, 0, 0]
        for gate in init_gates[Initial_state.value] + gate_list + readout_gates[Readout_circ.value]:
            if gate.name == 'rz':
                accum_rotZAng[gate.qubit-1] += gate.param
            opersblock_list += gate.get_pulse(QCQB123_params, accum_rotZAng[gate.qubit-1])

        # combine blocks for init, operations and readout into one state tomography block (sequentially reading the population of each basis state)
        statetomo_block = PulseBlock(name=name)

        # readout of the fluorescence rate corresponding to a single Readout_circ (population transfer to 00 and subsequent Rabi driving)
        #TODO: initial padding to ensure the same (or similar) overall sequence length for all readout circuits
        #if Readout_circ.value in (TQstates.State00.value, TQstates.State10.value):
        #    statetomo_block.append( self._get_idle_element(length=QCQB123_params['RF_pi'], increment=0) )
        for pulse in opersblock_list:
            statetomo_block.append(pulse)
        tau_step = QCQB123_params['NV_Cpi_len']/5
        statetomo_block.append(self._get_mw_element(length=0, increment=tau_step,
                                                    amp=QCQB123_params['NV_Cpi_amp'], freq=QCQB123_params['NV_Cpi_freq1'],
                                                    phase=accum_rotZAng[0])
                               )
        for laser_elem in gate_p00.get_pulse(QCQB123_params):
            statetomo_block.append(laser_elem)

        # Create block ensemble
        created_blocks = list()
        created_ensembles = list()
        created_sequences = list()
        created_blocks.append(statetomo_block)
        block_ensemble = PulseBlockEnsemble(name=name, rotating_frame=True)
        block_ensemble.append((statetomo_block.name, num_of_points-1))

        # Create and append sync trigger block if needed
        self._add_trigger(created_blocks=created_blocks, block_ensemble=block_ensemble)

        # get tau array for measurement ticks
        tau_array = np.arange(num_of_points) * tau_step
        # add metadata to invoke settings later on
        number_of_lasers = 2*num_of_points
        block_ensemble.measurement_information['alternating'] = False
        # ignore the laser pulses used for resetting the NV after QB3 init
        block_ensemble.measurement_information['laser_ignore_list'] = [_ for _ in range(0, number_of_lasers, 2)]
        block_ensemble.measurement_information['controlled_variable'] = tau_array
        block_ensemble.measurement_information['units'] = ('s', '')
        block_ensemble.measurement_information['number_of_lasers'] = number_of_lasers
        block_ensemble.measurement_information['counting_length'] = self._get_ensemble_count_length(
                            ensemble=block_ensemble, created_blocks=created_blocks)

        # append ensemble to created ensembles
        created_ensembles.append(block_ensemble)
        return created_blocks, created_ensembles, created_sequences


    def generate_QuantumCircuitQB1234(self, name='quantumcircuitQB1234',
                                    Initial_state=FourQstates.State0000, Readout_circ=FourQReadoutCircs.IZII1,
                                    NV_Cpi_len=1.0e-6, NV_Cpi_amp=0.05, NV_Cpi_freq1=1.432e9,
                                    RF_freq0=5.1e6, RF_amp0=0.02,  RF_freq1=5.1e6, RF_amp1=0.02,
                                    cyclesf=9, DD_N=2, RF_pi=20.0e-6,
                                    f1_uc=1.0, tau_uc=0.8e-6, order_uc=8,
                                    f1_c=1.0, tau_c=0.8e-6, order_c=8,
                                    tau_z=0.8e-6, order_z = 8,
                                    gate_operations = "sx(0)[1], rz(90)[2], sx(0)[2], sx(0)[3], sx(0)[4]",
                                    num_of_points=50,
                                    laser_on=20.0e-9, laser_off=60.0e-9):
        """

        """
        for methodParams in self._sgl.generate_method_params.values():
            if methodParams['name'] == name:
                QCQB1234_params = methodParams
                break

        gate_noop = xq1iGate(self._sgl, name='noop', param=0.0e-9)
        gate_p00 = xq1iGate(self._sgl, name='p00')
        gate_c0q1x = xq1iGate(self._sgl, name="c0x", qubit=1)
        gate_c0q2x = xq1iGate(self._sgl, name="c0x", qubit=2)
        gate_c1q2x = xq1iGate(self._sgl, name="c1x", qubit=2)
        gate_q1sx = xq1iGate(self._sgl, name="sx", qubit=1)
        gate_q1sy = xq1iGate(self._sgl, name="sy", qubit=1)
        gate_q3sx = xq1iGate(self._sgl, name="sx", qubit=3)
        gate_q3rzPihalf = xq1iGate(self._sgl, name="rz", qubit=3, param=90)
        gate_crotq3x = xq1iGate(self._sgl, name="crotx", qubit=3)
        gate_q4sx = xq1iGate(self._sgl, name="sx", qubit=4)
        gate_q4rzPihalf = xq1iGate(self._sgl, name="rz", qubit=4, param=90)
        gate_crotq4x = xq1iGate(self._sgl, name="crotx", qubit=4)

        # gates for initialization
        qb3_init_block = [gate_q1sy, gate_crotq3x, gate_q1sx, gate_q3rzPihalf, gate_crotq3x, gate_p00]
        qb4_init_block = [gate_q1sy, gate_crotq4x, gate_q1sx, gate_q4rzPihalf, gate_crotq4x, gate_p00]
        init_gates = {
            FourQstates.State0000.value: qb3_init_block + qb4_init_block,
            FourQstates.State0001.value: qb3_init_block + 2*[gate_q1sx] + qb4_init_block,
            FourQstates.State0010.value: 2*[gate_q1sx] + qb3_init_block + qb4_init_block,
            FourQstates.State0011.value: 2*[gate_q1sx] + qb3_init_block + 2*[gate_q1sx] + qb4_init_block,
            FourQstates.State0100.value: qb3_init_block + qb4_init_block + [gate_c0q2x],
            FourQstates.State0101.value: qb3_init_block + 2*[gate_q1sx] + qb4_init_block + [gate_c0q2x],
            FourQstates.State0110.value: 2*[gate_q1sx] + qb3_init_block + qb4_init_block + [gate_c0q2x],
            FourQstates.State0111.value: 2*[gate_q1sx] + qb3_init_block + 2*[gate_q1sx] + qb4_init_block + [gate_c0q2x],
            FourQstates.State1000.value: qb3_init_block + qb4_init_block + 2*[gate_q1sx],
            FourQstates.State1001.value: qb3_init_block + 2*[gate_q1sx] + qb4_init_block + 2*[gate_q1sx],
            FourQstates.State1010.value: 2*[gate_q1sx] + qb3_init_block + qb4_init_block + 2*[gate_q1sx],
            FourQstates.State1011.value: 2*[gate_q1sx] + qb3_init_block + 2*[gate_q1sx] + qb4_init_block + 2*[gate_q1sx],
            FourQstates.State1100.value: qb3_init_block + qb4_init_block + [gate_c0q2x] + 2*[gate_q1sx],
            FourQstates.State1101.value: qb3_init_block + 2*[gate_q1sx] + qb4_init_block + [gate_c0q2x] + 2*[gate_q1sx],
            FourQstates.State1110.value: 2*[gate_q1sx] + qb3_init_block + qb4_init_block + [gate_c0q2x] + 2*[gate_q1sx],
            FourQstates.State1111.value: 2*[gate_q1sx] + qb3_init_block + 2*[gate_q1sx] + qb4_init_block + [gate_c0q2x] + 2*[gate_q1sx]
        }

        # gates in circuit
        gate_list = []
        if not gate_operations == "":
            regex_pattern = re.compile("([^(]*)\(([0-9\.eE\+\-]*)\)\[([0-9]*)\]")
            for gate in gate_operations.replace(" ", "").split(","):
                regex_match = regex_pattern.match(gate)
                gate_name = regex_match.group(1)
                param = float(regex_match.group(2))
                qubit = int(regex_match.group(3))
                gate_list.append( xq1iGate(self._sgl, name=gate_name, qubit=qubit, param=param) )

        # gates for readout
        readout_gates = {
            FourQReadoutCircs.IIZI1.value: [gate_crotq3x, gate_q1sx, gate_q3rzPihalf, gate_crotq3x, gate_q1sy],
            FourQReadoutCircs.IIZI2.value: [gate_crotq3x, gate_q1sx, gate_q3rzPihalf, gate_crotq3x, gate_q1sy, gate_c0q2x],
            FourQReadoutCircs.IZII1.value: [],
            FourQReadoutCircs.IZII2.value: [gate_c0q1x],
            FourQReadoutCircs.IZZI2.value: [gate_crotq3x, gate_q1sx, gate_q3rzPihalf, gate_crotq3x, gate_q1sy, gate_c1q2x, gate_c0q1x],
            FourQReadoutCircs.ZIII2.value: [gate_c0q2x],
            FourQReadoutCircs.ZIZI1.value: [gate_q3sx, gate_q3rzPihalf, gate_q1sy, gate_crotq3x, gate_q1sx],
            FourQReadoutCircs.ZIZI2.value: [gate_q3sx, gate_q3rzPihalf, gate_q1sy, gate_crotq3x, gate_q1sx, gate_c0q2x],
            FourQReadoutCircs.ZZII2.value: [gate_c1q2x, gate_c0q1x],
            FourQReadoutCircs.ZZZI2.value: [gate_q3sx, gate_q3rzPihalf, gate_q1sy, gate_crotq3x, gate_q1sx, gate_c1q2x, gate_c0q1x],
            FourQReadoutCircs.IIIZ1.value: [gate_crotq4x, gate_q1sx, gate_q4rzPihalf, gate_crotq4x, gate_q1sy],
            FourQReadoutCircs.IIIZ2.value: [gate_crotq4x, gate_q1sx, gate_q4rzPihalf, gate_crotq4x, gate_q1sy, gate_c0q2x],
            FourQReadoutCircs.IIZZ1.value: [gate_crotq3x, gate_q1sx, gate_q3rzPihalf, gate_crotq3x, gate_q1sy, gate_q1sy, gate_q4sx, gate_q4rzPihalf, gate_crotq4x, gate_q1sx],
            FourQReadoutCircs.IIZZ2.value: [gate_crotq3x, gate_q1sx, gate_q3rzPihalf, gate_crotq3x, gate_q1sy, gate_q1sy, gate_q4sx, gate_q4rzPihalf, gate_crotq4x, gate_q1sx, gate_c0q2x],
            FourQReadoutCircs.IZIZ2.value: [gate_crotq4x, gate_q1sx, gate_q4rzPihalf, gate_crotq4x, gate_q1sy, gate_c1q2x, gate_c0q1x],
            FourQReadoutCircs.IZZZ2.value: [gate_crotq3x, gate_q1sx, gate_q3rzPihalf, gate_crotq3x, gate_q1sy, gate_q1sy, gate_q4sx, gate_q4rzPihalf, gate_crotq4x, gate_q1sx, gate_c1q2x, gate_c0q1x],
            FourQReadoutCircs.ZIIZ1.value: [gate_q4sx, gate_q4rzPihalf, gate_q1sy, gate_crotq4x, gate_q1sx],
            FourQReadoutCircs.ZIIZ2.value: [gate_q4sx, gate_q4rzPihalf, gate_q1sy, gate_crotq4x, gate_q1sx, gate_c0q2x],
            FourQReadoutCircs.ZIZZ1.value: [gate_q3sx, gate_q3rzPihalf, gate_q1sy, gate_crotq3x, gate_q1sx, gate_q4sx, gate_q4rzPihalf, gate_q1sy, gate_crotq4x, gate_q1sx],
            FourQReadoutCircs.ZIZZ2.value: [gate_q3sx, gate_q3rzPihalf, gate_q1sy, gate_crotq3x, gate_q1sx, gate_q4sx, gate_q4rzPihalf, gate_q1sy, gate_crotq4x, gate_q1sx, gate_c0q2x],
            FourQReadoutCircs.ZZIZ2.value: [gate_q4sx, gate_q4rzPihalf, gate_q1sy, gate_crotq4x, gate_q1sx, gate_c1q2x, gate_c0q1x],
            FourQReadoutCircs.ZZZZ2.value: [gate_q3sx, gate_q3rzPihalf, gate_q1sy, gate_crotq3x, gate_q1sx, gate_q4sx, gate_q4rzPihalf, gate_q1sy, gate_crotq4x, gate_q1sx, gate_c1q2x, gate_c0q1x],
        }

        # create pulse blocks from gates
        opersblock_list=[]
        accum_rotZAng = [0, 0, 0, 0]
        for gate in init_gates[Initial_state.value] + gate_list + readout_gates[Readout_circ.value]:
            if gate.name == 'rz':
                accum_rotZAng[gate.qubit-1] += gate.param
            opersblock_list += gate.get_pulse(QCQB1234_params, accum_rotZAng[gate.qubit-1])

        # combine blocks for init, operations and readout into one state tomography block (sequentially reading the population of each basis state)
        statetomo_block = PulseBlock(name=name)

        # readout of the fluorescence rate corresponding to a single Readout_circ (population transfer to 00 and subsequent Rabi driving)
        #TODO: initial padding to ensure the same (or similar) overall sequence length for all readout circuits
        #if Readout_circ.value in (TQstates.State00.value, TQstates.State10.value):
        #    statetomo_block.append( self._get_idle_element(length=QCQB123_params['RF_pi'], increment=0) )
        for pulse in opersblock_list:
            statetomo_block.append(pulse)
        tau_step = QCQB1234_params['NV_Cpi_len']/5
        statetomo_block.append(self._get_mw_element(length=0, increment=tau_step,
                                                    amp=QCQB1234_params['NV_Cpi_amp'], freq=QCQB1234_params['NV_Cpi_freq1'],
                                                    phase=accum_rotZAng[0])
                               )
        for laser_elem in gate_p00.get_pulse(QCQB1234_params):
            statetomo_block.append(laser_elem)

        # Create block ensemble
        created_blocks = list()
        created_ensembles = list()
        created_sequences = list()
        created_blocks.append(statetomo_block)
        block_ensemble = PulseBlockEnsemble(name=name, rotating_frame=True)
        block_ensemble.append((statetomo_block.name, num_of_points-1))

        # Create and append sync trigger block if needed
        self._add_trigger(created_blocks=created_blocks, block_ensemble=block_ensemble)

        # get tau array for measurement ticks
        tau_array = np.arange(num_of_points) * tau_step
        # add metadata to invoke settings later on
        number_of_lasers = 2*num_of_points
        block_ensemble.measurement_information['alternating'] = False
        # ignore the laser pulses used for resetting the NV after QB3 init
        block_ensemble.measurement_information['laser_ignore_list'] = [_ for _ in range(0, number_of_lasers, 2)]
        block_ensemble.measurement_information['controlled_variable'] = tau_array
        block_ensemble.measurement_information['units'] = ('s', '')
        block_ensemble.measurement_information['number_of_lasers'] = number_of_lasers
        block_ensemble.measurement_information['counting_length'] = self._get_ensemble_count_length(
                            ensemble=block_ensemble, created_blocks=created_blocks)

        # append ensemble to created ensembles
        created_ensembles.append(block_ensemble)
        return created_blocks, created_ensembles, created_sequences
