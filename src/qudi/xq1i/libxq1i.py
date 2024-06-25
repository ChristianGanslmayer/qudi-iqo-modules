import time
import datetime
import os
import json
from collections import OrderedDict
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from qutip import *
from qutip_qip.circuit import QubitCircuit, Gate
from qudi.util.network import netobtain
from qudi.logic.pulsed.predefined_generate_methods.qb1234_control_methods import xq1iGate
from qudi.logic.pulsed.predefined_generate_methods.qb1234_control_methods import TQstates, TQReadoutCircs, TQQSTReadoutCircs, ThrQstates, ThrQReadoutCircs, FourQstates, FourQReadoutCircs


# def covertNetrefToNumpyArray(data):
#     return np.array([float(_) for _ in data])

def likelihoodFunc(tParams, *nuMeasured):
    nuRhoPhys = measExpVal( rhoP(tParams) )
    costFuncVal = np.sum( (nuRhoPhys - nuMeasured)**2 / (2*nuRhoPhys) )
    return costFuncVal

def measExpVal(rho):
    expVal = np.zeros(16)
    basisSetOneQb = [qeye(2), sigmax(), sigmay(), sigmaz()]
    basisSetTwoQb = [tensor(op1, op2) for op1 in basisSetOneQb for op2 in basisSetOneQb]
    for i,op in enumerate(basisSetTwoQb):
        expVal[i] = 0.5*(expect(op, rho) + 1)
    return expVal

def rhoP(tParams):
    tMat = qdiags(
        [ tParams[0:4],
          [tParams[4]+1j*tParams[5], tParams[6]+1j*tParams[7], tParams[8]+1j*tParams[9]],
          [tParams[10]+1j*tParams[11], tParams[12]+1j*tParams[13]],
          [tParams[14]+1j*tParams[15]]
        ],
        [0,-1,-2,-3],
        dims=[[2,2], [2,2]]
    )
    tMatProd = tMat.dag()*tMat
    if isNonZero(tMatProd.tr()):
        return tMatProd / tMatProd.tr()
    else:
        return tMatProd

def tParamsOfRho(rho):
    tParams = np.zeros(16)
    det = np.linalg.det(rho.full())
    minFirst11 = matMinor(rho,0,0)
    minSecond1122 = matMinor(rho,[0,1],[0,1])
    #print(det, minFirst11, minSecond1122)
    # t_1
    if isNonZero(minFirst11):
        tParams[0] = np.sqrt( np.abs(det/minFirst11) )
    # t_2
    if isNonZero(minSecond1122):
        tParams[1] = np.sqrt( np.abs(minFirst11/minSecond1122) )
    # t_3
    if isNonZero(rho[3,3]):
        tParams[2] = np.sqrt( np.abs(minSecond1122/rho[3,3]) )
    # t_4
    tParams[3] = np.sqrt( np.abs(rho[3,3]) )
    # t_5, t_6
    if isNonZero(minFirst11*minSecond1122):
        tmp = matMinor(rho,0,1)/np.sqrt(minFirst11*minSecond1122)
        tParams[4] = np.real(tmp)
        tParams[5] = np.imag(tmp)
    # t_7, t_8
    if isNonZero(rho[3,3]*minSecond1122):
        tmp = matMinor(rho,[0,1],[0,2])/np.sqrt(rho[3,3]*minSecond1122)
        tParams[6] = np.real(tmp)
        tParams[7] = np.imag(tmp)
    # t_9, t_10
    if isNonZero(rho[3,3]):
        tParams[8] = np.real( rho[3,2]/np.sqrt(rho[3,3]) )
        tParams[9] = np.imag( rho[3,2]/np.sqrt(rho[3,3]) )
    # t_11, t_12
    if isNonZero(rho[3,3]*minSecond1122):
        tmp = matMinor(rho,[0,1],[1,2])/np.sqrt(rho[3,3]*minSecond1122)
        tParams[10] = np.real(tmp)
        tParams[11] = np.imag(tmp)
    # t_13, t_14
    if isNonZero(rho[3,3]):
        tParams[12] = np.real( rho[3,1]/np.sqrt(rho[3,3]) )
        tParams[13] = np.imag( rho[3,1]/np.sqrt(rho[3,3]) )
    # t_15, t_16
    if isNonZero(rho[3,3]):
        tParams[14] = np.real( rho[3,0]/np.sqrt(rho[3,3]) )
        tParams[15] = np.imag( rho[3,0]/np.sqrt(rho[3,3]) )
    return tParams

def likelihoodFuncPopul(tParams, *nuMeasured):
    nuPopulPhys = measExpValPopul( populP(tParams) )
    #costFuncVal = np.sum( (nuPopulPhys - nuMeasured)**2 / (2*nuPopulPhys) )
    costFuncVal = np.sum((nuPopulPhys - nuMeasured) ** 2)
    return costFuncVal

def measExpValPopul(populations):
    basisOne = [ np.array([1.0, 1.0]), np.array([1.0, -1.0]) ]
    basisNQ = basisOne
    for n in range(1, int( np.log2(len(populations)) )):
        basisTemp = basisNQ
        basisNQ = []
        for op1 in basisTemp:
            for op2 in basisOne:
                basisNQ.append(np.kron(op1, op2))
    expVal = np.zeros_like(populations)
    for i,op in enumerate(basisNQ):
        expVal[i] = 0.5 * (np.dot(op,populations) + 1)
    return expVal

def populP(tParams):
    normFact = np.sum(tParams**2)
    if isNonZero(normFact):
        return tParams**2 / normFact
    else:
        return np.ones_like(tParams) / len(tParams)

def tParamsOfPopul(populations):
    tParams = [np.sqrt(p) if p > 0 else 0 for p in populations]
    return tParams

def isNonZero(val):
    return abs(val) > 1e-15

def matMinor(mat, rowInds, colInds):
    return( np.linalg.det( np.delete(np.delete(mat.full(), rowInds, axis=0), colInds, axis=1) ) )

def gate_CeROTn():
    qcGateStash = QubitCircuit(N=1)
    qcGateStash.add_gate("RX", targets=0, arg_value=3*np.pi/2)
    qcGateStash.add_gate("RX", targets=0, arg_value=np.pi/2)
    qcGateStash.add_gate("RX", targets=0, arg_value=-np.pi / 2)
    # controlled ROT gate, DD
    return ( tensor(basis(2,0)*basis(2,0).dag(), qcGateStash.propagators()[1]) +
             tensor(basis(2,1)*basis(2,1).dag(), qcGateStash.propagators()[2]) )


class xq1i:
    POI_name = 'Qubit_XQ1i'
    calibParamsFilePrefix = os.path.join('./', 'calib_params', 'calib_params_')
    measResFilePrefix = os.path.join('./', 'measurement_results', 'run_')
    microwave_amplitude_LowPower = 0.003
    microwave_amplitude_HighPower = 0.05
    nucrabi_RFfreq0_amp = 0.02
    nucrabi_RFfreq1_amp = 0.02

    def __init__(self, pulsed_master_logic, pulsed_measurement_logic, sequence_generator_logic):
        self.pulsed_master_logic = pulsed_master_logic
        self.pulsed_measurement_logic = pulsed_measurement_logic
        self.sequence_generator_logic = sequence_generator_logic

        self.calib_params = OrderedDict()
        self.calib_params['res_freq'] = 1.4472e9
        self.calib_params['RF_freq0'] = 5.0962e6
        self.calib_params['RF_freq1'] = 2.9256e6
        self.calib_params['rabi_period_LowPower'] = 2.4e-6
        self.calib_params['rabi_period_HighPower'] = 170e-09
        self.calib_params['rabi_offset'] = 1.181
        self.calib_params['rabi_amplitude'] = 0.139
        self.calib_params['nucrabi_RFfreq0_period'] = 160.46e-6
        self.calib_params['nucrabi_RFfreq1_period'] = 160.53e-6
        self.loadCalibParams()

        self.generate_params = OrderedDict()
        self.generate_params['laser_channel'] = 'd_ch1'
        self.generate_params['sync_channel'] = 'd_ch2'
        self.generate_params['laser_length'] = 3.2e-06
        self.generate_params['laser_delay'] = 1.0e-9
        self.generate_params['wait_time'] = 70.5e-06
        self.generate_params['microwave_frequency'] = self.calib_params['res_freq']
        self.generate_params['microwave_amplitude'] = self.microwave_amplitude_HighPower
        self.generate_params['rabi_period'] = 170.50e-09
        self.pulsed_master_logic.set_generation_parameters(self.generate_params)

        self.rabi_params = self.pulsed_master_logic.generate_method_params['rabi']
        self.rabi_params['name'] = 'rabi'
        self.rabi_params['tau_start'] = 2.0e-9
        self.rabi_params['tau_step'] = 2.0e-9
        self.rabi_params['laser_on'] = 20.0e-9
        self.rabi_params['laser_off'] = 60.0e-9
        self.rabi_params['num_of_points'] = 50
        self.rabi_params['delay_time'] = 0.0e-6
        self.rabi_sweeps = 60000

        self.pulsedODMR_params = self.pulsed_master_logic.generate_method_params['pulsedodmr']
        self.pulsedODMR_params['name'] = 'pulsedODMR'
        self.pulsedODMR_params['freq_start'] = 1340.0e6
        self.pulsedODMR_params['freq_step'] = 0.50e6
        self.pulsedODMR_params['laser_on'] = 20.0e-9
        self.pulsedODMR_params['laser_off'] = 60.0e-9
        self.pulsedODMR_params['RF_pi'] = False
        self.pulsedODMR_params['RF_freq'] = 2.94e6
        self.pulsedODMR_params['RF_amp'] = 0.00
        self.pulsedODMR_params['RF_pilen'] = 30.0e-6
        self.pulsedODMR_params['num_of_points'] = 50
        self.pulsedODMR_sweeps = 60000

        self.ramsey_params = self.pulsed_master_logic.generate_method_params['ramsey']
        self.ramsey_params['name'] = 'ramsey'
        self.ramsey_params['tau_start'] = 0.0e-9
        self.ramsey_params['tau_step'] = 301.0e-9
        self.ramsey_params['laser_on'] = 20.0e-9
        self.ramsey_params['laser_off'] = 60.0e-9
        self.ramsey_params['num_of_points'] = 30
        self.ramsey_params['alternating'] = True
        self.ramsey_sweeps = 60e3

        self.hahn_params = self.pulsed_master_logic.generate_method_params['hahnecho_exp']
        self.hahn_params['name'] = 'hahn_echo'
        self.hahn_params['tau_start'] = 0.0e-9
        self.hahn_params['tau_end'] = 2.0e-3
        self.hahn_params['laser_on'] = 20.0e-9
        self.hahn_params['laser_off'] = 60.0e-9
        self.hahn_params['num_of_points'] = 20
        self.hahn_params['alternating'] = True
        self.hahn_sweeps = 60e3

        self.nucspect_params = self.pulsed_master_logic.generate_method_params['NucSpect']
        self.nucspect_params['name'] = 'nucspect'
        self.nucspect_params['NV_pi'] = False
        self.nucspect_params['RF_pi'] = False
        self.nucspect_params['RF_freq'] = 5.05e6
        self.nucspect_params['RF_amp'] = 0.00
        self.nucspect_params['RF_pi_len'] = 40.0e-9
        self.nucspect_params['freq_start'] = 5.05e6
        self.nucspect_params['freq_step'] = 2.0e3
        self.nucspect_params['spect_amp'] = self.nucrabi_RFfreq0_amp
        self.nucspect_params['spect_pi'] = 126000e-9
        self.nucspect_params['num_of_points'] = 50
        self.nucspect_params['laser_on'] = 20.0e-9
        self.nucspect_params['laser_off'] = 60.0e-9
        self.nucspect_sweeps = 60000

        self.nucrabi_params = self.pulsed_master_logic.generate_method_params['NucRabi']
        self.nucrabi_params['name'] = 'nucrabi'
        self.nucrabi_params['NV_pi'] = False
        self.nucrabi_params['RF_pi'] = False
        self.nucrabi_params['RF_freq'] = 5.05e6
        self.nucrabi_params['RF_amp'] = 0.00
        self.nucrabi_params['RF_pi_len'] = 100.0e-9
        self.nucrabi_params['Nuc_rabi_freq'] = 5.06e6
        self.nucrabi_params['Nuc_rabi_amp'] = self.nucrabi_RFfreq0_amp
        self.nucrabi_params['tau_start'] = 10.0e-9
        self.nucrabi_params['tau_step'] = 16.7e-6
        self.nucrabi_params['num_of_points'] = 25
        self.nucrabi_params['laser_on'] = 20.0e-9
        self.nucrabi_params['laser_off'] = 60.0e-9
        self.nucrabi_sweeps = 60000

        self.DDrfspect_params = self.pulsed_master_logic.generate_method_params['DDrf_Spect']
        self.DDrfspect_params['name'] = 'ddrf_spect'
        self.DDrfspect_params['freq'] = 5.096e6
        self.DDrfspect_params['RF_freq'] =2.56e6
        self.DDrfspect_params['RF_amp'] =0.020
        self.DDrfspect_params['cyclesf'] =4
        self.DDrfspect_params['rot_phase'] = 0
        self.DDrfspect_params['DD_order'] = 10
        self.DDrfspect_params['num_of_points'] = 10
        self.DDrfspect_params['laser_on'] = 20.0e-9
        self.DDrfspect_params['laser_off'] = 60.0e-9
        self.DDrfspect_sweeps = 100000

        self.axy8_params = self.pulsed_master_logic.generate_method_params['AXY']
        self.axy8_params['name'] = 'axy8'
        self.axy8_params['tau_start'] = 0.5e-6
        self.axy8_params['tau_step'] =  10.0e-9
        self.axy8_params['num_of_points'] = 50
        self.axy8_params['xy8_order'] = 4
        self.axy8_params['f1'] = 0.9
        self.axy8_params['scale_tau2_first'] = 1.0
        self.axy8_params['scale_tau2_last'] = 1.0
        self.axy8_params['init_pihalf_pulse'] = True
        self.axy8_params['Init_phase'] = 0
        self.axy8_params['Read_phase'] = 0
        self.axy8_params['laser_on'] = 20.0e-9
        self.axy8_params['laser_off'] = 60.0e-9
        self.axy8_sweeps = 20e3

        self.xy8_params = self.pulsed_master_logic.generate_method_params['xy8_tau']
        self.xy8_params['name'] = 'xy8_tau'
        self.xy8_params['tau_start'] = 0.5e-6
        self.xy8_params['tau_step'] =  10.0e-9
        self.xy8_params['num_of_points'] = 50
        self.xy8_params['xy8_order'] = 4
        self.xy8_params['init_phase'] = 90
        self.xy8_params['read_phase'] = 90
        self.xy8_params['laser_on'] = 20.0e-9
        self.xy8_params['laser_off'] = 60.0e-9
        self.xy8_params['alternating'] = True
        self.xy8_sweeps = 30e3

        self.QCQB12_params = self.pulsed_master_logic.generate_method_params['QuantumCircuitQB12']
        self.QCQB12_params['name'] = 'quantumcircuitQB12'
        self.QCQB12_params['NV_Cpi_amp'] = self.microwave_amplitude_LowPower
        self.QCQB12_params['RF_amp0'] = self.nucrabi_RFfreq0_amp
        self.QCQB12_params['RF_amp1'] = self.nucrabi_RFfreq1_amp
        self.QCQB12_params['cyclesf'] = 7
        self.QCQB12_params['DD_N'] = 8
        self.QCQB12_params['num_of_points'] = 20
        self.QCQB12_params['laser_on'] = 20.0e-9
        self.QCQB12_params['laser_off'] = 60.0e-9
        self.QCQB12_sweeps = 20e3

        self.QCQSTQB13_params = self.pulsed_master_logic.generate_method_params['QuantumCircuitQstQB13']
        self.QCQSTQB13_params['name'] = 'quantumcircuitQstQB13'
        self.QCQSTQB13_params['f1_uc'] = 0.95
        self.QCQSTQB13_params['tau_uc'] = 1602.5e-9
        self.QCQSTQB13_params['order_uc'] = 15
        self.QCQSTQB13_params['f1_c'] = 0.9
        self.QCQSTQB13_params['tau_c'] = 2*354.95e-9
        self.QCQSTQB13_params['order_c'] = 6
        self.QCQSTQB13_params['tau_z'] = 2*734.86e-9
        self.QCQSTQB13_params['order_z'] = 4
        self.QCQSTQB13_params['num_of_points'] = 20
        self.QCQSTQB13_params['laser_on'] = 20.0e-9
        self.QCQSTQB13_params['laser_off'] = 60.0e-9
        self.QCQSTQB13_sweeps = 60e3

        self.QCQB123_params = self.pulsed_master_logic.generate_method_params['QuantumCircuitQB123']
        self.QCQB123_params['name'] = 'quantumcircuitQB123'
        self.QCQB123_params['NV_Cpi_amp'] = self.microwave_amplitude_LowPower
        self.QCQB123_params['RF_amp0'] = self.nucrabi_RFfreq0_amp
        self.QCQB123_params['RF_amp1'] = self.nucrabi_RFfreq1_amp
        self.QCQB123_params['cyclesf'] = 7
        self.QCQB123_params['DD_N'] = 8
        self.QCQB123_params['f1_uc'] = 0.95
        self.QCQB123_params['tau_uc'] = 1610e-9
        self.QCQB123_params['order_uc'] = 15
        self.QCQB123_params['f1_c'] = 0.9
        self.QCQB123_params['tau_c'] = 805e-9
        self.QCQB123_params['order_c'] = 8
        self.QCQB123_params['tau_z'] = 1.5325e-6
        self.QCQB123_params['order_z'] = 4
        self.QCQB123_params['num_of_points'] = 20
        self.QCQB123_params['laser_on'] = 20.0e-9
        self.QCQB123_params['laser_off'] = 60.0e-9
        self.QCQB123_sweeps = 20e3

        self.QCQB1234_params = self.pulsed_master_logic.generate_method_params['QuantumCircuitQB1234']
        self.QCQB1234_params['name'] = 'quantumcircuitQB1234'
        self.QCQB1234_params['NV_Cpi_amp'] = self.microwave_amplitude_LowPower
        self.QCQB1234_params['RF_amp0'] = self.nucrabi_RFfreq0_amp
        self.QCQB1234_params['RF_amp1'] = self.nucrabi_RFfreq1_amp
        self.QCQB1234_params['cyclesf'] = 7
        self.QCQB1234_params['DD_N'] = 8
        self.QCQB1234_params['f1_uc'] = 0.95
        self.QCQB1234_params['tau_uc'] = 1608e-9
        self.QCQB1234_params['order_uc'] = 15
        self.QCQB1234_params['f1_c'] = 0.9
        self.QCQB1234_params['tau_c'] = 804e-9
        self.QCQB1234_params['order_c'] = 4
        self.QCQB1234_params['tau_z'] = 766.25e-9
        self.QCQB1234_params['order_z'] = 4
        self.QCQB1234_params['num_of_points'] = 20
        self.QCQB1234_params['laser_on'] = 20.0e-9
        self.QCQB1234_params['laser_off'] = 60.0e-9
        self.QCQB1234_sweeps = 20e3


    def saveCalibParams(self):
        filename = self.calibParamsFilePrefix + datetime.datetime.now().strftime('%Y%m%d_%H%M') + '.json'
        file = open(filename, 'w')
        json.dump(self.calib_params, file, indent=4)
        file.close()
        print(f"INFO: saved updated calibration parameters to file '{os.path.basename(filename)}'")


    def saveMeasurementResult(self, data):
        filename = self.measResFilePrefix + datetime.datetime.now().strftime('%Y%m%d_%H%M') + '.json'
        file = open(filename, 'w')
        json.dump(data, file, indent=4)
        file.close()


    def loadCalibParams(self):
        newestCalibFile = sorted( os.listdir( os.path.dirname(self.calibParamsFilePrefix) ), reverse=True )[0]
        file = open( os.path.join(os.path.dirname(self.calibParamsFilePrefix), newestCalibFile), 'r' )
        self.calib_params = json.load(file)
        timeStamp = datetime.datetime.strptime( newestCalibFile,  os.path.basename(xq1i.calibParamsFilePrefix) + '%Y%m%d_%H%M' + '.json' )
        print(f"INFO: loaded calibration parameters from file '{newestCalibFile}'")
        if timeStamp + datetime.timedelta(hours=3) < datetime.datetime.now():
            print('\033[91m' + ('WARNING: Most recent calibration data is older than 3 hours. ' +
                                'Please refer to the manual for recommended calibration intervals.') + '\033[0m')


    # def write_to_logfile(self, nametag, timestamp,name, **kwargs):
        # """ Write parameters to custom logfile with name nametag """
        # if type(timestamp) is not str:
            # timestamp = str(timestamp)
        # parameters = list(kwargs)
        # if len(parameters) == 1 and type(kwargs[parameters[0]]) is OrderedDict:
            # param_dict = kwargs[parameters[0]]
            # parameters = list(param_dict)
            # kwargs = param_dict
        # log_dir = savelogic.get_path_for_module('D:\Data')
        # log_path = os.path.join(log_dir, nametag + '.txt')
        # if not os.path.isfile(log_path):
            # with open(log_path, 'w') as logfile:
                # logfile.write('# timestamp\t')
                # logfile.write('# Name\t')
                # for param in parameters:
                    # logfile.write(param + '\t')
                # logfile.write('\n#\n')
        # with open(log_path, 'a') as logfile:
            # logfile.write(timestamp + '\t')
            # logfile.write(name + '\t')
            # for param in parameters:
                # logfile.write('{0:3.6e}\t'.format(kwargs[param]))
            # logfile.write('\n')
        # return

    def qb1_calibration(self):
        try:
            print(f"performing calibration of QB1 ...")

            # Measure Qubit-1 transition frequency#
            self.generate_params['microwave_frequency'] = self.calib_params['res_freq']
            self.pulsedODMR_params['freq_start'] = self.calib_params['res_freq'] - 5.0e6
            self.pulsedODMR_params['freq_step'] = 0.2e6
            self.do_pulsedODMR()
            time.sleep(2)
            result_dict = self.pulsed_measurement_logic.do_fit('Lorentzian Dip')
            self.calib_params['res_freq'] = float(result_dict.params['center'].value)
            time.sleep(2)

            # Measure Qubit-1 gate parameters#
            self.do_rabi(isSlow=False)  # perform fast Rabi
            result_dict = self.pulsed_measurement_logic.do_fit('Sine')
            self.calib_params['rabi_period_HighPower'] = float(1 / (result_dict.params['frequency'].value))
            self.generate_params['rabi_period'] = self.calib_params['rabi_period_HighPower']
            self.pulsed_master_logic.set_generation_parameters(self.generate_params)
            # xq1i.pulsed_measurement_logic.do_fit('No Fit')
            time.sleep(2)

            # calibration of CnNOTe (if nuclear spin in state 0, electron spin is flipped)#
            self.do_rabi(isSlow=True)  # perform slow Rabi
            result_dict = self.pulsed_measurement_logic.do_fit('Sine')
            self.calib_params['rabi_period_LowPower'] = float(1 / (result_dict.params['frequency'].value))
            self.calib_params['rabi_offset'] = float(result_dict.params['offset'].value)
            self.calib_params['rabi_amplitude'] = abs(float(result_dict.params['amplitude'].value))

            self.saveCalibParams()
        except KeyboardInterrupt:
            self._interruptPulsedMeasurement()
            print('\033[91m' + 'WARNING: User interrupt of QB1 calibration, measurement sequence stopped.' + '\033[0m')


    def qb2_calibration(self, type='partial'):
        try:
            print(f"performing calibration of QB2 ...")

            # Measure Qubit-2 transition frequency m_s=0#
            self.nucspect_params['NV_pi'] = False
            self.nucspect_params['freq_start'] = 5.05e6
            self.nucspect_params['spect_amp'] = self.nucrabi_RFfreq0_amp
            self.do_Nucspect()
            time.sleep(2)
            result_dict = self.pulsed_measurement_logic.do_fit('Lorentzian Dip')
            self.calib_params['RF_freq0'] = float(result_dict.params['center'].value)

            # Measure Qubit-2 gate parameters m_s=0#
            self.nucrabi_params['NV_pi'] = False
            self.nucrabi_params['Nuc_rabi_amp'] = self.nucrabi_RFfreq0_amp
            self.do_NucRabi()
            result_dict = self.pulsed_measurement_logic.do_fit('Sine')
            self.calib_params['nucrabi_RFfreq0_period'] = float(1 / (result_dict.params['frequency'].value))

            # Measure Qubit-2 transition frequency m_s=1#
            self.nucspect_params['NV_pi'] = True
            self.nucspect_params['spect_amp'] = self.nucrabi_RFfreq1_amp
            self.nucspect_params['freq_start'] = 2.90e6
            self.do_Nucspect()
            time.sleep(2)
            result_dict = self.pulsed_measurement_logic.do_fit('Lorentzian Dip')
            self.calib_params['RF_freq1'] = float(result_dict.params['center'].value)

            # Measure Qubit-2 gate parameters m_s=1#
            self.nucrabi_params['NV_pi'] = True
            self.nucrabi_params['Nuc_rabi_amp'] = self.nucrabi_RFfreq1_amp
            self.do_NucRabi()
            result_dict = self.pulsed_measurement_logic.do_fit('Sine')
            self.calib_params['nucrabi_RFfreq1_period'] = float(1 / (result_dict.params['frequency'].value))

            self.saveCalibParams()

            if type == 'full':
                # DDRF transition frequency measurement
                self.DDrfspect_params['freq'] = self.calib_params['RF_freq0']
                self.DDrfspect_params['RF_amp'] = 0.020
                self.DDrfspect_params['cyclesf'] = 4
                self.DDrfspect_params['DD_order'] = 10
                outfile = open("./14N_Calibration/DDRFamplist_freq_{:.0f}.txt".format(self.DDrfspect_params['freq']), "w")
                DDRFamplist = []
                freqls = np.arange(2900e3, 3000e3, 2e3).tolist()
                for freq in freqls:
                    self.DDrfspect_params['RF_freq'] = freq
                    self.DDrfspect_sweeps = 20000
                    self.do_DDrf_Spect()
                    result_dict = self.pulsed_measurement_logic.do_fit('Sine_Fixed_Freq_360')
                    amplitude = float(result_dict.params['amplitude'].value)
                    DDRFamplist.append([freq, amplitude])
                    outfile.write("{:.15f}\t{:.15f}\n".format(freq, amplitude))
                    outfile.flush()
                    time.sleep(2)
                outfile.close()
                fig = plt.figure()
                ax = plt.axes()
                ax.set_title('DDRF calibration')
                ax.set_xlabel('frequency in Hz')
                ax.set_ylabel('amplitude')
                DDRFampArray = np.array(DDRFamplist)
                ax.plot(DDRFampArray[:, 0], DDRFampArray[:, 1])
                plt.show()

        except KeyboardInterrupt:
            self._interruptPulsedMeasurement()
            print('\033[91m' + 'WARNING: User interrupt of QB2 calibration, measurement sequence stopped.' + '\033[0m')


    def qb3_calibration(self):
        try:
            print(f"performing calibration of QB3 ...")
            #self.do_AXY8_Spect()
            self.do_XY8_Spect()
        except KeyboardInterrupt:
            self._interruptPulsedMeasurement()
            print('\033[91m' + 'WARNING: User interrupt of QB3 calibration, measurement sequence stopped.' + '\033[0m')


    def qb4_calibration(self):
        try:
            print(f"performing calibration of QB4 ...")

            # DDRF transition frequency measurement
            self.DDrfspect_params['freq'] = 544.1e3
            self.DDrfspect_params['RF_amp'] = 0.020
            self.DDrfspect_params['cyclesf'] = 4
            self.DDrfspect_params['DD_order'] = 10
            outfile = open(self.calibParamsFilePrefix + "QB4_DDRFamplist_freq_{:.0f}.txt".format(self.DDrfspect_params['freq']), "w")
            DDRFamplist = []
            freqls = np.arange(170e3, 250e3, 2e3).tolist()
            for freq in freqls:
                self.DDrfspect_params['RF_freq'] = freq
                self.DDrfspect_sweeps = 20000
                self.do_DDrf_Spect()
                result_dict = self.pulsed_measurement_logic.do_fit('Sine_Fixed_Freq_360')
                amplitude = float(result_dict.params['amplitude'].value)
                DDRFamplist.append([freq, amplitude])
                outfile.write("{:.15f}\t{:.15f}\n".format(freq, amplitude))
                outfile.flush()
                time.sleep(2)
            outfile.close()
            fig = plt.figure()
            ax = plt.axes()
            ax.set_title('DDRF calibration')
            ax.set_xlabel('frequency in Hz')
            ax.set_ylabel('amplitude')
            DDRFampArray = np.array(DDRFamplist)
            ax.plot(DDRFampArray[:, 0], DDRFampArray[:, 1])
            plt.show()
        except KeyboardInterrupt:
            self._interruptPulsedMeasurement()
            print('\033[91m' + 'WARNING: User interrupt of QB4 calibration, measurement sequence stopped.' + '\033[0m')


    def _interruptPulsedMeasurement(self):
        if self.pulsed_measurement_logic.module_state() == 'locked':
            self.pulsed_master_logic.toggle_pulsed_measurement(False)
            # Wait until the self.pulsed_measurement_logic is actually idle and the measurement is stopped
            while self.pulsed_measurement_logic.module_state() == 'locked':
                time.sleep(0.5)

    def _executePulsedMeasurement(self, name, sweeps):
        with tqdm(total=sweeps, leave=True, unit='sweeps', desc=f' ... {name}') as pbar:
            time.sleep(0.5)
            self.pulsed_master_logic.sample_ensemble(name, with_load=False)
            while self.pulsed_master_logic.status_dict['sampling_ensemble_busy']:
                time.sleep(0.5)
            self.pulsed_master_logic.load_ensemble(name)
            while self.pulsed_master_logic.status_dict['loading_busy']:
                time.sleep(0.5)
            self.pulsed_master_logic.set_measurement_settings(invoke_settings=True)
            time.sleep(0.5)
            self.pulsed_master_logic.set_timer_interval(5)
            time.sleep(5.0)
            #return
            self.pulsed_master_logic.toggle_pulsed_measurement(True)
            while self.pulsed_measurement_logic.module_state() != 'locked':
                time.sleep(0.5)
            user_terminated = False
            while self.pulsed_measurement_logic.elapsed_sweeps < sweeps:
                pbar.n = int(self.pulsed_measurement_logic.elapsed_sweeps)
                pbar.refresh()
                if self.pulsed_measurement_logic.module_state() != 'locked':
                    user_terminated = True
                    break
                time.sleep(0.5)
            pbar.total = int(self.pulsed_measurement_logic.elapsed_sweeps)
            pbar.n = int(self.pulsed_measurement_logic.elapsed_sweeps)
            self.pulsed_master_logic.manually_pull_data()
            time.sleep(1)
            self.pulsed_master_logic.toggle_pulsed_measurement(False)
            # Wait until the self.pulsed_measurement_logic is actually idle and the measurement is stopped
            while self.pulsed_measurement_logic.module_state() == 'locked':
                time.sleep(0.5)


    def do_rabi(self, isSlow=False):
        if not isSlow:
            self.generate_params['microwave_frequency'] = self.calib_params['res_freq']
            self.generate_params['microwave_amplitude'] = self.microwave_amplitude_HighPower
            self.generate_params['rabi_period'] = self.calib_params['rabi_period_HighPower']
            self.pulsed_master_logic.set_generation_parameters(self.generate_params)
            self.rabi_params['tau_start'] = 0.0e-9
            self.rabi_params['tau_step'] = 17.0e-9
            self.rabi_params['delay_time'] = 50.0e-6
            self.rabi_params['num_of_points'] = 30
        else:
            self.generate_params['microwave_frequency'] = self.calib_params['res_freq']
            self.generate_params['microwave_amplitude'] = self.microwave_amplitude_LowPower  #value needs to be checked/maybe tuned
            self.generate_params['rabi_period'] = self.calib_params['rabi_period_HighPower']
            self.pulsed_master_logic.set_generation_parameters(self.generate_params)
            self.rabi_params['tau_start'] = 0.0e-9
            self.rabi_params['tau_step'] = 246.0e-9
            self.rabi_params['delay_time'] = 50.0e-6
            self.rabi_params['num_of_points'] = 30

        self.sequence_generator_logic.delete_ensemble('rabi')
        self.sequence_generator_logic.delete_block('rabi')
        self.pulsed_master_logic.generate_predefined_sequence('rabi', self.rabi_params)

        self._executePulsedMeasurement('rabi', self.rabi_sweeps)
        self.pulsed_measurement_logic.do_fit('Sine')
        self.pulsed_master_logic.save_measurement_data(tag=self.POI_name + '_Rabi'
                                                +'_freq_'+str(round((self.generate_params['microwave_frequency']/(1e6)),4))+'MHz'
                                                +'_amp_'+str(round((self.generate_params['microwave_amplitude']),4))+'V', with_error=False)
        # reset to fast Rabi parameters (default Rabi)
        if isSlow:
            self.generate_params['microwave_frequency'] = self.calib_params['res_freq']
            self.generate_params['microwave_amplitude'] = self.microwave_amplitude_HighPower
            self.generate_params['rabi_period'] = self.calib_params['rabi_period_HighPower']
            self.pulsed_master_logic.set_generation_parameters(self.generate_params)



    def do_pulsedODMR(self):
        self.generate_params['microwave_amplitude'] = self.microwave_amplitude_LowPower
        self.generate_params['rabi_period'] = self.calib_params['rabi_period_LowPower']
        self.pulsed_master_logic.set_generation_parameters(self.generate_params)

        self.sequence_generator_logic.delete_ensemble('pulsedODMR')
        self.sequence_generator_logic.delete_block('pulsedODMR')
        self.pulsed_master_logic.generate_predefined_sequence('pulsedodmr', self.pulsedODMR_params)

        self._executePulsedMeasurement('pulsedODMR', self.pulsedODMR_sweeps)
        self.pulsed_measurement_logic.do_fit('Lorentzian Dip')
        self.pulsed_master_logic.save_measurement_data(tag=self.POI_name + '_PODMR'
                                               +'_amp_'+str(round((self.generate_params['microwave_amplitude']),4))+'V', with_error=False)

        self.generate_params['microwave_amplitude'] = self.microwave_amplitude_HighPower
        self.generate_params['rabi_period'] = self.calib_params['rabi_period_HighPower']
        self.pulsed_master_logic.set_generation_parameters(self.generate_params)


    def do_ramsey(self):
        self.generate_params['wait_time'] = 1.5e-6
        self.pulsed_master_logic.set_generation_parameters(self.generate_params)

        self.sequence_generator_logic.delete_ensemble('ramsey')
        self.sequence_generator_logic.delete_block('ramsey')
        self.pulsed_master_logic.generate_predefined_sequence('ramsey', self.ramsey_params)

        self._executePulsedMeasurement('ramsey', self.ramsey_sweeps)
        result_dict = self.pulsed_measurement_logic.do_fit('Exp. Decay Sine')
        t2star = netobtain(result_dict.params)['decay'].value
        self.pulsed_master_logic.save_measurement_data(tag=self.POI_name + '_Ramsey', with_error=False)

        self.generate_params['wait_time'] = 70.5e-6
        self.pulsed_master_logic.set_generation_parameters(self.generate_params)

        tData = netobtain(self.pulsed_measurement_logic.signal_data[0])
        sigData = (
                    (netobtain(self.pulsed_measurement_logic.signal_data[2]) - netobtain(self.pulsed_measurement_logic.signal_data[1])) /
                    (netobtain(self.pulsed_measurement_logic.signal_data[1]) + netobtain(self.pulsed_measurement_logic.signal_data[2]))
                  )
        plt.figure(figsize=(7, 1.5))
        plt.scatter(tData, sigData, s=10)
        #plt.plot(tFit, sigFit, color='tab:blue')
        plt.grid()
        #plt.yticks([0, 0.5, 1.0])
        #plt.ylim([-0.3, 1.3])
        plt.title(rf'Ramsey experiment, measured dephasing time: $T_2^* =${t2star * 1e6:.2f} $\mu s$', fontsize=10)
        plt.ylabel('norm. counts')
        plt.xlabel('free precession time (s)')
        plt.show()


    def do_echo_exp(self):
        self.generate_params['wait_time'] = 1.5e-6
        self.pulsed_master_logic.set_generation_parameters(self.generate_params)

        self.sequence_generator_logic.delete_ensemble('hahn_echo')
        self.sequence_generator_logic.delete_block('hahn_echo')
        self.pulsed_master_logic.generate_predefined_sequence('hahnecho_exp', self.hahn_params)

        self._executePulsedMeasurement('hahn_echo', self.hahn_sweeps)
        result_dict = self.pulsed_measurement_logic.do_fit('Exp Decay')
        t2 = netobtain(result_dict.params)['decay'].value
        self.pulsed_master_logic.save_measurement_data(tag=self.POI_name + '_Hahn', with_error=False)

        self.generate_params['wait_time'] = 70.5e-6
        self.pulsed_master_logic.set_generation_parameters(self.generate_params)

        tData = netobtain(self.pulsed_measurement_logic.signal_data[0])
        sigData = (
                    (netobtain(self.pulsed_measurement_logic.signal_data[1]) - netobtain(self.pulsed_measurement_logic.signal_data[2])) /
                    (netobtain(self.pulsed_measurement_logic.signal_data[1]) + netobtain(self.pulsed_measurement_logic.signal_data[2]))
                  )
        plt.figure(figsize=(7, 1.5))
        plt.scatter(tData, sigData, s=10)
        #plt.plot(tFit, sigFit, color='tab:blue')
        plt.grid()
        #plt.yticks([0, 0.5, 1.0])
        #plt.ylim([-0.3, 1.3])
        plt.title(rf'Hahn echo experiment, measured coherence time: $T_2 =${t2 * 1e6:.2f} $\mu s$', fontsize=10)
        plt.ylabel('norm. counts')
        plt.xlabel('free precession time (s)')
        plt.show()


    def do_Nucspect(self):
        self.sequence_generator_logic.delete_ensemble('nucspect')
        self.sequence_generator_logic.delete_block('nucspect')
        self.pulsed_master_logic.generate_predefined_sequence('NucSpect', self.nucspect_params)

        self._executePulsedMeasurement('nucspect', self.nucspect_sweeps)
        self.pulsed_measurement_logic.do_fit('Lorentzian Dip')
        self.pulsed_master_logic.save_measurement_data(tag=self.POI_name + '_Nucspect'
                                                +'_freq_'+str(round((self.nucspect_params['spect_pi']/(1e-6)),4))+'us'
                                                +'_amp_'+str(round((self.nucspect_params['spect_amp']),4))+'V'
                                                +'_NVms1'+ str(self.nucspect_params['NV_pi']), with_error=False)


    def do_NucRabi(self):
        if self.nucrabi_params['NV_pi'] == False:
            self.nucrabi_params['Nuc_rabi_freq'] = self.calib_params['RF_freq0']
            self.nucrabi_params['Nuc_rabi_amp'] = self.nucrabi_RFfreq0_amp
        else:
            self.nucrabi_params['Nuc_rabi_freq'] = self.calib_params['RF_freq1']
            self.nucrabi_params['Nuc_rabi_amp'] = self.nucrabi_RFfreq1_amp
        self.sequence_generator_logic.delete_ensemble('nucrabi')
        self.sequence_generator_logic.delete_block('nucrabi')
        self.pulsed_master_logic.generate_predefined_sequence('NucRabi', self.nucrabi_params)

        self._executePulsedMeasurement('nucrabi', self.nucrabi_sweeps)
        self.pulsed_measurement_logic.do_fit('Sine')
        self.pulsed_master_logic.save_measurement_data(tag=self.POI_name + '_Nucrabi'
                                                +'_freq_'+str(round((self.nucrabi_params['Nuc_rabi_freq']/(1e6)),4))+'MHz'
                                                +'_amp_'+str(round((self.nucrabi_params['Nuc_rabi_amp']),3))+'V'
                                                +'_NVms1_'+ str(self.nucrabi_params['NV_pi']), with_error=False)


    def do_DDrf_Spect(self):
        self.sequence_generator_logic.delete_ensemble('ddrf_spect')
        self.sequence_generator_logic.delete_block('ddrf_spect')
        self.pulsed_master_logic.generate_predefined_sequence('DDrf_Spect', self.DDrfspect_params)

        self._executePulsedMeasurement('ddrf_spect', self.DDrfspect_sweeps)
        self.pulsed_measurement_logic.do_fit('Sine')
        self.pulsed_master_logic.save_measurement_data(tag= self.POI_name + '_DDrf_spect_'
                                                +'_Freq_'+str(round((self.DDrfspect_params['freq']/(1e6)),4))+'_MHz'
                                                +'_RFFreq_'+str(round((self.DDrfspect_params['RF_freq']/(1e6)),4))+'_MHz'
                                                +'_cycles_'+str(self.DDrfspect_params['cyclesf'])
                                                +'_DDorder_'+str(self.DDrfspect_params['DD_order'])
                                                +'_RotPhase_'+ str(self.DDrfspect_params['rot_phase']), with_error=False)


    def do_AXY8_Spect(self):
        self.sequence_generator_logic.delete_ensemble('axy8')
        self.sequence_generator_logic.delete_block('axy8')
        self.pulsed_master_logic.generate_predefined_sequence('AXY', self.axy8_params)

        self._executePulsedMeasurement('axy8', self.axy8_sweeps)
        self.pulsed_measurement_logic.do_fit('Lorentzian Peak')
        self.pulsed_master_logic.save_measurement_data(tag = self.POI_name +
                                                + f'_AXY8_order_{self.axy8_params["xy8_order"]}_f1_{self.axy8_params["f1"]:.2f}'
                                                ,with_error=False)


    def do_XY8_Spect(self):
        self.sequence_generator_logic.delete_ensemble('xy8_tau')
        self.sequence_generator_logic.delete_block('xy8_tau')
        self.pulsed_master_logic.generate_predefined_sequence('xy8_tau', self.xy8_params)

        #self._executePulsedMeasurement('xy8_tau', self.xy8_sweeps)
        #self.pulsed_measurement_logic.do_fit('Double Lorentzian Peaks')
        self.pulsed_master_logic.save_measurement_data(tag = self.POI_name + f'_XY8_order_{self.xy8_params["xy8_order"]}'
                                                ,with_error=False)

        tData = netobtain(self.pulsed_measurement_logic.signal_data[0])
        sigData = (
                    (netobtain(self.pulsed_measurement_logic.signal_data[2]) - netobtain(self.pulsed_measurement_logic.signal_data[1])) /
                    (netobtain(self.pulsed_measurement_logic.signal_data[1]) + netobtain(self.pulsed_measurement_logic.signal_data[2]))
                  )
        plt.figure(figsize=(7, 1.5))
        plt.scatter(tData, sigData, s=10)
        #plt.plot(tFit, sigFit, color='tab:blue')
        plt.grid()
        #plt.yticks([0, 0.5, 1.0])
        #plt.ylim([-0.3, 1.3])
        plt.title(rf'XY8 dynamical decoupling experiment', fontsize=10)
        plt.ylabel('norm. counts')
        plt.xlabel(r'$\tau$ (s)')
        plt.show()


    def do_QuantumCircuitQB12(self, qcQB12, initState=TQstates.State00, readoutCirc=TQReadoutCircs.P00, sweeps=100e3):
        self.QCQB12_sweeps = sweeps
        self.QCQB12_params['NV_Cpi_len'] = self.calib_params['rabi_period_LowPower']/2
        self.QCQB12_params['NV_Cpi_freq1'] = self.calib_params['res_freq']
        self.QCQB12_params['RF_freq0'] = self.calib_params['RF_freq0']
        self.QCQB12_params['RF_freq1'] = self.calib_params['RF_freq1']
        self.QCQB12_params['RF_pi'] = self.calib_params['nucrabi_RFfreq0_period']/2 #(self.calib_params['nucrabi_RFfreq0_period'] + self.calib_params['nucrabi_RFfreq1_period'])/4
        self.QCQB12_params['Initial_state'] = initState
        self.QCQB12_params['Readout_circ'] = readoutCirc
        self.QCQB12_params['gate_operations'] = ", ".join([f"{gate.name}({gate.param})[{gate.qubit}]"  for gate in qcQB12])
        #print(self.QCQB12_params['gate_operations'])
        self.sequence_generator_logic.delete_ensemble('quantumcircuitQB12')
        self.sequence_generator_logic.delete_block('quantumcircuitQB12')
        self.pulsed_master_logic.generate_predefined_sequence('QuantumCircuitQB12', self.QCQB12_params)

        self._executePulsedMeasurement('quantumcircuitQB12', self.QCQB12_sweeps)
        time.sleep(1)
        self.pulsed_master_logic.save_measurement_data(tag = self.POI_name + '_QCQB12_'
                                                + '_Initstate_' + self.QCQB12_params['Initial_state'].name
                                                + '_Readcirc_' + self.QCQB12_params['Readout_circ'].name + '_'
                                                + '_'.join([f"{gate.name}QB{gate.qubit}" for gate in qcQB12]),
                                                with_error=True)
        # workaround to prevent exception in yaml-dump of status variables during qudi shutdown
        # (cannot handle enum type behind an rpyc-netref, exception raised in "represent_undefined" in ruamel/yaml/representer.py)
        self.QCQB12_params['Initial_state'] = None
        self.QCQB12_params['Readout_circ'] = None


    def do_QuantumCircuitQstQB13(self, qcQB13, initState=TQstates.State00, readoutCirc=TQQSTReadoutCircs.IX, sweeps=100e3):
        self.QCQSTQB13_sweeps = sweeps
        self.QCQSTQB13_params['Initial_state'] = initState
        self.QCQSTQB13_params['Readout_circ'] = readoutCirc
        self.QCQSTQB13_params['gate_operations'] = ", ".join([f"{gate.name}({gate.param})[{gate.qubit}]"  for gate in qcQB13])
        self.sequence_generator_logic.delete_ensemble('quantumcircuitQstQB13')
        self.sequence_generator_logic.delete_block('quantumcircuitQstQB13')
        self.pulsed_master_logic.generate_predefined_sequence('QuantumCircuitQstQB13', self.QCQSTQB13_params)

        self._executePulsedMeasurement('quantumcircuitQstQB13', self.QCQSTQB13_sweeps)
        time.sleep(1)
        self.pulsed_master_logic.save_measurement_data(tag = self.POI_name + '_QCQSTQB13_'
                                                + '_Initstate_' + self.QCQSTQB13_params['Initial_state'].name
                                                + '_Readcirc_' + self.QCQSTQB13_params['Readout_circ'].name + '_'
                                                + '_'.join([f"{gate.name}QB{gate.qubit}" for gate in qcQB13]),
                                                with_error=True)
        # workaround to prevent exception in yaml-dump of status variables during qudi shutdown
        # (cannot handle enum type behind an rpyc-netref, exception raised in "represent_undefined" in ruamel/yaml/representer.py)
        self.QCQSTQB13_params['Initial_state'] = None
        self.QCQSTQB13_params['Readout_circ'] = None


    def do_QuantumCircuitQB123(self, qcQB123, initState=ThrQstates.State000, readoutCirc=ThrQReadoutCircs.IZI1, sweeps=100e3):
        self.QCQB123_sweeps = sweeps
        self.QCQB123_params['NV_Cpi_len'] = self.calib_params['rabi_period_LowPower']/2
        self.QCQB123_params['NV_Cpi_freq1'] = self.calib_params['res_freq']
        self.QCQB123_params['RF_freq0'] = self.calib_params['RF_freq0']
        self.QCQB123_params['RF_freq1'] = self.calib_params['RF_freq1']
        self.QCQB123_params['RF_pi'] = (self.calib_params['nucrabi_RFfreq0_period'] + self.calib_params['nucrabi_RFfreq1_period'])/4
        self.QCQB123_params['Initial_state'] = initState
        self.QCQB123_params['Readout_circ'] = readoutCirc
        self.QCQB123_params['gate_operations'] = ", ".join([f"{gate.name}({gate.param})[{gate.qubit}]"  for gate in qcQB123])
        self.sequence_generator_logic.delete_ensemble('quantumcircuitQB123')
        self.sequence_generator_logic.delete_block('quantumcircuitQB123')
        self.pulsed_master_logic.generate_predefined_sequence('QuantumCircuitQB123', self.QCQB123_params)

        self._executePulsedMeasurement('quantumcircuitQB123', self.QCQB123_sweeps)
        time.sleep(1)
        self.pulsed_master_logic.save_measurement_data(tag = self.POI_name + '_QCQB123_'
                                                + '_Initstate_' + self.QCQB123_params['Initial_state'].name
                                                + '_Readcirc_' + self.QCQB123_params['Readout_circ'].name + '_'
                                                + '_'.join([f"{gate.name}QB{gate.qubit}" for gate in qcQB123]),
                                                with_error=True)
        # workaround to prevent exception in yaml-dump of status variables during qudi shutdown
        # (cannot handle enum type behind an rpyc-netref, exception raised in "represent_undefined" in ruamel/yaml/representer.py)
        self.QCQB123_params['Initial_state'] = None
        self.QCQB123_params['Readout_circ'] = None


    def do_QuantumCircuitQB1234(self, qcQB1234, initState=FourQstates.State0000, readoutCirc=FourQReadoutCircs.IIIZ1, sweeps=100e3):
        self.QCQB1234_sweeps = sweeps
        self.QCQB1234_params['NV_Cpi_len'] = self.calib_params['rabi_period_LowPower']/2
        self.QCQB1234_params['NV_Cpi_freq1'] = self.calib_params['res_freq']
        self.QCQB1234_params['RF_freq0'] = self.calib_params['RF_freq0']
        self.QCQB1234_params['RF_freq1'] = self.calib_params['RF_freq1']
        self.QCQB1234_params['RF_pi'] = (self.calib_params['nucrabi_RFfreq0_period'] + self.calib_params['nucrabi_RFfreq1_period'])/4
        self.QCQB1234_params['Initial_state'] = initState
        self.QCQB1234_params['Readout_circ'] = readoutCirc
        self.QCQB1234_params['gate_operations'] = ", ".join([f"{gate.name}({gate.param})[{gate.qubit}]"  for gate in qcQB1234])
        self.sequence_generator_logic.delete_ensemble('quantumcircuitQB1234')
        self.sequence_generator_logic.delete_block('quantumcircuitQB1234')
        self.pulsed_master_logic.generate_predefined_sequence('QuantumCircuitQB1234', self.QCQB1234_params)

        self._executePulsedMeasurement('quantumcircuitQB1234', self.QCQB1234_sweeps)
        time.sleep(1)
        self.pulsed_master_logic.save_measurement_data(tag = self.POI_name + '_QCQB1234_'
                                                + '_Initstate_' + self.QCQB1234_params['Initial_state'].name
                                                + '_Readcirc_' + self.QCQB1234_params['Readout_circ'].name + '_'
                                                + '_'.join([f"{gate.name}QB{gate.qubit}" for gate in qcQB1234]),
                                                with_error=True)
        # workaround to prevent exception in yaml-dump of status variables during qudi shutdown
        # (cannot handle enum type behind an rpyc-netref, exception raised in "represent_undefined" in ruamel/yaml/representer.py)
        self.QCQB1234_params['Initial_state'] = None
        self.QCQB1234_params['Readout_circ'] = None


    def run_quantum_circuit(self, ciruit, initState=TQstates.State00, sweeps=100e3):
        try:
            if isinstance(initState, FourQstates):
                basisStates = FourQstates
                readoutCircs = FourQReadoutCircs
                circFunc = self.do_QuantumCircuitQB1234
                populFunc = self.calcFourQPopulations
            elif isinstance(initState, ThrQstates):
                basisStates = ThrQstates
                readoutCircs = ThrQReadoutCircs
                circFunc = self.do_QuantumCircuitQB123
                populFunc = self.calcThrQPopulations
            else:
                basisStates = TQstates
                readoutCircs = TQReadoutCircs
                circFunc = self.do_QuantumCircuitQB12
                populFunc = self.calcTQPopulations
            expectation_values = {}
            #fig, axs = plt.subplots(len(readStates), 1, sharex=True, figsize=(6, 1.5*len(readStates)))
            #fig.tight_layout(pad=2.5)
            for ind, readoutCirc in enumerate(readoutCircs):
                print(f"running readout circuit {ind+1} of {len(readoutCircs)} ...")
                circFunc(ciruit, initState=initState, readoutCirc=readoutCirc, sweeps=sweeps)
                tData = netobtain (self.pulsed_measurement_logic.signal_data[0])
                sigData = self.getPopulationFromCounts( netobtain(self.pulsed_measurement_logic.signal_data[1]) )
                errData = netobtain(self.pulsed_measurement_logic.measurement_error[1]) / (
                            2 * self.calib_params['rabi_amplitude'])
                result_dict = self.pulsed_measurement_logic.do_fit('Sine')
                tFit, sigFit = self.getFitFromNormalizedCounts(tData, sigData)
                expectation_values[readoutCirc.value] = sigFit[0]
                plt.figure(figsize=(5, 1.2))
                plt.errorbar(tData, sigData, yerr=errData, fmt='o', markersize=4)
                plt.plot(tFit, sigFit, color='tab:blue')
                plt.grid()
                plt.yticks([0, 0.5, 1.0])
                plt.ylim([-0.3, 1.3])
                plt.title(f'measurement data for readout circuit {ind+1}', fontsize=10)
                plt.ylabel('norm. counts')
                plt.xlabel('Rabi driving time (s)')
                plt.show()

            populations = populFunc(expectation_values)
            optimizerResult = optimize.minimize(likelihoodFuncPopul,
                                                tParamsOfPopul(populations),
                                                args=measExpValPopul(populations))
            populationsPhys = populP(optimizerResult.x)
            self.saveMeasurementResult([{'sweeps': sweeps}, populationsPhys.tolist()])

            fig = plt.figure(figsize=(len(populationsPhys), 5))
            plt.bar( np.arange(1, len(populationsPhys)+1), populationsPhys, width=0.6)
            fig.axes[0].set_xticks( np.arange(1, len(populationsPhys)+1) )
            fig.axes[0].set_xticklabels( [state.value for state in basisStates] )
            plt.grid(axis='y')
            plt.ylim((-0.05, 1.05))
            plt.title(f'measurement outcome after {len(readoutCircs)} x {sweeps:,.0f} runs', fontsize=10)
            plt.xlabel('state')
            plt.ylabel('population')
            plt.savefig(xq1i.measResFilePrefix + f"{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf", bbox_inches='tight')
            plt.show()

            # countsDark = xq1i.calib_params['rabi_offset'] - xq1i.calib_params['rabi_amplitude']
            # raw_populations = np.array( [(float(ppl)-countsDark)/(2*xq1i.calib_params['rabi_amplitude']) for ppl in pulsed_measurement_logic.signal_data[1]] )
            # populations = (1 - np.sum(raw_populations))/4 + raw_populations

            # fig, ax = plt.subplots()
            # ax.bar(['00', '01', '10', '11'], populations,
            #        yerr = [float(err)/(2*xq1i.calib_params['rabi_amplitude']) for err in pulsed_measurement_logic.measurement_error[1]],
            #        width=0.6)
            # ax.set_xlabel('state')
            # ax.set_ylabel('population')
            # ax.set_ylim( (-0.3, 1.1) )
            # plt.show()
        except KeyboardInterrupt:
            self._interruptPulsedMeasurement()
            self.QCQB12_params['Initial_state'] = None
            self.QCQB12_params['Readout_circ'] = None
            self.QCQB123_params['Initial_state'] = None
            self.QCQB123_params['Readout_circ'] = None
            self.QCQB1234_params['Initial_state'] = None
            self.QCQB1234_params['Readout_circ'] = None
            plt.close()
            print('\033[91m' + 'WARNING: User interrupt of circuit execution, measurement sequence stopped.' + '\033[0m')


    def run_quantum_circuit_qst(self, ciruit, initState=TQstates.State00, sweeps=100e3):
        try:
            expectation_values = {}
            for ind, readoutCirc in enumerate(TQQSTReadoutCircs):
                print(f"running readout circuit {ind + 1} of {len(TQQSTReadoutCircs)} ...")
                self.do_QuantumCircuitQstQB13(ciruit, initState=initState, readoutCirc=readoutCirc, sweeps=sweeps)
                tData = netobtain(self.pulsed_measurement_logic.signal_data[0])
                sigData = self.getPopulationFromCounts( netobtain(self.pulsed_measurement_logic.signal_data[1]) )
                errData = netobtain(self.pulsed_measurement_logic.measurement_error[1]) / (
                        2 * self.calib_params['rabi_amplitude'])
                result_dict = self.pulsed_measurement_logic.do_fit('Sine')
                tFit, sigFit = self.getFitFromNormalizedCounts(tData, sigData, type='no14N')
                expectation_values[readoutCirc.value] = sigFit[0]
                # if readoutCirc.name in ['ZI', 'ZZ']:
                #     expectation_values[readoutCirc.value] = 0.9
                # elif readoutCirc.name in ['IZ']:
                #     expectation_values[readoutCirc.value] = 0.1
                # else:
                #     expectation_values[readoutCirc.value] = 0.5 + np.random.rand()/10
                plt.figure(figsize=(5, 1.2))
                plt.errorbar(tData, sigData, yerr=errData, fmt='o', markersize=4)
                plt.plot(tFit, sigFit, color='tab:blue')
                plt.grid()
                plt.yticks([0, 0.5, 1.0])
                plt.ylim([-0.3, 1.3])
                plt.title(f'measurement data for readout circuit {readoutCirc.name}', fontsize=10)
                plt.ylabel('norm. counts')
                plt.xlabel('Rabi driving time (s)')
                plt.show()

            rhoPhys = self.calcTQDensMat(expectation_values)
            psiInit = tensor(Qobj([[1],[-1j]])/np.sqrt(2), Qobj([[1],[-1j]])/np.sqrt(2))
            rhoTheory = (gate_CeROTn()*psiInit) * (gate_CeROTn()*psiInit).dag()
            fidTheoryPhys = metrics.fidelity(rhoTheory, rhoPhys) * 100
            self.saveMeasurementResult([{ 'sweeps': sweeps, 'real': np.real(rhoPhys.full()).tolist(), 'imag': np.imag(rhoPhys.full()).tolist() }])

            tickLabels = ["00", "01", "10", "11"]
            x = np.arange(0.25, rho.shape[0])
            y = np.arange(0.25, rho.shape[1])
            xx, yy = np.meshgrid(x, y, indexing='ij')
            boxDelta = 0.05
            fig = plt.figure(figsize=(10, 7))
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            for ax, rho, rhoTh in zip([ax1, ax2], [np.real(rhoPhys.full()), np.imag(rhoPhys.full())],
                                      [np.real(rhoTheory.full()), np.imag(rhoTheory.full())]):
                for k in range(len(xx.ravel())):
                    ax.bar3d(xx.ravel()[k], yy.ravel()[k], np.zeros_like(xx.ravel())[k], 0.5, 0.5, rho.ravel()[k],
                             color='tab:blue', alpha=1, zorder=1000)
                    ax.bar3d(xx.ravel()[k] - boxDelta, yy.ravel()[k] - boxDelta, np.zeros_like(xx.ravel())[k],
                             0.5 + 2 * boxDelta, 0.5 + 2 * boxDelta, rhoTh.ravel()[k],
                             edgecolor='black', linewidth=1.2, alpha=0)
                ax.view_init(elev=20, azim=-35, roll=0)
                ax.set_box_aspect((4, 4, 4), zoom=0.8)
                ax.set_xlim([0, rho.shape[0]])
                ax.set_ylim([0, rho.shape[0]])
                ax.set_zlim([-0.6, 1.0])
                ax.set_xticks(np.arange(0.5, rho.shape[0]), tickLabels)
                ax.set_yticks(np.arange(0.5, rho.shape[0]), tickLabels)
                ax.set_zticks(np.arange(-0.6, 1.0, 0.3))
                ax.set_xlabel(r"$m$")
                ax.set_ylabel(r"$n$")
                ax1.set_zlabel(r"$\mathrm{Re}(\rho_{mn})$")
                ax2.set_zlabel(r"$\mathrm{Im}(\rho_{mn})$")
            fig.suptitle( rf"measurement outcome after {len(TQQSTReadoutCircs)} x {sweeps:,.0f} runs, "
                          rf"$F(\rho_\mathrm{{th}}, \rho_\mathrm{{ph}})=${fidTheoryPhys:.1f}%",
                          fontsize=10, y=0.72)
            plt.savefig(xq1i.measResFilePrefix + f"{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        bbox_inches='tight')
            plt.show()
        except KeyboardInterrupt:
            self._interruptPulsedMeasurement()
            self.QCQSTQB13_params['Initial_state'] = None
            self.QCQSTQB13_params['Readout_circ'] = None
            plt.close()
            print('\033[91m' + 'WARNING: User interrupt of circuit execution, measurement sequence stopped.' + '\033[0m')


    def gate(self, name, qubit=0, param=0):
        return xq1iGate(self.sequence_generator_logic, name=name, qubit=qubit, param=param)


    def getPopulationFromCounts(self, counts):
        return (counts - (self.calib_params['rabi_offset']-self.calib_params['rabi_amplitude'])) / (2*self.calib_params['rabi_amplitude'])


    def getFitFromNormalizedCounts(self, tData, sigData, type='standard'):
        p0 = [ np.average(sigData), (sigData.max()-sigData.min())/2, 45 ]
        period = self.calib_params['rabi_period_HighPower'] if type=='no14N' else self.calib_params['rabi_period_LowPower']
        fit_func = lambda t, *p: p[0] + p[1]*np.cos( 2*np.pi/period*t + np.pi*p[2]/180 )
        fitparams, fitparams_covariance = optimize.curve_fit(fit_func, tData, sigData, p0=p0, maxfev=1000)
        match type:
            case 'no14N':
                fitparams[0] = 0.5
                if abs(fitparams[1]) > 0.5:
                    fitparams[1] = np.sign(fitparams[1]) * 0.5
            case _:
                if fitparams[0] > 0.5:
                    fitparams[0] = 0.5
                elif fitparams[0] < 0:
                    fitparams[0] = 0
                if abs(fitparams[1]) > 0.5:
                    fitparams[1] = np.sign(fitparams[1])*0.5
                if abs(fitparams[1]) > fitparams[0]:
                    fitparams[0] = abs(fitparams[1])
        tFit = np.linspace(0, tData[-1], 100)
        sigFit = fit_func(tFit, *fitparams)
        return tFit,sigFit


    def calcTQPopulations(self, vals):
        id_diag = np.array([1.0, 1.0])
        z_diag = np.array([1.0, -1.0])
        basisOne = [id_diag, z_diag]
        basisTwo = [ np.kron(op1, op2) for op1 in basisOne for op2 in basisOne ]
        expect_vals = [
            2*( vals[TQReadoutCircs.P00.value] + vals[TQReadoutCircs.P10.value] ) - 1,
            2*( vals[TQReadoutCircs.P00.value] + vals[TQReadoutCircs.P01.value] ) - 1,
            1 - 2*( vals[TQReadoutCircs.P01.value] + vals[TQReadoutCircs.P10.value] )
        ]
        populations = np.ones( len(TQstates), dtype=np.float64 ) / 4
        for op,expval in zip(basisTwo[1:],expect_vals):
            populations += expval*op / 4
        return populations


    def calcTQDensMat(self, vals):
        readOutDict = {"II": 1,
                       "IX": -1,
                       "IY": -1,
                       "IZ": -1,
                       "XI": -1,
                       "XX": 1,
                       "XY": -1,
                       "XZ": 1,
                       "YI": 1,
                       "YX": 1,
                       "YY": -1,
                       "YZ": 1,
                       "ZI": 1,
                       "ZX": 1,
                       "ZY": -1,
                       "ZZ": 1,
                       }
        basisSetOneQb = [qeye(2), sigmax(), sigmay(), sigmaz()]
        basisSetTwoQb = [tensor(op1, op2) for op1 in basisSetOneQb for op2 in basisSetOneQb]
        rho = basisSetTwoQb[0]/4
        for readOut,op in zip(TQQSTReadoutCircs, basisSetTwoQb[1:]):
            rho += readOutDict[readOut.name]*(2*vals[readOut.value]-1)/4 * op
        optimizerResult = optimize.minimize( likelihoodFunc, tParamsOfRho(rho), args=measExpVal(rho) )
        rhoPhys = rhoP(optimizerResult.x)
        return rhoPhys


    def calcThrQPopulations(self, vals):
        id_diag = np.array([1.0, 1.0])
        z_diag = np.array([1.0, -1.0])
        basisOne = [id_diag, z_diag]
        basisThree = [ np.kron(op1, np.kron(op2, op3)) for op1 in basisOne for op2 in basisOne for op3 in basisOne ]
        expect_vals = [
            -2*( vals[ThrQReadoutCircs.IIZ1.value] + vals[ThrQReadoutCircs.IIZ2.value] ) + 1,
             2*( vals[ThrQReadoutCircs.IZI1.value] + vals[ThrQReadoutCircs.IZI2.value] ) - 1,
            -2*( vals[ThrQReadoutCircs.IIZ1.value] + vals[ThrQReadoutCircs.IZZ2.value] ) + 1,
             2*( vals[ThrQReadoutCircs.IZI1.value] + vals[ThrQReadoutCircs.ZII2.value] ) - 1,
             2*( vals[ThrQReadoutCircs.ZIZ1.value] + vals[ThrQReadoutCircs.ZIZ2.value] ) - 1,
             2*( vals[ThrQReadoutCircs.IZI1.value] + vals[ThrQReadoutCircs.ZZI2.value] ) - 1,
             2*( vals[ThrQReadoutCircs.ZIZ1.value] + vals[ThrQReadoutCircs.ZZZ2.value] ) - 1
        ]
        populations = np.ones( len(ThrQstates), dtype=np.float64 ) / 8
        for op,expval in zip(basisThree[1:],expect_vals):
            populations += expval*op / 8
        return populations


    def calcFourQPopulations(self, vals):
        id_diag = np.array([1.0, 1.0])
        z_diag = np.array([1.0, -1.0])
        basisOne = [id_diag, z_diag]
        basisFour = [ np.kron(op1, np.kron(op2, np.kron(op3, op4))) for op1 in basisOne for op2 in basisOne for op3 in basisOne for op4 in basisOne ]
        expect_vals = [
            -2*( vals[FourQReadoutCircs.IIIZ1.value] + vals[FourQReadoutCircs.IIIZ2.value] ) + 1,
            -2*( vals[FourQReadoutCircs.IIZI1.value] + vals[FourQReadoutCircs.IIZI2.value] ) + 1,
            -2*( vals[FourQReadoutCircs.IIZZ1.value] + vals[FourQReadoutCircs.IIZZ2.value] ) + 1,
             2*( vals[FourQReadoutCircs.IZII1.value] + vals[FourQReadoutCircs.IZII2.value] ) - 1,
            -2*( vals[FourQReadoutCircs.IIIZ1.value] + vals[FourQReadoutCircs.IZIZ2.value] ) + 1,
            -2*( vals[FourQReadoutCircs.IIZI1.value] + vals[FourQReadoutCircs.IZZI2.value] ) + 1,
            -2*( vals[FourQReadoutCircs.IIZZ1.value] + vals[FourQReadoutCircs.IZZZ2.value] ) + 1,
             2*( vals[FourQReadoutCircs.IZII1.value] + vals[FourQReadoutCircs.ZIII2.value] ) - 1,
             2*( vals[FourQReadoutCircs.ZIIZ1.value] + vals[FourQReadoutCircs.ZIIZ2.value] ) - 1,
             2*( vals[FourQReadoutCircs.ZIZI1.value] + vals[FourQReadoutCircs.ZIZI2.value] ) - 1,
             2*( vals[FourQReadoutCircs.ZIZZ1.value] + vals[FourQReadoutCircs.ZIZZ2.value] ) - 1,
             2*( vals[FourQReadoutCircs.IZII1.value] + vals[FourQReadoutCircs.ZZII2.value] ) - 1,
             2*( vals[FourQReadoutCircs.ZIIZ1.value] + vals[FourQReadoutCircs.ZZIZ2.value] ) - 1,
             2*( vals[FourQReadoutCircs.ZIZI1.value] + vals[FourQReadoutCircs.ZZZI2.value] ) - 1,
             2*( vals[FourQReadoutCircs.ZIZZ1.value] + vals[FourQReadoutCircs.ZZZZ2.value] ) - 1,
        ]
        populations = np.ones( len(FourQstates), dtype=np.float64 ) / 16
        for op,expval in zip(basisFour[1:],expect_vals):
            populations += expval*op / 16
        return populations
