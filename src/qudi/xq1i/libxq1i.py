from collections import OrderedDict
import numpy as np
import time
import datetime
import os
import json
from qudi.logic.pulsed.predefined_generate_methods.qb12_control_methods import xq1iGate
from qudi.logic.pulsed.predefined_generate_methods.qb12_control_methods import TQstates


def covertNetrefToNumpyArray(data):
    return np.array([float(_) for _ in data])


class xq1i:
    POI_name = 'NV_011'
    calibParamsFilePrefix = os.path.join('./', 'calib_params', 'calib_params_')
    microwave_amplitude_LowPower = 0.001
    microwave_amplitude_HighPower = 0.1
    nucrabi_RFfreq0_amp = 0.02
    nucrabi_RFfreq1_amp = 0.025

    def __init__(self, pulsed_master_logic, pulsed_measurement_logic, sequence_generator_logic):
        self.pulsed_master_logic = pulsed_master_logic
        self.pulsed_measurement_logic = pulsed_measurement_logic
        self.sequence_generator_logic = sequence_generator_logic

        self.calib_params = OrderedDict()
        self.calib_params['res_freq'] = 1.4482e9
        self.calib_params['RF_freq0'] = 5.0962e6
        self.calib_params['RF_freq1'] = 2.9256e6
        self.calib_params['rabi_period_LowPower'] = 2.0e-6
        self.calib_params['rabi_period_HighPower'] = 33.0e-09
        self.calib_params['rabi_offset'] = 1.181
        self.calib_params['rabi_amplitude'] = 0.139
        self.calib_params['nucrabi_RFfreq0_period'] = 72.46e-6
        self.calib_params['nucrabi_RFfreq1_period'] = 73.53e-6
        self.loadCalibParams()

        self.generate_params = OrderedDict()
        self.generate_params['laser_channel'] = 'd_ch1'
        self.generate_params['sync_channel'] = 'd_ch2'
        self.generate_params['laser_length'] = 3.2e-06
        self.generate_params['laser_delay'] = 1.0e-9
        self.generate_params['wait_time'] = 1.5e-06
        self.generate_params['microwave_frequency'] = 1442.27e6
        self.generate_params['microwave_amplitude'] = 0.1
        self.generate_params['rabi_period'] = 30.50e-09

        self.rabi_params = self.pulsed_master_logic.generate_method_params['rabi']
        self.rabi_params['name'] = 'rabi'
        self.rabi_params['tau_start'] = 2.0e-9
        self.rabi_params['tau_step'] = 2.0e-9
        self.rabi_params['laser_on'] = 20.0e-9
        self.rabi_params['laser_off'] = 60.0e-9
        self.rabi_params['num_of_points'] = 50
        self.rabi_params['delay_time'] = 0.0e-6
        self.rabi_sweeps = 100000

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
        self.pulsedODMR_params['num_of_points'] = 100
        self.pulsedODMR_sweeps = 100000

        self.nucspect_params = self.pulsed_master_logic.generate_method_params['NucSpect']
        self.nucspect_params['name'] = 'nucspect'
        self.nucspect_params['NV_pi'] = False
        self.nucspect_params['RF_pi'] = False
        self.nucspect_params['RF_freq'] = 5.05e6
        self.nucspect_params['RF_amp'] = 0.00
        self.nucspect_params['RF_pi_len'] = 40.0e-9
        self.nucspect_params['freq_start'] = 5.05e6
        self.nucspect_params['freq_step'] = 2.0e3
        self.nucspect_params['spect_amp'] = 0.02
        self.nucspect_params['spect_pi'] = 40000e-9
        self.nucspect_params['num_of_points'] = 50
        self.nucspect_params['laser_on'] = 20.0e-9
        self.nucspect_params['laser_off'] = 60.0e-9
        self.nucspect_sweeps = 100000

        self.nucrabi_params = self.pulsed_master_logic.generate_method_params['NucRabi']
        self.nucrabi_params['name'] = 'nucrabi'
        self.nucrabi_params['NV_pi'] = False
        self.nucrabi_params['RF_pi'] = False
        self.nucrabi_params['RF_freq'] = 5.05e6
        self.nucrabi_params['RF_amp'] = 0.00
        self.nucrabi_params['RF_pi_len'] = 100.0e-9
        self.nucrabi_params['Nuc_rabi_freq'] = 5.06e6
        self.nucrabi_params['Nuc_rabi_amp'] = 0.02
        self.nucrabi_params['tau_start'] = 10.0e-9
        self.nucrabi_params['tau_step'] = 8.0e-6
        self.nucrabi_params['num_of_points'] = 25
        self.nucrabi_params['laser_on'] = 20.0e-9
        self.nucrabi_params['laser_off'] = 60.0e-9
        self.nucrabi_sweeps = 100000

        self.DDrfspect_params = self.pulsed_master_logic.generate_method_params['DDrf_Spect']
        self.DDrfspect_params['name'] = 'ddrf_spect'
        self.DDrfspect_params['freq'] = 5.096e6
        self.DDrfspect_params['RF_freq'] =2.56e6
        self.DDrfspect_params['RF_amp'] =0.025
        self.DDrfspect_params['cyclesf'] =7
        self.DDrfspect_params['rot_phase'] = 0
        self.DDrfspect_params['DD_order'] = 6
        self.DDrfspect_params['num_of_points'] = 10
        self.DDrfspect_params['laser_on'] = 20.0e-9
        self.DDrfspect_params['laser_off'] = 60.0e-9
        self.DDrfspect_sweeps = 100000

        self.QCQB12_params  = self.pulsed_master_logic.generate_method_params['QuantumCircuitQB12']
        self.QCQB12_params['name'] = 'quantumcircuitQB12'
        self.QCQB12_params['NV_Cpi_amp'] = self.microwave_amplitude_LowPower
        self.QCQB12_params['RF_amp0'] = self.nucrabi_RFfreq0_amp
        self.QCQB12_params['RF_amp1'] = self.nucrabi_RFfreq1_amp
        self.QCQB12_params['cyclesf'] = 7
        self.QCQB12_params['DD_N'] = 8
        self.QCQB12_params['num_of_points'] = 20
        self.QCQB12_params['laser_on'] = 20.0e-9
        self.QCQB12_params['laser_off'] = 60.0e-9
        self.QCQB12_sweeps = 600

    def saveCalibParams(self):
        filename = self.calibParamsFilePrefix + datetime.datetime.now().strftime('%Y%m%d_%H%M') + '.json'
        file = open(filename, 'w')
        json.dump(self.calib_params, file, indent=4)
        file.close()

    def loadCalibParams(self):
        newestCalibFile = sorted( os.listdir( os.path.dirname(self.calibParamsFilePrefix) ), reverse=True )[0]
        file = open( os.path.join(os.path.dirname(self.calibParamsFilePrefix), newestCalibFile), 'r' )
        self.calib_params = json.load(file)
        timeStamp = datetime.datetime.strptime( newestCalibFile,  os.path.basename(xq1i.calibParamsFilePrefix) + '%Y%m%d_%H%M' + '.json' )
        print(f"INFO: loaded calibration parameters from file '{newestCalibFile}'")
        if timeStamp + datetime.timedelta(hours=3) < datetime.datetime.now():
            print('\033[91m' + 'WARNING: Most recent calibration data is older than 3 hours. Please recalibrate.' + '\033[0m')

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


    def _executePulsedMeasurement(self, name, sweeps):
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
        self.pulsed_master_logic.toggle_pulsed_measurement(True)
        while self.pulsed_measurement_logic.module_state() != 'locked':
            time.sleep(0.5)
        user_terminated = False
        while self.pulsed_measurement_logic.elapsed_sweeps < sweeps:
            if self.pulsed_measurement_logic.module_state() != 'locked':
                user_terminated = True
                break
            time.sleep(0.5)
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
            self.rabi_params['tau_step'] = 3.4e-9
            self.rabi_params['delay_time'] = 50.0e-6
            self.rabi_params['num_of_points'] = 30
        else:
            self.generate_params['microwave_frequency'] = self.calib_params['res_freq']
            self.generate_params['microwave_amplitude'] = self.microwave_amplitude_LowPower  #value needs to be checked/maybe tuned
            self.generate_params['rabi_period'] = self.calib_params['rabi_period_HighPower']
            self.pulsed_master_logic.set_generation_parameters(self.generate_params)
            self.rabi_params['tau_start'] = 0.0e-9
            self.rabi_params['tau_step'] = 190.0e-9
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
        self.DDrfspect_params['freq'] = self.calib_params['RF_freq0']
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


    def do_QuantumCircuitQB12(self, qcQB12):
        self.QCQB12_params['NV_Cpi_len'] = self.calib_params['rabi_period_LowPower']/2
        self.QCQB12_params['NV_Cpi_freq1'] = self.calib_params['res_freq']
        self.QCQB12_params['RF_freq0'] = self.calib_params['RF_freq0']
        self.QCQB12_params['RF_freq1'] = self.calib_params['RF_freq1']
        self.QCQB12_params['RF_pi'] = (self.calib_params['nucrabi_RFfreq0_period'] + self.calib_params['nucrabi_RFfreq1_period'])/4
        self.QCQB12_params['gate_operations'] = ", ".join([f"{gate.name}({gate.param})[{gate.qubit}]"  for gate in qcQB12])
        #print(self.QCQB12_params['gate_operations'])
        self.sequence_generator_logic.delete_ensemble('quantumcircuitQB12')
        self.sequence_generator_logic.delete_block('quantumcircuitQB12')
        self.pulsed_master_logic.generate_predefined_sequence('QuantumCircuitQB12', self.QCQB12_params)

        self._executePulsedMeasurement('quantumcircuitQB12', self.QCQB12_sweeps)
        time.sleep(1)
        self.pulsed_master_logic.save_measurement_data(tag = self.POI_name + '_QCQB12_'
                                                + '_Initstate_' + self.QCQB12_params['Initial_state'].name
                                                + '_Readstate_' + self.QCQB12_params['Readout_state'].name + '_'
                                                + '_'.join([f"{gate.name}QB{gate.qubit}" for gate in qcQB12]),
                                                with_error=True)

    def gate(self, name, qubit=0, param=0):
        return xq1iGate(self.sequence_generator_logic, name=name, qubit=qubit, param=param)

    def getPopulationFromCounts(self, counts):
        return (counts - (self.calib_params['rabi_offset']-self.calib_params['rabi_amplitude'])) / (2*self.calib_params['rabi_amplitude'])
