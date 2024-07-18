# Import Dependencies
import random
from typing import List

import numpy
import math
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline


class IoT_Device:
    def __init__(self, X: int, Y: int, devType: int, long: float, lat: float, clusterheadNum: int = None):
        self.indX = X
        self.indY = Y
        self.lat = lat
        self.long = long
        self.mean_AoI = 0

        # Communication Specifications
        self._comms = {
            "LoRa_Max_Distance_m": 5000,
            "LoRa_Bit_Rate_bit/s": 24975,
            "LoRa_Current_A": 800,
            "LoRa_Voltage_V": 3.7,
            "LoRa_Power_W": 0.0008 * 3.7,

            "AmBC_Max_Distance_m": 500,
            "AmBC_Bit_Rate_bit/s": 1592,
            "AmBC_Power_W": 0.00352 / 1000,
            "AmBC_Voltage_V": 3.3,
            "AmBC_Current_A": 1.2
        }

        self.spctrlLow = 0  # Spectral bandwidth for power calculations
        self.spctrlHigh = numpy.inf
        self.solar_powered = True

        if devType == 1:
            self.type = 1
            self.typeStr = "Wireless Sensor"
            self.head = None
            self.queue = None

            self.azimuth = 0  # 0 for North
            self.tilt = 0  # 0 for no horizontal tilt
            self.h = 0

            # Sensor Specifications
            self.max_col_rate = 64_000  # 64 bits per sample
            self.sample_freq = 15  # 15 minutes between sampling
            self.sample_len = 30  # 30 sec sample duration
            self.max_data = 256_000  # 256 kB maximum data storage
            self.stored_data = random.randint(0, 256000)
            self.sens_pow = 2.2  # 2.2 W power consumption
            self.sens_amp = self.sens_pow / self._comms.get("AmBC_Voltage_V") * 1_000

            # Battery Specifics
            self.solarArea = 20 * 40  # 20 mm x 40 mm
            self._C = 1  # F
            self.max_energy = 1_280  # Ah
            self.charge_rate = 2.56  # A/h
            self.discharge_rate = 0.08  # A/h
            self.stored_energy = round(self.max_energy * 1_000)

        else:
            self.type = 2
            self.sens_table = None
            self.headSerial = clusterheadNum
            self.typeStr = "Clusterhead"

            self.azimuth = 180
            self.tilt = 45
            self.h = 0

            # CH Specs
            self.max_data = 25_000_000
            self.stored_data = random.randint(1_028_000, 25_000_000)

            self.solarArea = 200 * 400  # 20 cm x 40 cm
            self._C = 3200  # F (Battery Supported)
            self.max_energy = 1_510  # Ah
            self.charge_rate = 3.02  # A/s
            self.discharge_rate = 0.755  # s
            self.stored_energy = round(self.max_energy * 1_000)

    def reset(self):
        self.indX = random.randint(0, 100)
        self.indY = random.randint(0, 100)
        self.mean_AoI = 0
        self.stored_energy = round(self.max_energy * 1_000)
        if self.type == 1:
            self.stored_data = random.randint(128_000, 256_000)
        else:
            self.stored_data = random.randint(1_028_000, 25_000_000)

    # Call location
    def get_indicies(self):
        return self.indX, self.indY

    def set_head(self, head: int, queue: int):
        self.head = head
        self.queue = queue

    def harvest_data(self, step):
        if self.solar_powered:
            if step % self.sample_freq == 0:
                self.stored_data = min(self.stored_data + self.max_col_rate, self.max_data)
                return True
            else:
                return False
        elif self.stored_energy > round(self.sens_amp * 30):
            if step % self.sample_freq == 0:
                self.stored_data = min(self.stored_data + self.max_col_rate, self.max_data)
                self.stored_energy -= round(self.sens_amp * 30)
                return True
            else:
                return False
        else:
            return False

    def harvest_energy(self, alpha, env, step):
        spectra = env.getIrradiance(self.lat, self.long, self.tilt, self.azimuth, step)
        interference = env.getInterference(self.indX, self.indY, self.type)
        f = InterpolatedUnivariateSpline(spectra['wavelength'], spectra['poa_global'])
        powDensity = alpha * (1 - interference) * f.integral(self.spctrlLow, self.spctrlHigh)
        power = (powDensity * self.solarArea / (1000 * 1000)) * 1_000_000

        if power > 0.0:
            self.stored_energy += round(power/self._comms.get("LoRa_Voltage_V") * 1_000_000)
            self.solar_powered = True
        else:
            self.solar_powered = False

    # Uploading data from a sensor
    def ws_upload_data(self, X, Y):
        if math.sqrt(pow((self.indX - X), 2) + pow((self.indY - Y), 2)) <= \
                self._comms.get("AmBC_Max_Distance_m"):
            return min(self._comms.get("AmBC_Bit_Rate_bit/s") * 30, self.stored_data)
        else:
            return -1

    # Clusterhead-Specific Tasks
    def set_sensor_data(self, sens_list: list):
        sens_active = [True] * (len(sens_list))
        sens_aoi = [0] * (len(sens_list))
        self.sens_table = pd.concat(
            [pd.DataFrame(sens_list), pd.DataFrame(sens_active), pd.DataFrame(sens_aoi)],
            axis=1
        )
        self.sens_table.rename(
            columns={0: "Sensor", 1: "Connection_Status", 2: "AoI"},
            inplace=True
        )

    def ch_download(self, step):
        rotations = math.ceil(len(self.sens_table.index) / 2)
        rotation = step % rotations
        sensor = rotation * 2
        activeChannels = []
        sensor1 = self.sens_table.iloc[sensor, 0]
        activeChannels.append(max(0, sensor1.ws_upload_data(self.indX, self.indY)))

        if rotation < (rotations - 1) or len(self.sens_table.index) % 2 == 0:
            sensor2 = self.sens_table.iloc[sensor + 1, 0]
            activeChannels.append(max(0, sensor2.ws_upload_data(self.indX, self.indY)))

        totalChannels = 0
        for channel in range(len(activeChannels)):
            if activeChannels[channel] > 0:
                self.sens_table.iloc[sensor + channel, 1] = True
                self.sens_table.iloc[sensor + channel, 2] = step
                self.stored_data += activeChannels[channel]
                totalChannels += 1
            else:
                self.sens_table.iloc[sensor + channel, 1] = False

        self.stored_energy -= round(self._comms.get("LoRa_Current_A") * 30 * totalChannels)
        self.mean_AoI = self.sens_table.iat[0, 2]
        for sens in range(len(self.sens_table) - 1):
            if step - self.sens_table.iat[sens + 1, 2] > step - self.mean_AoI:
                self.mean_AoI += self.sens_table.iat[sens + 1, 2]
        self.mean_AoI = round(self.mean_AoI / len(self.sens_table))

    def ch_upload(self, X: int, Y: int):
        if self.solar_powered:
            if math.sqrt(pow((self.indX - X), 2) + pow((self.indY - Y), 2)) <= \
                    self._comms.get("LoRa_Max_Distance_m"):

                if self.stored_data > 0:
                    self.stored_data -= min(self._comms.get("LoRa_Bit_Rate_bit/s") * 56, self.stored_data)
                    sent_data = min(self._comms.get("LoRa_Bit_Rate_bit/s") * 56, self.stored_data)

                    return sent_data, self.mean_AoI

                else:
                    return 0, self.mean_AoI

            else:
                return -1, self.mean_AoI
        elif self.stored_energy > round(self._comms.get("LoRa_Current_A") * 60):
            if math.sqrt(pow((self.indX - X), 2) + pow((self.indY - Y), 2)) <= \
                    self._comms.get("LoRa_Max_Distance_m"):

                if self.stored_data > 0:
                    self.stored_data -= min(self._comms.get("LoRa_Bit_Rate_bit/s") * 56, self.stored_data)
                    sent_data = min(self._comms.get("LoRa_Bit_Rate_bit/s") * 56, self.stored_data)

                    self.stored_energy -= round(self._comms.get("LoRa_Current_A") * 60)
                    return sent_data, self.mean_AoI

                else:
                    self.stored_energy -= round(self._comms.get("LoRa_Current_A") * 60)
                    return 0, self.mean_AoI

            else:
                self.stored_energy -= round(self._comms.get("LoRa_Current_A") * 60)
                return -1, self.mean_AoI
        else:
            return -1, self.mean_AoI

    def charge_time(self, X: int, Y: int, charge):
        if round(self.indX) == round(X) and round(self.indY) == round(Y):
            if self.solar_powered and charge:
                return 60.0
            elif self.stored_energy > round((6.8 / (2.5 * 60) * 1_000_000)) and charge:
                self.stored_energy -= round((6.8 / (2.5 * 60) * 1_000_000))
                return 60.0
            else:
                return 0
        else:
            return 0

    def get_dest(self, state, full_state, model, step, _=None):
        if self.stored_data > self.max_data * 0.20:
            return False, False, self, _, state, _, self.headSerial, _

        for CH in range(len(full_state) - 1):
            if full_state.iloc[CH + 1, 3] == 0:
                return False, True, full_state.iloc[CH + 1, 0], _, state, _, self.headSerial, _

        sensMapping: List[List[int]] = [[0] * 3 for _ in range(5)]
        count = 0
        for sens in range(len(self.sens_table)):
            if not (self.sens_table.iloc[sens, 1]) and (count < 5):
                sensMapping[count][0], sensMapping[count][1], sensMapping[count][2] = sens, \
                    math.sqrt(pow((self.indX - self.sens_table.iloc[sens, 0].indX), 2) + \
                              pow((self.indY - self.sens_table.iloc[sens, 0].indY), 2)), (-5 + count)
                state[sensMapping[count][2]][1], state[sensMapping[count][2]][2] = sensMapping[count][1], \
                    self.sens_table.iloc[sens, 2]
            count += 1

        action = model.act(state)

        if action < (len(full_state) - 1):
            return True, True, full_state.iloc[action + 1, 0], _, state, _, action, _
        else:
            sensor = self.sens_table.iloc[sensMapping[action - len(full_state) + 1][0], 0]
            self.sens_table.iloc[sensMapping[action - len(full_state) + 1][0], 2] = step
            state1 = state
            for Iter in range(5):
                state[len(full_state) + Iter][1], state[len(full_state) + Iter][2] = 0, 0

            action2 = model.act(state) % (len(full_state) - 1)
            return True, True, sensor, full_state.iloc[action2 + 1, 0], state1, state, action, action2
