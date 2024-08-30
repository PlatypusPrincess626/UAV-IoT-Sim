# Import Dependencies
import random
from typing import List

import numpy
import math
import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline


class IoT_Device:
    def __init__(self, X: int, Y: int, devType: int, long: float, lat: float, clusterheadNum: int = None):
        self.indX = X
        self.indY = Y
        self.lat = lat
        self.long = long
        self.max_AoI = 0

        # Communication Specifications
        self._comms = {
            "LoRa_Max_Distance_m": 5000,
            "LoRa_Bit_Rate_bit/s": 24975,
            "LoRa_Current_A": 4200,  # micro-amps transmitting
            "LoRa_Voltage_V": 3.7,
            "LoRa_Power_W": 0.020,  # transmitting
            "LoRa_Upkeep_W": 0.004,  # static
            "Lora_Upkeep_A": 1200,  # micro-amps

            "AmBC_Max_Distance_m": 500,
            "AmBC_Bit_Rate_bit/s": 1592,
            "AmBC_Power_W": 0.00259,  # upkeep
            "AmBC_Voltage_V": 3.3,
            "AmBC_Current_A": 785  # micro-amps upkeep
        }

        # Total amount for each device/ clusterhead

        self.spctrlLow = 0  # Spectral bandwidth for power calculations
        self.spctrlHigh = numpy.inf
        self.solar_powered = True
        self.cpu_pow = 3.7  # microwatts/ 2 micro Joule
        self.cpu_amps = 1_000  # micro-amps

        if devType == 1:
            self.type = 1
            self.typeStr = "Wireless Sensor"
            self.head = None
            self.queue = None

            self.azimuth = 0  # 0 for North
            self.tilt = 0  # 0 for no horizontal tilt
            self.h = 0

            # Sensor Specifications
            self.max_col_rate = 16  # 16 bits / 2 bytes per sample
            self.sample_freq = 0.25  # 15 seconds between sampling
            self.sample_len = 1  # 30 sec sample duration
            self.sample_rate_per_s = 300
            self.bits_per_min = round(self.max_col_rate * self.sample_len * self.sample_rate_per_s / self.sample_freq)

            self.max_data = 256_000  # 256 kB maximum data storage
            self.reset_min = round(self.max_data * 0.10)
            self.reset_max = round(self.max_data * 0.25)
            self.stored_data = random.randint(self.reset_min, self.reset_max)
            self.sens_pow = 2.2  # 2.2 W power consumption
            self.sens_amp = self.sens_pow / self._comms.get("AmBC_Voltage_V") * 1_000

            # Battery Specifics
            self.solarArea = 0.2 * 0.4  # 20 mm x 40 mm
            self._C = 1  # F
            self.max_energy = 1_280  # Ah
            self.charge_rate = 2.56  # A/h
            self.discharge_rate = 0.08  # A/h
            self.stored_energy = round(self.max_energy * 1_000)

        else:
            self.type = 2
            self.sens_table = None
            self.active_table = None
            self.age_table = None

            self.num_sensors = 0
            self.headSerial = clusterheadNum
            self.typeStr = "Clusterhead"

            self.azimuth = 180
            self.tilt = 45
            self.h = 0

            # CH Specs
            self.max_data = 12_500_000
            self.reset_min = round(self.max_data * 0.10)
            self.reset_max = round(self.max_data * 0.25)
            self.stored_data = self.reset_max

            self.solarArea = 2 * 4  # 20 cm x 40 cm
            self._C = 3200  # F (Battery Supported)
            self.max_energy = 6_800  # Ah
            self.charge_rate = 3.02  # A/s
            self.discharge_rate = 0.755  # s
            self.stored_energy = round(self.max_energy * 1_000)

    def reset(self):
        self.max_AoI = 0
        self.stored_energy = round(self.max_energy * 1_000)
        if self.type == 1:
            self.stored_data = random.randint(0, self.reset_max)
        else:
            self.stored_data = self.reset_max
            for sens in range(self.num_sensors):
                self.active_table[sens] = True
                self.age_table[sens] = 0

    # Call location
    def get_indicies(self):
        return self.indX, self.indY

    def set_head(self, head: int, queue: int):
        self.head = head
        self.queue = queue

    def harvest_data(self, step):
        if self.solar_powered:
            self.stored_data = min(self.bits_per_min, self.max_data)
            return True

        elif self.stored_energy > round(self.sens_amp * 4):
            self.stored_data = min(self.bits_per_min, self.max_data)
            self.stored_energy -= round(self.sens_amp * 4)
            return True

        else:
            return False

    def harvest_energy(self, alpha, env, step):
        spectra = env.getIrradiance(self.lat, self.long, self.tilt, self.azimuth, step)
        # interference = env.getInterference(self.indX, self.indY, self.type)
        if self.type == 1:
            interference = random.random()
        else:
            interference = 1


        f = InterpolatedUnivariateSpline(spectra['wavelength'], spectra['poa_global'])
        powDensity = f.integral(self.spctrlLow, self.spctrlHigh)
        power = abs(alpha / 100) * interference * powDensity * self.solarArea

        if power * 1_000_000 > 0.0:
            self.stored_energy += round((power / self._comms.get("LoRa_Voltage_V")) * 1_000_000)
            self.solar_powered = True
        else:
            self.solar_powered = False

        power_upkeep = round(self.cpu_amps + self._comms.get("AmBC_Current_A"))
        if self.type == 2:
            power_upkeep += round(self._comms.get("Lora_Upkeep_A"))
        self.stored_energy -= power_upkeep

    # Uploading data from a sensor
    def ws_upload_data(self, X, Y):
        if math.sqrt(pow((self.indX - X), 2) + pow((self.indY - Y), 2)) <= \
                self._comms.get("AmBC_Max_Distance_m"):
            return min(self._comms.get("AmBC_Bit_Rate_bit/s") * 26, self.stored_data)
        else:
            return -1

    # Clusterhead-Specific Tasks
    def set_sensor_data(self, sens_list: list):
        self.num_sensors = len(sens_list)

        active_list = []
        age_list = []
        for sens in range(len(sens_list)):
            active_list.append(True)
            age_list.append(0)

        self.sens_table = pd.DataFrame(np.array(sens_list))
        self.active_table = np.array(active_list)
        self.age_table = np.array(age_list)

        self.sens_table.rename(
            columns={0: "Sensor", 1: "Connection_Status", 2: "AoI"},
            inplace=True
        )

    def ch_download(self, step):
        rotations = math.ceil(len(self.sens_table.index) / 2)
        rotation = step % rotations
        sensor = rotation * 2
        activeChannels = []
        sensor1 = self.sens_table.iat[sensor, 0]
        activeChannels.append(max(0, sensor1.ws_upload_data(self.indX, self.indY)))

        if rotation < (rotations - 1) or len(self.sens_table.index) % 2 == 0:
            sensor2 = self.sens_table.iat[sensor + 1, 0]
            activeChannels.append(max(0, sensor2.ws_upload_data(self.indX, self.indY)))

        totalChannels = 0
        for channel in range(len(activeChannels)):
            if activeChannels[channel] >= 0:
                self.active_table[sensor + channel] = True
                self.age_table[sensor + channel] = step
                self.stored_data += activeChannels[channel]
                totalChannels += 1
            else:
                self.active_table[sensor + channel] = False

        self.stored_energy -= round(self._comms.get("LoRa_Current_A"))

        # ADF 2.0
        self.max_AoI = self.age_table[0]
        for sens in range(len(self.sens_table.index) - 1):
            if self.age_table[sens] < self.max_AoI:
                self.max_AoI = self.age_table[sens]

        #     self.max_AoI += self.age_table[sens]
        # self.max_AoI = round(self.max_AoI / self.num_sensors)
        # ADF 1.0
        # self.mean_AoI = step

    def ch_upload(self, X: int, Y: int):
        if self.solar_powered:
            if math.sqrt(pow((self.indX - X), 2) + pow((self.indY - Y), 2)) <= \
                    self._comms.get("LoRa_Max_Distance_m"):

                if self.stored_data > 0:
                    self.stored_data -= min(self._comms.get("LoRa_Bit_Rate_bit/s") * 56, self.stored_data)
                    sent_data = min(self._comms.get("LoRa_Bit_Rate_bit/s") * 56, self.stored_data)

                    return sent_data, self.max_AoI

                else:
                    return 0, self.max_AoI

            else:
                return -1, self.max_AoI
        elif self.stored_energy > round(self._comms.get("LoRa_Current_A") * 60):
            if math.sqrt(pow((self.indX - X), 2) + pow((self.indY - Y), 2)) <= \
                    self._comms.get("LoRa_Max_Distance_m"):

                if self.stored_data > 0:
                    self.stored_data -= min(self._comms.get("LoRa_Bit_Rate_bit/s") * 56, self.stored_data)
                    sent_data = min(self._comms.get("LoRa_Bit_Rate_bit/s") * 56, self.stored_data)

                    self.stored_energy -= round(self._comms.get("LoRa_Current_A") * 60)
                    return sent_data, self.max_AoI

                else:
                    self.stored_energy -= round(self._comms.get("LoRa_Current_A") * 60)
                    return 0, self.max_AoI

            else:
                self.stored_energy -= round(self._comms.get("LoRa_Current_A") * 60)
                return -1, self.max_AoI
        else:
            return -1, self.max_AoI

    def charge_time(self, X: int, Y: int, charge):
        if abs(self.indX - X) < 1.0 and abs(self.indY - Y) < 1.0:
            if self.solar_powered and charge:
                return 60.0
            elif self.stored_energy > self.max_energy * 0.25 and charge:
                self.stored_energy -= round(6_800_000 / (1 * 60))
                return 60.0
            elif self.stored_energy > self.max_energy * 0.5:
                return 30.0
            else:
                return 0
        else:
            return 0

    def get_dest(self, state, full_sensor_list, model, step, no_hold, force_change, targetSerial, _=None):
        # if self.stored_data >= (self._comms.get("AmBC_Bit_Rate_bit/s") * 52 * 5) and no_hold and not force_change:
        #     # ADF 2.0
        #     return False, False, self, _, state, _, self.headSerial, _
        #     # ADF 1.0
        #     # return False, False, self, state, self.headSerial

        for CH in range(len(state) - 6):
            if state[CH + 1][2] < 1.0:
                # ADF 2.0
                return False, True, full_sensor_list.iat[CH + 1, 0], _, state, _, CH, _
                # ADF 1.0
                # return False, True, full_sensor_list.iat[CH + 1, 0], state, CH

        # ADF 2.0
        sensMapping: List[List[int]] = [[0, 0, 0] for _ in range(5)]
        count = 0
        for sens in range(self.num_sensors):
            if not (self.active_table[sens]) and (count < 5):
                sensMapping[count][0], sensMapping[count][1], sensMapping[count][2] = \
                    (sens,
                     math.sqrt(pow((self.indX - self.sens_table.iat[sens, 0].indX), 2) +
                               pow((self.indY - self.sens_table.iat[sens, 0].indY), 2)),
                     (-5 + count)
                     )

                state[sensMapping[count][2]][1], state[sensMapping[count][2]][2] = \
                    sensMapping[count][1], self.age_table[sens]

                count += 1

        action = model.act(state)
        if force_change:
            lowest = state[self.headSerial + 1][2]
            while targetSerial == action:
                for sens in range(len(state) - 1):
                    if state[sens + 1][2] < lowest:
                        lowest = state[sens + 1][2]
                        action = sens

        # ADF 2.0
        if action < (len(state) - 6):
            return True, True, full_sensor_list.iat[action + 1, 0], _, state, _, action, _
        else:
            sensor = self.sens_table.iat[sensMapping[action - len(full_sensor_list.index) + 1][0], 0]

            self.age_table[sensMapping[action - len(full_sensor_list.index) + 1][0]] = step
            state1 = state

            for Iter in range(5):
                state[len(state) - 5 + Iter][1], state[len(state) - 5 + Iter][2] = 0, 0

            action2 = model.act(state) % (len(full_sensor_list.index) - 1)
            if force_change:
                while targetSerial == action:
                    lowest = state[self.headSerial + 1][2]
                    for ch in range(len(state) - 6):
                        if state[ch + 1][2] < lowest:
                            lowest = state[ch + 1][2]
                            action2 = ch

            return True, True, sensor, full_sensor_list.iat[action2 + 1, 0], state1, state, action, action2

        # ADF 1.0
        # return True, True, full_sensor_list.iat[action + 1, 0], state, action
