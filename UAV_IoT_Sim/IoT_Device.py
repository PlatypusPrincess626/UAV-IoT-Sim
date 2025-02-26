# Import Dependencies
import random
from typing import List

import numpy
import math
import networkx as nx
import numpy as np
import pandas as pd
import copy
from scipy.interpolate import InterpolatedUnivariateSpline

class IoT_Device:
    def __init__(self, X: int, Y: int, devType: int, long: float, lat: float, clusterheadNum: int = None):
        self.indX = X
        self.indY = Y
        self.lat = lat
        self.long = long
        self.max_AoI = 0
        self.avg_AoI = 0

        # Communication Specifications
        self._comms = {
            "LoRa_Max_Distance_m": 5000,
            "LoRa_Bit_Rate_bit/s": 24975,
            "LoRa_Current_A": 4200,  # micro-amps transmitting
            "LoRa_Voltage_V": 3.7,
            "LoRa_Power_W": 0.020,  # transmitting
            "LoRa_Upkeep_W": 0.004,  # static
            "Lora_Upkeep_A": 1200,  # micro-amps

            "AmBC_Max_Distance_m": 800,
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
            self.data_table = None

            self.num_sensors = 0
            self.headSerial = clusterheadNum
            self.last_target = 0

            self.tour = []
            self.tourA = []
            self.tourB = []

            self.target_time = 0
            self.typeStr = "Clusterhead"

            self.azimuth = 180
            self.tilt = 45
            self.h = 0

            # CH Specs
            self.max_data = 12_500_000
            self.reset_min = round(self.max_data * 0.10)
            self.reset_max = round(self.max_data * 0.25)
            self.stored_data = self.reset_max
            self.contribution = 0
            self.action_p = 0

            self.solarArea = 2 * 4  # 20 cm x 40 cm
            self._C = 3200  # F (Battery Supported)
            self.max_energy = 6_800  # Ah
            self.charge_rate = 1/3  # A/s
            self.discharge_rate = 0.755  # s
            self.stored_energy = round(self.max_energy * 1_000)

            self.next_tour = []
            self.next_dist = 0.0
            self.next1_tour = []
            self.next1_dist = 0.0
            self.next2_tour = []
            self.next2_dist = 0.0

    def reset(self):
        self.last_target = 0
        self.target_time = 0
        self.tour = []
        self.tourA = []
        self.tourB = []

        self.max_AoI = 0
        self.avg_AoI = 0
        self.stored_energy = round(self.max_energy * 1_000)
        self.contribution = 0
        self.action_p = 0

        self.next_tour = []
        self.next_dist = 0.0
        self.next1_tour = []
        self.next1_dist = 0.0
        self.next2_tour = []
        self.next2_dist = 0.0

        if self.type == 1:
            self.stored_data = random.randint(0, self.reset_max)
        else:
            self.stored_data = random.randint(0, self.reset_max)
            self.data_table[0] = self.stored_data
            for sens in range(self.num_sensors):
                self.active_table[sens] = True
                self.age_table[sens] = 0
                self.data_table[sens+1] = 0

    # Call location
    def get_indicies(self):
        return self.indX, self.indY

    def harvest_data(self, step):
        if self.solar_powered:
            self.stored_data += min(self.bits_per_min, self.max_data)
            return True

        elif self.stored_energy > round(self.sens_amp * 4):
            self.stored_data += min(self.bits_per_min, self.max_data)
            self.stored_energy -= round(self.sens_amp * 4)
            return True

        else:
            return False

    def harvest_energy(self, alpha, env, step):
        spectra = env.getIrradiance(self.lat, self.long, self.tilt, self.azimuth, step)
        # interference = env.getInterference(self.indX, self.indY, self.type)
        if self.type == 1:
            interference = 1 - (random.random() * 0.8)
        else:
            interference = 1


        f = InterpolatedUnivariateSpline(spectra['wavelength'], spectra['poa_global'])
        powDensity = f.integral(self.spctrlLow, self.spctrlHigh)
        power = abs(alpha / 100) * interference * powDensity * self.solarArea

        if power * 1_000_000 > 0.0:
            if self.stored_energy < (0.8 * self.max_energy * 1_000):
                self.stored_energy += round(2 * 1_000 * self.max_energy / ((self.charge_rate) * 60))
            else:
                self.stored_energy += round(0.5 * 1_000 * (self.max_energy / ((self.charge_rate) * 60)))
            self.solar_powered = True
        else:
            self.solar_powered = False

        power_upkeep = round(self.cpu_amps + self._comms.get("AmBC_Current_A"))
        if self.type == 2:
            power_upkeep += round(self._comms.get("Lora_Upkeep_A"))
        self.stored_energy -= power_upkeep

    # Uploading data from a sensor
    def ws_upload_data(self, X, Y):
        if math.sqrt(pow((self.indX - X), 2) + pow((self.indY - Y), 2)) <= self._comms.get("AmBC_Max_Distance_m"):
            temp = min(self._comms.get("AmBC_Bit_Rate_bit/s") * 26, self.stored_data)
            self.stored_data -= temp
            return temp
        else:
            return -1

    # Clusterhead-Specific Tasks
    def set_sensor_data(self, sens_list: list):
        self.num_sensors = len(sens_list)

        active_list = []
        age_list = []
        data_list = [self.stored_data]
        for sens in range(len(sens_list)):
            active_list.append(True)
            age_list.append(0)
            data_list.append(0)

        self.sens_table = pd.DataFrame(np.array(sens_list))
        self.active_table = np.array(active_list)
        self.age_table = np.array(age_list)
        self.data_table = np.array(data_list)

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
        activeChannels.append(sensor1.ws_upload_data(self.indX, self.indY))

        if rotation < (rotations - 1) or len(self.sens_table.index) % 2 == 0:
            sensor2 = self.sens_table.iat[sensor + 1, 0]
            activeChannels.append(sensor2.ws_upload_data(self.indX, self.indY))

        totalChannels = 0
        for channel in range(len(activeChannels)):
            if activeChannels[channel] >= 0:
                self.active_table[sensor + channel] = True
                self.age_table[sensor + channel] = step
                self.data_table[sensor + channel + 1] += activeChannels[channel]
                self.stored_data += activeChannels[channel]
                totalChannels += 1
            else:
                self.active_table[sensor + channel] = False

        self.stored_energy -= round(self._comms.get("LoRa_Current_A"))

        # ADF 2.0
        self.max_AoI = self.age_table[0]
        self.avg_AoI = self.age_table[0]
        for sens in range(len(self.sens_table.index) - 1):
            self.avg_AoI += self.age_table[sens+1]
            if self.age_table[sens+1] < self.max_AoI:
                self.max_AoI = self.age_table[sens+1]
        self.avg_AoI = math.ceil(self.avg_AoI / len(self.age_table))
        # ADF 1.0
        # self.max_AoI = step

    def ch_upload(self, X: int, Y: int):
        if self.solar_powered:
            if math.sqrt(pow((self.indX - X), 2) + pow((self.indY - Y), 2)) <= self._comms.get("LoRa_Max_Distance_m"):

                if self.stored_data > 0:
                    sent_data = min(self._comms.get("LoRa_Bit_Rate_bit/s") * 56, self.stored_data)
                    self.stored_data -= sent_data
                    self.contribution += sent_data

                    return sent_data, self.max_AoI, self.avg_AoI

                else:
                    return 0, self.max_AoI, self.avg_AoI

            else:
                return -1, self.max_AoI, self.avg_AoI
        elif self.stored_energy > round(self._comms.get("LoRa_Current_A")):
            if math.sqrt(pow((self.indX - X), 2) + pow((self.indY - Y), 2)) <= self._comms.get("LoRa_Max_Distance_m"):

                if self.stored_data > 0:
                    sent_data = min(self._comms.get("LoRa_Bit_Rate_bit/s") * 56, self.stored_data)
                    self.stored_data -= min(self._comms.get("LoRa_Bit_Rate_bit/s") * 56, self.stored_data)
                    self.contribution += sent_data

                    self.stored_energy -= round(self._comms.get("LoRa_Current_A"))
                    return sent_data, self.max_AoI, self.avg_AoI

                else:
                    self.stored_energy -= round(self._comms.get("LoRa_Current_A"))
                    return 0, self.max_AoI, self.avg_AoI

            else:
                self.stored_energy -= round(self._comms.get("LoRa_Current_A"))
                return -1, self.max_AoI, self.avg_AoI
        else:
            return -1, self.max_AoI, self.avg_AoI

    def charge_time(self, X: int, Y: int, charge):
        if abs(self.indX - X) < 1.0 and abs(self.indY - Y) < 1.0:
            if self.solar_powered and charge:
                return 60.0
            elif self.stored_energy > self.max_energy * 0.25 and charge:
                self.stored_energy -= round(6_800_000 / (self.charge_rate * 60))
                return 60.0
            elif self.stored_energy > self.max_energy * 0.25:
                if not self.solar_powered:
                    self.stored_energy -= round(0.5 * 6_800_000 / (self.charge_rate * 60))
                return 30.0
            else:
                return 0
        else:
            return 0

    def get_dest(self, state, full_sensor_list, model, model_p,
                 step, p_count, targetType, targetSerial, force_out,  _=None):
        """
        Returns next destination for the UAV.

        Args:
            state:              state recieved from UAV including important metrics
            full_sensor_list:   list of sensor objects without regard to metrics
            model:              model for deciding UAV destination
            model_p:            model for UAV charging plan
            step:               the current timestep
            p_count:            time left for charging (made by previous decision)
            targetType:         flag for sensor or UAV targeting
            targetSerial:       current cluster target of UAV trajectory (should be self?)
            _:                  variable with value "None"

        Returns:

        """

        # Must check through list of sensors... Must recieve data on how much each sensor contributes.
        """Old Contribution: DATE"""
        # my_contribution = state[self.headSerial + 1][1]
        # if my_contribution > self.contribution:
        #     self.contribution = my_contribution
        #
        #     self.age_table[self.last_target] = self.target_time     # Sensors on range 1 to num_sens
        #
        #     self.max_AoI = self.age_table[0]
        #     self.avg_AoI = self.age_table[0]
        #     for sens in range(len(self.sens_table.index) - 1):
        #         self.avg_AoI += self.age_table[sens + 1]
        #         if self.age_table[sens + 1] < self.max_AoI:
        #             self.max_AoI = self.age_table[sens + 1]
        #     self.avg_AoI = math.ceil(self.avg_AoI / len(self.age_table))
        #
        #     state[self.headSerial + 1][2] = step - self.max_AoI
        #     state[self.headSerial + 1][3] = step - self.avg_AoI
        """New Contribution"""
        my_contribution = state[self.headSerial + 1][1]
        if my_contribution > self.contribution:
            self.contribution = my_contribution

            for sens in range(len(self.active_table)):
                if not self.active_table[sens]:
                    self.age_table[sens] = self.target_time     # Sensors on range 1 to num_sens

            self.max_AoI = self.age_table[0]
            self.avg_AoI = self.age_table[0]
            for sens in range(len(self.sens_table.index) - 1):
                self.avg_AoI += self.age_table[sens + 1]
                if self.age_table[sens + 1] < self.max_AoI:
                    self.max_AoI = self.age_table[sens + 1]
            self.avg_AoI = math.ceil(self.avg_AoI / len(self.age_table))

            state[self.headSerial + 1][2] = step - self.max_AoI
            state[self.headSerial + 1][3] = step - self.avg_AoI
        """END"""


        """Old Decision State: Just State"""
        # decision_state = copy.deepcopy(state)

        """New Decision State"""
        decision_state = copy.deepcopy(state)
        decision_state[0][2] = state[0][2] + round(p_count * 6_800_000 / (self.charge_rate * 60))
        for CH in range(len(state) - 1):
            decision_state[CH+1][2] = state[CH+1][2] + p_count
            decision_state[CH+1][3] = state[CH+1][3] + p_count
        """END"""

        action = -1
        target = self
        model_help = True
        change_transit = False
        dist = 0

        """Old Targeting"""
        # max_idx = -1
        # max_age = 720.0
        # max_sens = None
        # for sens in range(len(self.active_table)):
        #     if not self.active_table[sens]:
        #         if self.age_table[sens] < max_age:
        #             max_idx = sens
        #             max_age = self.age_table[sens]
        #             max_sens = self.sens_table.iat[sens, 0]
        #
        # decision_state[-1][1] = self.data_table[max_idx]
        # decision_state[-1][2] = self.age_table[max_idx]
        # decision_state[-1][1] = self.age_table[max_idx]
        #
        # action = model.act(decision_state)
        #
        # if action == 5:
        #     target = max_sens
        #     self.last_target = max_idx
        #     self.target_time = step
        #     model_help = True
        #     targetType = False
        # else:
        #     while force_out and action == self.headSerial:
        #         action = random.randint(0, len(full_sensor_list) - 2)
        #
        #     target = full_sensor_list.iat[action + 1, 0]
        #     self.last_target = target.headSerial
        #     self.target_time = step
        #     model_help = True
        #     targetType = True
        #
        # change_transit = True
        # dist = math.sqrt(pow((target.indX - self.indX), 2) + pow((target.indY - self.indY), 2))
        # self.tour = []
        #
        # AoI_peak = decision_state[1][2]
        # AoI_avg = decision_state[1][3]
        # for entry in range(len(full_sensor_list) - 2):
        #     AoI_avg += decision_state[entry + 2][3]
        #     AoI = decision_state[entry + 2][2]
        #     if AoI > AoI_peak:
        #         AoI_peak = AoI
        # AoI_avg = math.ceil(AoI_avg / len(full_sensor_list))
        #
        # return (model_help, change_transit, target, decision_state, action,
        #         dist, AoI_peak, AoI_avg, self.tour, targetType)

        """New Sensor Targeting: Christofides' Tour"""
        if not targetType:
            """
            For sensor targeting use Christofides
            if target type = False
            """
            inactive = []
            for sens in range(len(self.active_table)):
                if not self.active_table[sens]:
                    inactive.append(sens)     # Sensors on range 1 to num_sens

            if len(inactive) == 1:
                target = self.sens_table.iat[inactive[0], 0]
                dist = 2 * math.sqrt(pow((target.indX - self.indX), 2) + pow((target.indY - self.indY), 2))

                self.last_target = self.headSerial
                self.target_time = step
                action = self.headSerial
                model_help = False

            elif len(inactive) > 1:
                G = nx.Graph()
                for i in range(len(inactive)+1):
                    for j in range(i+1, len(inactive)+1):
                        if i == 0:
                            G.add_edge(
                                i,
                                j,
                                weight = math.sqrt(pow((self.sens_table.iat[inactive[j-1], 0].indX - self.indX), 2)
                                                   + pow((self.sens_table.iat[inactive[j-1], 0].indY - self.indY), 2))
                            )  # Use Euclidean distance (2-norm) as graph weight
                        elif j == 0:
                            G.add_edge(
                                i,
                                j,
                                weight=math.sqrt(pow((self.sens_table.iat[inactive[i-1], 0].indX - self.indX), 2)
                                                 + pow((self.sens_table.iat[inactive[i-1], 0].indY - self.indY), 2))
                            )  # Use Euclidean distance (2-norm) as graph weight
                        else:
                            G.add_edge(
                                i,
                                j,
                                weight = math.sqrt(pow((self.sens_table.iat[inactive[j-1], 0].indX
                                                        - self.sens_table.iat[inactive[i-1], 0].indX), 2)
                                                   + pow((self.sens_table.iat[inactive[j-1], 0].indY
                                                          - self.sens_table.iat[inactive[i-1], 0].indY), 2))
                            )

                tour = nx.algorithms.approximation.christofides(G)
                dists = []
                for i in range(len(tour)):
                    if i == 0:
                        dists.append(math.sqrt(pow((self.sens_table.iat[inactive[tour[i+1]-1], 0].indX - self.indX), 2)
                                              + pow((self.sens_table.iat[inactive[tour[i+1]-1], 0].indY - self.indY), 2)))
                    elif i+1 == len(tour):
                        dists.append(math.sqrt(pow((self.sens_table.iat[inactive[tour[i]-1], 0].indX - self.indX), 2)
                                              + pow((self.sens_table.iat[inactive[tour[i]-1], 0].indY - self.indY), 2)))
                    else:
                        dists.append(math.sqrt(pow((self.sens_table.iat[inactive[tour[i+1]-1], 0].indX -
                                                   self.sens_table.iat[inactive[tour[i]-1], 0].indX), 2)
                                              + pow((self.sens_table.iat[inactive[tour[i+1]-1], 0].indY -
                                                     self.sens_table.iat[inactive[tour[i]-1], 0].indY), 2)))

                # split into two function if too much distance\
                tour_discharge = round((sum(dists) / 15) * 1_000 * 6_800 / (1 * 60 * 60))
                if (state[0][2] - tour_discharge) >= (0.2 * 6_800 * 1_000):
                    self.tour = [self.sens_table.iat[inactive[tour[i]-1], 0] for i in range(len(tour))]
                    dist = sum(dists)
                    target = self.tour[0]
                elif len(tour) > 1:
                    tour1 = tour[0:math.ceil(len(tour)/2)]
                    tour2 = tour[math.ceil(len(tour)/2):]
                    dists1= (sum(dists[0:math.ceil(len(dists)/2)]) +
                             math.sqrt(pow((self.sens_table.iat[inactive[tour1[-1] - 1], 0].indX - self.indX), 2)
                                       + pow((self.sens_table.iat[inactive[tour1[-1] - 1], 0].indY - self.indY), 2))
                             )
                    dists2= (sum(dists[math.ceil(len(dists)/2):]) +
                             math.sqrt(pow((self.sens_table.iat[inactive[tour2[0] - 1], 0].indX - self.indX), 2)
                                       + pow((self.sens_table.iat[inactive[tour2[0] - 1], 0].indY - self.indY), 2))
                             )

                    tour1_discharge = round((dists1 / 15) * 1_000 * 6_800 / (1 * 60 * 60))
                    if (state[0][2] - tour1_discharge) >= (0.2 * 6_800 * 1_000):
                        self.tour = [self.sens_table.iat[inactive[tour1[i] - 1], 0] for i in range(len(tour1))]
                        dist = dists1
                        target = self.tour[0]
                    elif len(tour1) > 1:
                        distsA = dists[0:math.ceil(len(dists)/2)]
                        tour11 = tour[0:math.ceil(len(tour1) / 2)]
                        tour21 = tour[math.ceil(len(tour1) / 2):]
                        dists11 = (sum(distsA[0:math.ceil(len(distsA) / 2)]) +
                                  math.sqrt(pow((self.sens_table.iat[inactive[tour11[-1] - 1], 0].indX - self.indX), 2)
                                            + pow((self.sens_table.iat[inactive[tour11[-1] - 1], 0].indY - self.indY),
                                                  2))
                                  )
                        dists21 = (sum(distsA[math.ceil(len(distsA) / 2):]) +
                                  math.sqrt(pow((self.sens_table.iat[inactive[tour21[0] - 1], 0].indX - self.indX), 2)
                                            + pow((self.sens_table.iat[inactive[tour21[0] - 1], 0].indY - self.indY), 2))
                                  + math.sqrt(pow((self.sens_table.iat[inactive[tour21[-1] - 1], 0].indX - self.indX), 2)
                                            + pow((self.sens_table.iat[inactive[tour21[-1] - 1], 0].indY - self.indY), 2))
                                  )

                        self.tour = [self.sens_table.iat[inactive[tour11[i] - 1], 0] for i in
                                     range(len(tour11))]
                        dist = dists11
                        self.next1_tour = [self.sens_table.iat[inactive[tour21[i] - 1], 0] for i in
                                          range(len(tour21))]
                        self.next1_dist = dists21
                        target = self.tour[0]

                    tour2_discharge = round((dists2 / 15) * 1_000 * 6_800 / (1 * 60 * 60))
                    if (state[0][2] - tour2_discharge) >= (0.2 * 6_800 * 1_000):
                        self.next_tour = [self.sens_table.iat[inactive[tour2[i] - 1], 0] for i in
                                          range(len(tour2))]
                        self.next_dist = dists2
                    elif len(tour2) > 1:
                        distsB = dists[math.ceil(len(dists)/2):]
                        tour12 = tour[0:math.ceil(len(tour2) / 2)]
                        tour22 = tour[math.ceil(len(tour2) / 2):]
                        dists12 = (sum(distsB[0:math.ceil(len(distsB) / 2)]) +
                                   math.sqrt(pow((self.sens_table.iat[inactive[tour12[-1] - 1], 0].indX - self.indX), 2)
                                             + pow((self.sens_table.iat[inactive[tour12[-1] - 1], 0].indY - self.indY),2))
                                   + math.sqrt(pow((self.sens_table.iat[inactive[tour12[0] - 1], 0].indX - self.indX), 2)
                                             + pow((self.sens_table.iat[inactive[tour12[0] - 1], 0].indY - self.indY),2))
                                   )
                        dists22 = (sum(distsB[math.ceil(len(distsB) / 2):]) +
                                   math.sqrt(pow((self.sens_table.iat[inactive[tour22[0] - 1], 0].indX - self.indX), 2)
                                             + pow((self.sens_table.iat[inactive[tour22[0] - 1], 0].indY - self.indY),
                                                   2))
                                   )

                        self.next_tour = [self.sens_table.iat[inactive[tour12[i] - 1], 0] for i in
                                     range(len(tour12))]
                        self.next_dist = dists12
                        self.next2_tour = [self.sens_table.iat[inactive[tour22[i] - 1], 0] for i in
                                           range(len(tour22))]
                        self.next2_dist = dists22

                self.last_target = self.headSerial
                self.target_time = step
                action = self.headSerial
                model_help = False

            else:
                targetType = True
                model_help = True

        elif len(self.next_tour) > 0:
            self.tour = self.next_tour
            dist = self.next_dist
            target = self.tour[0]
            self.target_time = step
            action = self.headSerial
            model_help = False
            targetType = False

            self.next_tour = []
            self.next_dist = 0.0

        elif len(self.next1_tour) > 0:
            self.tour = self.next1_tour
            dist = self.next1_dist
            target = self.tour[0]
            self.target_time = step
            action = self.headSerial
            model_help = False
            targetType = False

            self.next1_tour = []
            self.next1_dist = 0.0

        elif len(self.next2_tour) > 0:
            self.tour = self.next2_tour
            dist = self.next2_dist
            target = self.tour[0]
            self.target_time = step
            action = self.headSerial
            model_help = False
            targetType = False

            self.next2_tour = []
            self.next2_dist = 0.0

        # Next CH
        if model_help:
            """
            For choosing next CH
            """

            action = model.act(decision_state)

            # Force the change in the sensor
            # while action == targetSerial:
            #     action = random.randint(0,len(full_sensor_list)-2)

            target = full_sensor_list.iat[action + 1, 0]
            change_transit = True

            dist = math.sqrt(pow((target.indX - self.indX), 2) + pow((target.indY - self.indY), 2))

        AoI_peak = decision_state[1][2]
        AoI_avg = decision_state[1][3]
        for entry in range(len(full_sensor_list) - 2):
            AoI_avg += decision_state[entry + 2][3]
            AoI = decision_state[entry + 2][2]
            if AoI > AoI_peak:
                AoI_peak = AoI
        AoI_avg = math.ceil(AoI_avg / len(full_sensor_list))

        """Dual Model Addition"""
        p_state = [dist, state[0][2] + round(p_count * 6_800_000 / (self.charge_rate * 60)),
                   AoI_peak + p_count, AoI_avg + p_count]
        # Value from 0 to 30
        self.action_p = model_p.act(p_state)
        """END"""

        """Single Model"""
        # return (model_help, change_transit, target, decision_state, action,
        #         dist, AoI_peak, AoI_avg, self.tour, targetType)
        """Dual Model"""
        return (model_help, change_transit, target, decision_state, action, self.action_p, p_state,
                dist, AoI_peak, AoI_avg, self.tour, targetType)
        """END"""