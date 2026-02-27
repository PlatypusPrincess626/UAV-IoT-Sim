"""
File: devices_iot.py
Author: Mason Conkel
Creation Date: 2025-05-29
Description: This script simulates the edge devices to be used in the simulation of a remote IoT network.
"""
# Import Dependencies
from scipy.interpolate import InterpolatedUnivariateSpline
import pandas as pd
import numpy as np
import random
import math
import copy
import networkx as nx

class EdgeDevice:
    """
    Parent class for the construction of an edge device
    """
    def __init__(self, x: int, y: int, longitude: float, latitude: float, max_data: int):
        # Assign positional information
        self.id_x = x  # X index
        self.id_y = y  # Y index
        self.id_z = 0  # Z index
        self.long = longitude  # Longitude
        self.lat = latitude  # Latitude

        # PLoRa specs
        self._comms = {
            "max_dist_lora": 5_000,  # LoRa maximum distance (m)
            "max_bitrate_lora": 24_975,  # LoRa maximum bitrate (bps)
            "current_active_lora": 4_200,  # LoRa active current (muA)
            "current_sleep_lora": 1_200,  # LoRa dormant current (muA)
            "voltage_lora": 3.7,  # LoRa voltage requirements (V)
            "pow_active_lora": 20,  # LoRa active power (muW)
            "pow_sleep_lora": 4,  # LoRa dormant power (muW)

            "max_dist_ambc": 800,  # AmBC maximum distance (m)
            "max_bitrate_ambc": 1_592,  # AmBC maximum bitrate (bps)
            "pow_ambc": 3,  # AmBC upkeep power (muW)
            "voltage_ambc": 3.3,  # AmBC voltage requirements (V)
            "current_ambc": 785  # AmBC upkeep current (muA)
        }

        # CPU Specs
        self.pow_cpu = 3.7  # CPU power (muW)
        self.current_cpu = 1_000  # CPU current drain (muA)

        # Solar Specs
        self.spectral_low = 0  # Lower bandwidth bound
        self.spectral_high = np.inf  # Upper bandwidth bound
        self.is_solar = True  # Flag for solar powered

        # Sensor data storage
        self.max_data = max_data  # Sensor maximum data storage
        self.reset_min = round(self.max_data * 0.10)  # Minimum data amount on reset
        self.reset_max = round(self.max_data * 0.25)  # Maximum data amount on reset

        self.spectral_response = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.23], [0.25], [0.27], [0.29], [0.31], [0.33], [0.3376], [0.3452], [0.3529],
                                           [0.3605], [0.3681], [0.3757], [0.3833], [0.3910], [0.3986], [0.4062],
                                           [0.4138], [0.4214], [0.4290], [0.4367], [0.4443], [0.4595], [0.4694],
                                           [0.4824], [0.4976], [0.5174], [0.5263], [0.5433], [0.5586], [0.5647],
                                           [0.5695], [0.5814], [0.5910], [0.5948], [0.5986], [0.6024], [0.6119],
                                           [0.6271], [0.6393], [0.6427], [0.6274], [0.6107], [0.5714], [0.5321],
                                           [0.4830], [0.4634], [0.4437], [0.4339], [0.4202], [0.3986], [0.3652],
                                           [0.3357], [0.3092], [0.2179], [0.1589], [10.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])

    # Data fetching functions
    def get_indicies(self):
        return self.id_x, self.id_y

    def get_coords(self):
        return self.lat, self.long

    def battery_calc(self, stored_percentage, max_storage, amp_spent, step_charge):
        stored_percentage -= amp_spent / (max_storage * 60) # mA / mA or A / A
        stored_percentage = min(100.0, (stored_percentage + step_charge))

        return stored_percentage

    def optimal_move(self, env, r_max, x_ch, y_ch, alpha, move_type, step, speed):
        x_c, y_c = self.get_indicies()
        pt_optimal = [x_c, y_c]

        # Determine starting point (higher alpha)
        if move_type == 0:
            obfuscation_potential = 100.0
            for x in range(2 * math.floor(r_max) + 1):
                for y in range(2 * math.floor(r_max) + 1):
                    x_try = x_c + x - math.floor(r_max)
                    y_try = y_c + y - math.floor(r_max)
                    if math.sqrt((x_c - x_try) ** 2 + (y_c - y_try) ** 2) < r_max:
                        curr_obfuscation = env.get_obfuscation(x_try, y_try, step)
                        avg_nxt_obfuscation = 0
                        for w in range(3):
                            for v in range(3):
                                avg_nxt_obfuscation += env.get_obfuscation(x_try + math.floor((w - 1) * speed),
                                                                           y_try + math.floor((v - 1) * speed),
                                                                           step + 1)
                        avg_nxt_obfuscation = 1/9 * avg_nxt_obfuscation
                        avg_stl_obfuscation = 0
                        for timestep in range(env.max_num_steps - step):
                            avg_stl_obfuscation += env.get_obfuscation(x_try, y_try, timestep)
                        avg_stl_obfuscation = avg_stl_obfuscation / (env.max_num_steps - step)
                        if (curr_obfuscation + alpha * avg_nxt_obfuscation + (1 - alpha) * avg_stl_obfuscation
                                < obfuscation_potential):
                            obfuscation_potential = (curr_obfuscation + alpha * avg_nxt_obfuscation +
                                                     (1 - alpha) * avg_stl_obfuscation)
                            pt_optimal = [x_try, y_try]
        else:
            obfuscation_potential = 100.0
            for x in range(3):
                for y in range(3):
                    next_x = x_ch + math.floor((x - 1) * speed)
                    next_y = y_ch + math.floor((y - 1) * speed)
                    if math.sqrt((x_c - next_x) ** 2 + (y_c - next_y) ** 2) < r_max:
                        curr_obfuscation = env.get_obfuscation(x_ch, y_ch, step)
                        avg_nxt_obfuscation = 0
                        for w in range(3):
                            for v in range(3):
                                avg_nxt_obfuscation += env.get_obfuscation(next_x + math.floor((w - 1) * speed),
                                                                           next_y + math.floor((v - 1) * speed),
                                                                           step + 1)
                        avg_nxt_obfuscation = 1 / 9 * avg_nxt_obfuscation
                        avg_stl_obfuscation = 0
                        for timestep in range(env.max_num_steps - step):
                            avg_stl_obfuscation += env.get_obfuscation(next_x, next_y, timestep)
                        avg_stl_obfuscation = avg_stl_obfuscation / (env.max_num_steps - step)
                        if (curr_obfuscation + alpha * avg_nxt_obfuscation + (1 - alpha) * avg_stl_obfuscation
                                < obfuscation_potential):
                            obfuscation_potential = (curr_obfuscation + alpha * avg_nxt_obfuscation +
                                                     (1 - alpha) * avg_stl_obfuscation)
                            pt_optimal = [next_x, next_y]

        return pt_optimal


    def find_power(self, env, x: int, y: int, step: int,
                   sol_area: int, tilt: int, azimuth: int):

        spectra = env.get_spectrum(self.lat, self.long, tilt, azimuth, step)
        interference = env.get_obfuscation(x, y, step)
        cell_current = np.trapz(spectra['poa_global'] * self.spectral_response,
                                spectra['wavelength'], axis=0)
        a = step / 60 + 2
        alpha = abs(104 - 65 * a + 47 * pow(a, 2) - 12 * pow(a, 3) + pow(a, 4))
        power = abs(alpha / 100) * interference * cell_current * sol_area

        return power

    def path(self, env, num_legs: int, r_max: int, max_speed: int, acceleration: int, u_rr: float, w: int,
             sol_area: int, sol_current: int, sol_voltage: int, tilt: int, azimuth: int):
        legs = []
        times = []
        leg = 0
        centroid = [self.get_indicies()]
        pt_max = [self.get_indicies()]
        t_run = 0
        pts = []

        inactive_flag = False
        while not inactive_flag:
            power = self.find_power(env, self.id_x, self.id_y, t_run, sol_area, tilt, azimuth)

            if power / sol_current > sol_voltage * 0.8:
                t_run += 1
            else:
                inactive_flag = True
        t_max = t_run

        for x in range(2 * math.floor(r_max) + 1):
            for y in range(2 * math.floor(r_max) + 1):
                x_try = centroid[0] + x - math.floor(r_max)
                y_try = centroid[1] + y - math.floor(r_max)
                t_run = 0
                if math.sqrt((centroid[0] - x_try) ** 2 + (centroid[1] - y_try) ** 2) < r_max:
                    pts.append([x_try, y_try])
                    inactive_flag = False
                    while not inactive_flag:
                        power = self.find_power(env, x_try, y_try, t_run, sol_area, tilt, azimuth)
                        if power / sol_current > sol_voltage * 0.8:
                            t_run += 1
                        else:
                            inactive_flag = True
                    if t_run > t_max:
                        pt_max = [x_try, y_try]
                        t_max = t_run
        legs.append(pt_max)
        times.append(t_max)
        t_temp = t_max
        leg += 1

        while leg < (num_legs - 1):
            max_energy = 0
            for pt in pts:
                t_run = t_temp
                energy_harvest = 0
                travel_dist = math.sqrt((pt_max[0] - pt[0]) ** 2 + (pt_max[1] - pt[1]) ** 2)
                accel_dist = ((0.5 * acceleration * (max_speed/acceleration)**2) +
                              (max_speed * (max_speed/acceleration) - 0.5 * acceleration * (max_speed/acceleration)**2))

                if accel_dist < travel_dist:
                    t_accel = math.sqrt(travel_dist/acceleration)
                    v_final = acceleration * t_accel
                    energy_move = w/9.8 * v_final**2
                else:
                    t_remain = (travel_dist - accel_dist) / max_speed
                    power_velocity = w * 3600 * max_speed * u_rr
                    energy_move = w/9.8 * max_speed**2 + power_velocity * t_remain

                inactive_flag = False
                while not inactive_flag:
                    power = self.find_power(env, pt[0], pt[1], t_run, sol_area, tilt, azimuth)
                    if power / sol_current > sol_voltage * 0.8:
                        t_run += 1
                        energy_harvest += power
                    else:
                        inactive_flag = True

                if energy_harvest - energy_move > max_energy:
                    pt_max = [pt[0], pt[1]]
                    t_max = t_run
                    max_energy = energy_harvest - energy_move
            legs.append(pt_max)
            times.append(t_max)
            t_temp = t_max
            leg += 1

        max_energy = 0
        for pt in pts:
            energy_harvest = 0
            t_run = t_temp
            travel_dist = math.sqrt((pt_max[0] - pt[0]) ** 2 + (pt_max[1] - pt[1]) ** 2)
            accel_dist = ((0.5 * acceleration * (max_speed / acceleration) ** 2) +
                          (max_speed * (max_speed / acceleration) - 0.5 * acceleration *
                           (max_speed / acceleration) ** 2))

            if accel_dist < travel_dist:
                t_accel = math.sqrt(travel_dist / acceleration)
                v_final = acceleration * t_accel
                energy_move = w / 9.8 * v_final ** 2
            else:
                t_remain = (travel_dist - accel_dist) / max_speed
                power_velocity = w * 3600 * max_speed * u_rr
                energy_move = w / 9.8 * max_speed ** 2 + power_velocity * t_remain

            while t_run < env.max_num_steps:
                power = self.find_power(env, pt[0], pt[1], t_run, sol_area, tilt, azimuth)
                if power / sol_current > sol_voltage * 0.8:
                    t_run += 1
                    energy_harvest += power

            if energy_harvest - energy_move > max_energy:
                pt_max = [pt[0], pt[1]]
                t_max = t_run
                max_energy = energy_harvest - energy_move
        legs.append(pt_max)
        times.append(t_max)
        leg += 1

        return legs, times


class Sensor(EdgeDevice):
    """
    Child class of EdgeDevice that simulates a sensor
    """
    def __init__(self, x: int, y: int, long: float, lat: float, data_type: int):
        super().__init__(x, y, long, lat, 256_000)
        self.type = 1
        self.data_type = data_type
        self.stored_data = random.randint(self.reset_min, self.reset_max)  # Initialization of data storage
        self.curr_pt = [self.id_x, self.id_y]

        # Sensor power requirements
        self.sens_pow = 2.2  # 2.2 W power consumption
        self.sens_amp = self.sens_pow/self._comms["voltage_ambc"]

        # Solar panel specifics
        self.azimuth = 0  # 0 for North pointing
        self.tilt = 0  # 0 for no horizontal tilt
        self.solar_voltage = 5  # V
        self.solar_current = 2  # A
        self.solar_power = 10  # W
        self.solar_area = .137 * .222   # 137 mm x 222 mm

        # Sensor data collection
        self.max_col_rate = 16  # Sensor max collection rate
        self.sample_freq = 25  # Percent of a minute spent sampling
        self.sample_len = 1  # Number of seconds to sample over
        self.sample_rate = 300  # Samples taken per period
        self.bits_per_min = round(self.max_col_rate * self.sample_len *
                                  self.sample_rate / (self.sample_freq / 100))

        # Battery Specifics
        self.C = 1  # Capacitance of the energy storage (F)
        self.max_energy = 1_280  # Maximum battery capacity (mAh)
        self.charge_rate = 1/3  # 20 mins to charge
        self.discharge_rate = 2  # 2 h to discharge
        self.stored_energy = 100.0  # Percent

        self.total_amp_spent = 0.0
        self.step_charge = 100 / (self.charge_rate * 60)

    def reset(self):
        self.stored_energy = 100.0 # Percent
        self.stored_data = random.randint(0, self.reset_max)
        self.total_amp_spent = 0.0

    def step(self, alpha, env, step):
        self.total_amp_spent = 0.0
        self.harvest_energy(alpha, env, step)
        self.harvest_data()
        self.stored_energy = self.battery_calc(self.stored_energy, self.max_energy, self.total_amp_spent, self.step_charge)

    def step_battery(self):
        self.stored_energy = self.battery_calc(self.stored_energy, self.max_energy, self.total_amp_spent, self.step_charge)

    def harvest_data(self):
        if self.is_solar:
            self.stored_data += min(self.bits_per_min, self.max_data)
            return True

        elif self.stored_energy > round(10):
            self.stored_data += min(self.bits_per_min, self.max_data)
            self.total_amp_spent += self.sens_amp * (self.sample_freq / 100)
            return True

        else:
            return False

    def harvest_energy(self, alpha, env, step):
        spectra = env.get_spectrum(self.lat, self.long, self.tilt, self.azimuth, step)
        interference = env.get_obfuscation(self.id_x, self.id_y, step)
        cell_current = np.trapz(spectra['poa_global'] * self.spectral_response, spectra['wavelength'], axis=0)
        power = abs(alpha / 100) * interference * cell_current * self.solar_area

        if power / self.solar_current > self.solar_voltage * 0.8:
            self.is_solar = True
        else:
            self.is_solar = False

        amp_upkeep = (self.current_cpu + self._comms.get("current_ambc")) / 1_000  # to mA
        self.total_amp_spent += amp_upkeep

    def upload(self, X, Y):
        if (math.sqrt(pow((self.id_x - X), 2) + pow((self.id_y - Y), 2))
                <= self._comms.get("max_dist_ambc")):
            temp = min(self._comms.get("max_bitrate_ambc") * 26, self.stored_data)
            self.stored_data -= temp
            return temp
        else:
            return -1


class ClusterHead(EdgeDevice):
    """
    Child class of EdgeDevice that simulates a cluster head
    """
    def __init__(self, env, x: int, y: int, long: float, lat: float, sensor_list: list, r_move: float, cluster_num: int):
        super().__init__(x, y, long, lat, 12_500_000)
        self.type = 2
        self.typeStr = "Clusterhead"
        self.cluster_num = cluster_num
        self.stored_data = self.reset_max  # Storage initialization

        # Solar panel specifics
        self.azimuth = 180
        self.tilt = 45
        self.h = 0
        self.solar_area = 1.020 * 0.520  # 1020 mm  x 520 mm
        self.solar_voltage = 18  # V
        self.solar_current = 6  # A
        self.solar_power = 100  # W

        self.r_max = r_move
        self.speed = 20  # m / min
        self.acceleration = 300  # m / min**2
        self.u_rr = 0.1
        self.num_legs = 4
        self.w = 21  # N
        self.legs, self.times = self.path(env, self.num_legs, self.r_max, self.speed, self.acceleration,
                                          self.u_rr, self.w, self.solar_area, self.solar_current,
                                          self.solar_voltage, self.tilt, self.azimuth)

        self.leg_num = 0
        self.leg = self.legs[self.leg_num]
        self.est_point = self.leg
        self.curr_pt = self.leg
        self.curr_spd = 0

        # State tracking of sensors
        self.max_age = 0
        self.avg_age = 0
        self.num_sensors = 0
        self.contribution = 0
        self.action_p = 0
        self.sens_dict = None
        self.sens_table = None
        self.active_table = None
        self.age_table = None
        self.data_table = None
        self.set_sensor_data(sensor_list)

        # Target tracking for UAV
        """
        Note: Too many lists for full use. Find method to store various lists
        at different sizes to reduce numbers of variables.
        """
        self.last_target = 0
        self.target_time = 0

        self.tour = []
        self.next_tour = []
        self.next_dist = 0.0
        self.next1_tour = []
        self.next1_dist = 0.0
        self.next2_tour = []
        self.next2_dist = 0.0

        # Battery Specifics
        self.C = 3200  # Capacitance of the energy storage (F)
        self.max_energy = 12_000  # Maximum battery capacity (mAh)
        self.charge_rate = 1 / 3  # Charge rate of battery (A/h)
        self.discharge_rate = 0.655  # Discharge rate of batter (A/h)
        self.battery_voltage = 15

        self.uav_charge_amp = 10  # A

        self.stored_energy = 100  # Battery initialization
        self.total_amp_spent = 0.0
        self.step_charge = 100 / (self.charge_rate * 60)

    def reset(self):
        # Reset State
        self.max_age = 0
        self.avg_age = 0
        self.data_table[0] = self.stored_data
        for sens in range(self.num_sensors):
            self.active_table[sens] = True
            self.age_table[sens] = 0
            self.data_table[sens + 1] = 0

        # Reset data and battery
        self.stored_data = random.randint(0, self.reset_max)
        self.stored_energy = 100
        self.total_amp_spent = 0.0
        self.step_charge = 100 / (self.charge_rate * 60)

        self.leg_num = 0
        self.leg = self.legs[self.leg_num]
        self.est_point = self.leg
        self.curr_pt = self.leg
        self.curr_spd = 0

        # Reset target
        self.contribution = 0
        self.action_p = 0
        self.last_target = 0
        self.target_time = 0

        self.tour = []
        self.next_tour = []
        self.next_dist = 0.0
        self.next1_tour = []
        self.next1_dist = 0.0
        self.next2_tour = []
        self.next2_dist = 0.0

    def step(self, alpha, env, step):
        self.total_amp_spent = 0.0
        self.harvest_energy(alpha, env, step)
        self.download(step)

    def step_battery(self):
        self.stored_energy = self.battery_calc(self.stored_energy, self.max_energy, self.total_amp_spent, self.step_charge)

    def harvest_energy(self, alpha, env, step):
        power = self.find_power(env, self.curr_pt[0], self.curr_pt[1], step, self.solar_area, self.tilt, self.azimuth)

        # Check if the machine is powered
        if power / self.solar_current > self.solar_voltage * 0.8:
            self.is_solar = True
        else:
            self.is_solar = False

        if step == self.times[self.leg_num]:
            self.leg_num += 1
            self.leg = self.legs[self.leg_num]

        energy_move = 0
        if self.curr_pt != self.leg:
            travel_dist = math.sqrt((self.leg[0] - self.est_point[0]) ** 2 + (self.leg[1] - self.est_point[1]) ** 2)
            accel_dist = ((0.5 * self.acceleration * (self.speed / self.acceleration) ** 2) +
                          (self.speed * (self.speed / self.acceleration) - 0.5 * self.acceleration *
                           (self.speed / self.acceleration) ** 2))
            if accel_dist < travel_dist:
                t_accel = math.sqrt(travel_dist / self.acceleration)
                v_final = self.acceleration * t_accel
                energy_move = self.w / 9.8 * v_final ** 2
                self.curr_pt = self.leg
                self.est_point = self.leg
            elif (accel_dist + (1 - 2 * self.speed / self.acceleration) * self.speed) >= travel_dist:
                t_remain = (travel_dist - accel_dist) / self.speed
                power_velocity = self.w * 3600 * self.speed * self.u_rr
                energy_move = self.w / 9.8 * self.speed ** 2 + power_velocity * t_remain
                self.curr_pt = self.leg
                self.est_point = self.leg
            elif self.curr_spd < self.speed:
                t_remain = (travel_dist - 0.5 * accel_dist) / self.speed
                power_velocity = self.w * 3600 * self.speed * self.u_rr
                energy_move = self.w / 9.8 * self.speed ** 2 + power_velocity * t_remain
                travelled = self.speed * t_remain + 0.5 * self.acceleration * (self.speed / self.acceleration)**2
                self.est_point = [self.est_point[0] + (travelled / travel_dist) * (self.leg[0] - self.est_point[0]),
                                  self.est_point[1] + (travelled / travel_dist) * (self.leg[1] - self.est_point[1])]
                self.curr_pt = [round(self.est_point[0]), round(self.est_point[1])]
            elif (0.5 * accel_dist + (1 - self.speed / self.acceleration) * self.speed) > travel_dist:
                t_remain = (travel_dist - 0.5 * accel_dist) / self.speed
                power_velocity = self.w * 3600 * self.speed * self.u_rr
                energy_move = self.w / 9.8 * self.speed ** 2 + power_velocity * t_remain
                self.curr_pt = self.leg
                self.est_point = self.leg
            else:
                power_velocity = self.w * 3600 * self.speed * self.u_rr
                energy_move = power_velocity
                travelled = self.speed
                self.est_point = [self.est_point[0] + (travelled / travel_dist) * (self.leg[0] - self.est_point[0]),
                                  self.est_point[1] + (travelled / travel_dist) * (self.leg[1] - self.est_point[1])]
                self.curr_pt = [round(self.est_point[0]), round(self.est_point[1])]

            self.total_amp_spent += energy_move / self.battery_voltage


        amp_upkeep = (self.current_cpu + self._comms.get("current_active_lora")) / 1_000  # to mA
        self.total_amp_spent += amp_upkeep

    def set_sensor_data(self, sens_list: list):
        self.num_sensors = len(sens_list)

        self.active_table = np.array([0] * self.num_sensors)
        self.age_table = np.array([0] * self.num_sensors)
        self.sens_table = pd.DataFrame(np.array(sens_list))

        data_list = [self.stored_data]
        for sensor in range(self.num_sensors):
            data_list.append(0)
        self.data_table = np.array(data_list)

    def download(self, step):
        rotations = math.ceil(len(self.sens_table.index) / 2)
        rotation = step % rotations
        sensor = rotation * 2
        activeChannels = []
        sensor1 = self.sens_table.iat[sensor, 0]
        activeChannels.append(sensor1.upload(self.id_x, self.id_y))

        if rotation < (rotations - 1) or len(self.sens_table.index) % 2 == 0:
            sensor2 = self.sens_table.iat[sensor + 1, 0]
            activeChannels.append(sensor2.upload(self.id_x, self.id_y))

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

        # ADF 2.0
        self.max_age = self.age_table[0]
        self.avg_age = self.age_table[0]
        for sens in range(len(self.sens_table.index) - 1):
            self.avg_age += self.age_table[sens + 1]
            if self.age_table[sens + 1] < self.max_age:
                self.max_age = self.age_table[sens + 1]
        self.avg_age = math.ceil(self.avg_age / len(self.age_table))

    def upload(self, X: int, Y: int):
        if self.is_solar:
            if math.sqrt(pow((self.id_x - X), 2) + pow((self.id_y - Y), 2)) <= self._comms.get("max_dist_lora"):

                if self.stored_data > 0:
                    sent_data = min(self._comms.get("max_bitrate_lora") * 56, self.stored_data)
                    self.stored_data -= sent_data
                    self.contribution += sent_data

                    return sent_data, self.max_age, self.avg_age

                else:
                    return 0, self.max_age, self.avg_age

            else:
                return -1, self.max_age, self.avg_age

        # Extra drain for another comms channel
        elif self.stored_energy > ( ( self._comms.get("current_active_lora") / 1_000 ) / ( self.max_energy * 60 ) ):
            if math.sqrt(pow((self.id_x - X), 2) + pow((self.id_y - Y), 2)) <= self._comms.get("max_dist_lora"):

                if self.stored_data > 0:
                    sent_data = min(self._comms.get("max_bitrate_lora") * 56, self.stored_data)
                    self.stored_data -= sent_data
                    self.contribution += sent_data
                    time = sent_data / self._comms.get("max_bitrate_lora") * 56

                    self.total_amp_spent += (self._comms.get("current_active_lora") / 1_000) * (time / 60)
                    return sent_data, self.max_age, self.avg_age

                else:
                    self.total_amp_spent += self._comms.get("current_active_lora") / 1_000
                    self.total_amp_spent += (self._comms.get("current_active_lora") / 1_000) * (4 / 60)
                    return 0, self.max_age, self.avg_age

            else:
                return -1, self.max_age, self.avg_age
        else:
            return -1, self.max_age, self.avg_age

    def charge_time(self, X: int, Y: int, charge):
        if abs(self.id_x - X) < 1.0 and abs(self.id_y - Y) < 1.0:
            if self.is_solar and charge:
                return 60.0
            elif self.stored_energy > 75 and charge:
                self.total_amp_spent += self.uav_charge_amp * 1_000
                return 60.0
            elif self.stored_energy > 25:
                if not self.is_solar:
                    self.total_amp_spent += self.uav_charge_amp * 1_000 * (30 / 60)
                return 30.0
            else:
                return 0
        else:
            return 0

    def get_dest(self, state, full_sensor_list, model, model_p,
                     step, p_count, targetType, _=None):
    
        my_contribution = state[self.cluster_num + 1][1]
        if my_contribution > self.contribution:
            self.contribution = my_contribution

            for sens in range(len(self.active_table)):
                if not self.active_table[sens]:
                    self.age_table[sens] = self.target_time     # Sensors on range 1 to num_sens

            self.max_age = self.age_table[0]
            self.avg_age = self.age_table[0]
            for sens in range(len(self.sens_table.index) - 1):
                self.avg_age += self.age_table[sens + 1]
                if self.age_table[sens + 1] < self.max_age:
                    self.max_age = self.age_table[sens + 1]
            self.avg_age = math.ceil(self.avg_age / len(self.age_table))

            state[self.cluster_num + 1][2] = step - self.max_age
            state[self.cluster_num + 1][3] = step - self.avg_age

        decision_state = copy.deepcopy(state)
        decision_state[0][2] = state[0][2] + round(p_count * 6_800_000 / (self.charge_rate * 60))
        for CH in range(len(state) - 1):
            decision_state[CH+1][2] = state[CH+1][2] + p_count
            decision_state[CH+1][3] = state[CH+1][3] + p_count

        action = -1
        target = self
        model_help = True
        change_transit = False
        dist = 0

        """Christofides' Tour"""
        if not targetType:
            inactive = []
            for sens in range(len(self.active_table)):
                if not self.active_table[sens]:
                    inactive.append(sens)     # Sensors on range 1 to num_sens

            if len(inactive) == 1:
                target = self.sens_table.iat[inactive[0], 0]
                dist = 2 * math.sqrt(pow((target.id_x - self.id_x), 2) + pow((target.id_y - self.id_y), 2))

                self.last_target = self.cluster_num
                self.target_time = step
                action = self.cluster_num
                model_help = False

            elif len(inactive) > 1:
                G = nx.Graph()
                for i in range(len(inactive)+1):
                    for j in range(i+1, len(inactive)+1):
                        if i == 0:
                            G.add_edge(
                                i,
                                j,
                                weight = math.sqrt(pow((self.sens_table.iat[inactive[j-1], 0].id_x - self.id_x), 2)
                                                   + pow((self.sens_table.iat[inactive[j-1], 0].id_y - self.id_y), 2))
                            )  # Use Euclidean distance (2-norm) as graph weight
                        elif j == 0:
                            G.add_edge(
                                i,
                                j,
                                weight=math.sqrt(pow((self.sens_table.iat[inactive[i-1], 0].id_x - self.id_x), 2)
                                                 + pow((self.sens_table.iat[inactive[i-1], 0].id_y - self.id_y), 2))
                            )  # Use Euclidean distance (2-norm) as graph weight
                        else:
                            G.add_edge(
                                i,
                                j,
                                weight = math.sqrt(pow((self.sens_table.iat[inactive[j-1], 0].id_x
                                                        - self.sens_table.iat[inactive[i-1], 0].id_x), 2)
                                                   + pow((self.sens_table.iat[inactive[j-1], 0].id_y
                                                          - self.sens_table.iat[inactive[i-1], 0].id_y), 2))
                            )

                tour = nx.algorithms.approximation.christofides(G)
                dists = []
                for i in range(len(tour)):
                    if i == 0:
                        dists.append(math.sqrt(pow((self.sens_table.iat[inactive[tour[i+1]-1], 0].id_x - self.id_x), 2)
                                              + pow((self.sens_table.iat[inactive[tour[i+1]-1], 0].id_y - self.id_y), 2)))
                    elif i+1 == len(tour):
                        dists.append(math.sqrt(pow((self.sens_table.iat[inactive[tour[i]-1], 0].id_x - self.id_x), 2)
                                              + pow((self.sens_table.iat[inactive[tour[i]-1], 0].id_y - self.id_y), 2)))
                    else:
                        dists.append(math.sqrt(pow((self.sens_table.iat[inactive[tour[i+1]-1], 0].id_x -
                                                   self.sens_table.iat[inactive[tour[i]-1], 0].id_x), 2)
                                              + pow((self.sens_table.iat[inactive[tour[i+1]-1], 0].id_y -
                                                     self.sens_table.iat[inactive[tour[i]-1], 0].id_y), 2)))

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
                             math.sqrt(pow((self.sens_table.iat[inactive[tour1[-1] - 1], 0].id_x - self.id_x), 2)
                                       + pow((self.sens_table.iat[inactive[tour1[-1] - 1], 0].id_y - self.id_y), 2))
                             )
                    dists2= (sum(dists[math.ceil(len(dists)/2):]) +
                             math.sqrt(pow((self.sens_table.iat[inactive[tour2[0] - 1], 0].id_x - self.id_x), 2)
                                       + pow((self.sens_table.iat[inactive[tour2[0] - 1], 0].id_y - self.id_y), 2))
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
                                  math.sqrt(pow((self.sens_table.iat[inactive[tour11[-1] - 1], 0].id_x - self.id_x), 2)
                                            + pow((self.sens_table.iat[inactive[tour11[-1] - 1], 0].id_y - self.id_y),
                                                  2))
                                  )
                        dists21 = (sum(distsA[math.ceil(len(distsA) / 2):]) +
                                  math.sqrt(pow((self.sens_table.iat[inactive[tour21[0] - 1], 0].id_x - self.id_x), 2)
                                            + pow((self.sens_table.iat[inactive[tour21[0] - 1], 0].id_y - self.id_y), 2))
                                  + math.sqrt(pow((self.sens_table.iat[inactive[tour21[-1] - 1], 0].id_x - self.id_x), 2)
                                            + pow((self.sens_table.iat[inactive[tour21[-1] - 1], 0].id_y - self.id_y), 2))
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
                                   math.sqrt(pow((self.sens_table.iat[inactive[tour12[-1] - 1], 0].id_x - self.id_x), 2)
                                             + pow((self.sens_table.iat[inactive[tour12[-1] - 1], 0].id_y - self.id_y),2))
                                   + math.sqrt(pow((self.sens_table.iat[inactive[tour12[0] - 1], 0].id_x - self.id_x), 2)
                                             + pow((self.sens_table.iat[inactive[tour12[0] - 1], 0].id_y - self.id_y),2))
                                   )
                        dists22 = (sum(distsB[math.ceil(len(distsB) / 2):]) +
                                   math.sqrt(pow((self.sens_table.iat[inactive[tour22[0] - 1], 0].id_x - self.id_x), 2)
                                             + pow((self.sens_table.iat[inactive[tour22[0] - 1], 0].id_y - self.id_y),
                                                   2))
                                   )

                        self.next_tour = [self.sens_table.iat[inactive[tour12[i] - 1], 0] for i in
                                     range(len(tour12))]
                        self.next_dist = dists12
                        self.next2_tour = [self.sens_table.iat[inactive[tour22[i] - 1], 0] for i in
                                           range(len(tour22))]
                        self.next2_dist = dists22

                self.last_target = self.cluster_num
                self.target_time = step
                action = self.cluster_num
                model_help = False

            else:
                targetType = True
                model_help = True

        elif len(self.next_tour) > 0:
            self.tour = self.next_tour
            dist = self.next_dist
            target = self.tour[0]
            self.target_time = step
            action = self.cluster_num
            model_help = False
            targetType = False

            self.next_tour = []
            self.next_dist = 0.0

        elif len(self.next1_tour) > 0:
            self.tour = self.next1_tour
            dist = self.next1_dist
            target = self.tour[0]
            self.target_time = step
            action = self.cluster_num
            model_help = False
            targetType = False

            self.next1_tour = []
            self.next1_dist = 0.0

        elif len(self.next2_tour) > 0:
            self.tour = self.next2_tour
            dist = self.next2_dist
            target = self.tour[0]
            self.target_time = step
            action = self.cluster_num
            model_help = False
            targetType = False

            self.next2_tour = []
            self.next2_dist = 0.0

        # Next CH
        """Cluster Targeting"""
        if model_help:
            action = model.act(decision_state)

            target = full_sensor_list.iat[action + 1, 0]
            change_transit = True

            dist = math.sqrt(pow((target.id_x - self.id_x), 2) + pow((target.id_y - self.id_y), 2))

        AoI_peak = decision_state[1][2]
        AoI_avg = decision_state[1][3]
        for entry in range(len(full_sensor_list) - 2):
            AoI_avg += decision_state[entry + 2][3]
            AoI = decision_state[entry + 2][2]
            if AoI > AoI_peak:
                AoI_peak = AoI
        AoI_avg = math.ceil(AoI_avg / len(full_sensor_list))

        p_state = [dist, state[0][2] + round(p_count * 6_800_000 / (self.charge_rate * 60)),
                   AoI_peak + p_count, AoI_avg + p_count]
        # Value from 0 to 30
        self.action_p = model_p.act(p_state)

        return (model_help, change_transit, target, decision_state, action, self.action_p, p_state,
                dist, AoI_peak, AoI_avg, self.tour, targetType)