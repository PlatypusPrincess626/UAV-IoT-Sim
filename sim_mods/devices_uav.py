"""
File: devices_iot.py
Author: Mason Conkel
Creation Date: 2025-10-06
Description: This script simulates the uav devices to be used in the simulation of a remote IoT network.
"""
import math
import pandas as pd
import numpy as np


class QuadUAV:
    def __init__(self, X: int, Y: int, long: float, lat: float, uavNum: int, CHList: list):
        self.action = None
        self.type = 3
        self.serial = uavNum
        self.typeStr = "UAV"

        # Communication Specifications
        self._comms = {
            "max_dist_lora": 5_000,
            "max_bitrate_lora": 24_975,
            "": 4_200,  # micro-amps transmitting
            "voltage_lora": 3.7,
            "pow_active_lora": 0.020,  # transmitting
            "pow_sleep_lora": 0.004,  # static
            "current_sleep_lora": 1_200,  # micro-amps

            "max_dist_ambc": 800,
            "max_bitrate_ambc": 1_592,
            "pow_ambc": 0.00259,  # upkeep
            "voltage_ambc": 3.3,
            "current_ambc": 785  # micro-amps upkeep
        }

        # Positioning
        self.id_x = int(X)
        self.id_y = int(Y)
        self.maxH = 20
        self.h = 1

        # Trajectory
        self.target = None
        self.targetHead = None
        self.last_Head = None
        self.targetSerial = 0
        self.tour = []
        self.tour_iter = 0

        self.target_x = int(X)
        self.target_y = int(Y)
        self.lat = lat
        self.long = long
        self.last_AoI = 0
        self.bad_target = 0

        # Movement
        self.maxSpd = 15  # 15 m/s max speed cap
        self.maxTurn = 200  # 200 degree per sec max yaw
        self.maxClimb = 3  # 3 m/s ascent and descent
        self.inRange = True

        self.p_cycle = 0
        self.p_count = 0

        # State
        self.crash = False
        self.model_transit = False
        self.no_hold = True
        self.force_change = False
        self.force_count = 0
        self.targetType = True

        self.origin_state = None
        self.origin_action = None

        # Pandas version of state used for environment comparisons
        ch_list = [len(CHList)]

        for ch in range(len(CHList)):
            ch_list.append(CHList[ch])

        self.full_sensor_list = pd.DataFrame(np.array(ch_list))
        self.full_sensor_list.rename(
            columns={0: "Device"},
            inplace=True
        )

        # Battery Usage
        self.max_energy = 6_800  # 6800 mAh
        self.cpu_pow = 3.7  # milli-watts/ 2 micro Joule
        self.cpu_amps = 1_000  # micro-amps

        self.charge_rate = 1/3  # 20 min charging time
        self.flight_discharge = 0.75  # 45 min flight time
        self.flight_amps = self.max_energy / self.flight_discharge # A in 1 min
        self.launch_time = self.maxH / self.maxClimb # s
        self.launch_cost = 18.889 * self.launch_time / 60 # mA

        self.is_charging = False

        self.stored_energy = 100  # Initialize at full battery
        self.total_amp_spent = 0.0
        self.percent_gain = 0.0
        self.step_charge = 100 / (self.charge_rate * 60)
        self.step_discharge = 100 / (self.flight_discharge * 60)

        # State used for model
        """
        Old state +1, Date +2, New +1
        """
        self.state = [[0, 0, 0, 0] for _ in range(len(CHList) + 1)]

        self.state[0][0], self.state[0][1], self.state[0][2] = -1, 0, 100
        count = 0
        for row in range(len(self.state) - 1):
            self.state[row + 1][0] = count
            count += 1

        self.step_comms_cost = 0
        self.step_move_cost = 0
        self.energy_harvested = 0

    def reset(self):
        # Reset Flags
        self.stored_energy = 100
        self.total_amp_spent = 0.0
        self.percent_gain = 0.0

        self.crash = False
        self.model_transit = False
        self.is_charging = False
        self.no_hold = True
        self.force_change = False
        self.force_count = 0
        self.h = 1

        self.target = None
        self.targetHead = None
        self.last_Head = None

        self.tour = []
        self.tour_iter = 0

        self.bad_target = 0
        self.targetType = True

        self.step_move_cost = 0
        self.step_comms_cost = 0
        self.energy_harvested = 0
        self.last_AoI = 0
        self.p_cycle = 0
        self.p_count = 0

        # Reset State
        self.state[0][0], self.state[0][1], self.state[0][2] = -1, 0, 100

        for row in range(len(self.state) - 1):
            self.state[row + 1][1] = 0
            self.state[row + 1][2] = 0
            self.state[row + 1][3] = 0

    def battery_calc(self, stored_percentage, max_storage, amp_spent, step_charge):
        stored_percentage -= amp_spent / (max_storage * 60) # mAm / mAm
        stored_percentage = min(100.0, (stored_percentage + step_charge))

        return stored_percentage

    def step_battery(self):
        self.stored_energy = self.battery_calc(self.stored_energy, self.max_energy, self.total_amp_spent, self.percent_gain)

        self.state[0][2] = self.stored_energy
        return 0

    # Internal UAV Mechanics
    def navigate_step(self, env: object):
        self.energy_harvested = 0

        self.total_amp_spent += self.cpu_amps / 1_000
        if (math.sqrt(pow(self.id_x - self.target_x, 2) + pow(self.id_y - self.target_y, 2))
                < self._comms["max_dist_ambc"]):
            self.target_x, self.target_y = self.target.curr_pt

        maxDist = math.sqrt(pow(self.id_x - self.target_x, 2) + pow(self.id_y - self.target_y, 2))

        if maxDist < 1.0:
            if self.h == 1 and self.target.type == 2:
                self.h = 0
                self.total_amp_spent += self.launch_cost

        elif self.stored_energy > self.step_discharge:
            time = 0.0
            if self.h == 0:
                self.h = 1
                self.total_amp_spent += self.launch_cost
                time += self.launch_time

            if maxDist < self.maxSpd * ((60.0 - time) - self.launch_time):
                self.id_x = self.target_x
                self.id_y = self.target_y
                time = maxDist / self.maxSpd
                if self.target.type == 2:
                    self.h = 0
                    self.total_amp_spent += self.launch_cost

            else:
                time = (60.0 - time)
                vectAngle = math.atan(abs(self.target_y - self.id_y) / max(abs(self.target_x - self.id_x), 1))  # Returns radians
                directionX = (self.target_x - self.id_x) / max(abs(self.target_x - self.id_x), 1)
                directionY = (self.target_y - self.id_y) / max(abs(self.target_y - self.id_y), 1)
                self.id_x += math.floor(directionX * self.maxSpd * time * math.cos(vectAngle))
                self.id_y += math.floor(directionY * self.maxSpd * time * math.sin(vectAngle))

            self.total_amp_spent += self.flight_amps * time / 60

        else:
            self.crash = True

    # Finish with battery drain
    # UAV-IoT Communication
    def receive_data(self, step):
        totalData = 0
        device = self.target
        train_model = False
        change_archives = False

        if self.target.type == 1:
            dataReturn = max(0, device.upload(int(self.id_x), int(self.id_y)))

            if math.sqrt(pow((self.id_x - self.target.id_x), 2) + pow((self.id_y - self.target.id_y), 2)) < \
                    self._comms.get("max_dist_ambc"):

                totalData += dataReturn

                if self.tour_iter < len(self.tour):
                    self.target = self.tour[self.tour_iter]
                    self.tour_iter += 1
                    train_model, change_archives = False, False
                else:
                    # Assess Reward at this point
                    self.target = self.targetHead
                    self.targetType = True
                    self.p_cycle = 0
                    train_model, change_archives = False, False

                self.target_x = self.target.id_x
                self.target_y = self.target.id_y
                self.inRange = True

            else:
                self.inRange = False

            totalTime = totalData / self._comms.get("max_bitrate_ambc")
            self.total_amp_spent += (totalTime * self._comms.get("current_active_lora")
                                     + (60 - totalTime) * self._comms.get("current_sleep_lora")) / 60

            self.state[self.last_Head + 1][2] = step - self.last_AoI
            self.state[self.last_Head + 1][1] += totalData
            self.state[0][1] += totalData
            self.targetSerial = self.targetHead.cluster_num


        else:
            if math.sqrt(pow((self.id_x - self.target.id_x), 2) + pow((self.id_y - self.target.id_y), 2)) < \
                    self._comms.get("max_dist_lora"):

                self.inRange = True

                dataReturn, self.last_AoI, avg_AoI = device.upload(int(self.id_x), int(self.id_y))
                dataReturn = max(0, dataReturn)
                totalData += dataReturn

                totalTime = totalData / self._comms.get("max_bitrate_lora")
                self.total_amp_spent += (totalTime * self._comms.get("current_active_lora")
                                         + (60 - totalTime) * self._comms.get("current_sleep_lora")) / 60

                # ADF 2
                self.state[self.targetSerial + 1][2] = step - self.last_AoI
                self.state[self.targetSerial + 1][3] = step - avg_AoI
                self.state[self.targetSerial + 1][1] += totalData

                self.state[0][1] += totalData
            else:
                totalTime = 4
                self.total_amp_spent += (totalTime * self._comms.get("current_active_lora")
                                         + (60 - totalTime) * self._comms.get("current_sleep_lora")) / 60
                self.inRange = False

        for CH in range(len(self.full_sensor_list) - 1):
            if device.type == 1:
                self.state[CH + 1][2] += 1
                self.state[CH + 1][3] += 1
            elif device.cluster_num != self.full_sensor_list.iat[CH + 1, 0].cluster_num:
                self.state[CH + 1][2] += 1
                self.state[CH + 1][3] += 1
            elif not self.inRange:
                self.state[CH + 1][2] += 1
                self.state[CH + 1][3] += 1

        return train_model, change_archives

    def receive_energy(self):
        excess_percent = 0.0
        self.percent_gain = 0.0
        if self.target.type == 2 and self.h == 0:
            t = self.target.charge_time(int(self.id_x), int(self.id_y), self.is_charging)

            if (self.is_charging and abs(self.id_x-self.target_x) < 1.0 and
                    abs(self.id_y-self.target_y) < 1.0 and t < 1.0):
                self.no_hold = False

            if self.stored_energy < 80:
                self.percent_gain = 2 * self.step_charge * (t / 60)
            else:
                self.percent_gain = 0.5 * self.step_charge * (t / 60)

            excess_percent = self.stored_energy + self.percent_gain - 100

            self.energy_harvested = self.percent_gain

        return excess_percent

    def set_dest(self, model, model_p, step, _=None):
        train_model = False
        DCH = 0
        used_model = False
        train_p = False
        action_p = 0
        p_state = [0, 0, 0, 0]

        if self.targetHead is not None:
            self.last_Head = self.targetHead.cluster_num

        if self.target is None:
            minDist = 10_000.0
            print(self.full_sensor_list.shape)
            minCH = self.full_sensor_list.iat[1, 0]
            for CH in range(len(self.full_sensor_list.index) - 1):
                dist = math.sqrt(pow((self.id_x - self.full_sensor_list.iat[CH + 1, 0].id_x), 2)
                                 + pow((self.id_y - self.full_sensor_list.iat[CH + 1, 0].id_y), 2))
                if dist < minDist:
                    minDist = dist
                    minCH = self.full_sensor_list.iat[CH + 1, 0]

            self.target = minCH
            self.targetHead = minCH
            self.target_x = minCH.id_x
            self.target_y = minCH.id_y
            self.targetSerial = self.targetHead.cluster_num
            self.targetType = False
            DCH = 0

        elif not self.inRange:
            self.target = self.target
            self.p_cycle -= 1
            self.force_count += 1

        elif self.target.type == 1:
            self.target = self.target
            self.p_cycle -= 1
            self.force_count += 1

        else:
            ### TargetType > sensor
            self.force_change = True if self.force_count > 30 else False

            # True, True, sensor, CHstate, action, action_p
            (used_model, changed_transit, dest, state, action, action_p, p_state,
             dist, peak, avg, tour, self.targetType) =\
                self.target.get_dest(self.state, self.full_sensor_list, model, model_p,
                                     step, self.p_count, self.targetType, self.targetSerial, self.force_change)

            self.p_cycle -= 1

            if self.targetType:
                if dest.cluster_num == self.targetSerial:
                    self.force_count += 1
                else:
                    self.force_count = 1
            else:
                self.force_count += 1

            DCH = self.targetSerial
            self.action = action


            if self.model_transit and changed_transit:
                self.model_transit = False

            if self.h == 0:
                self.p_count -= 1


            """
            Old Charging: Static Charging
            """
            # if self.state[0][2] < 0.2 * self.max_energy * 1_000:
            #     used_model = False
            #     self.is_charging = True
            #     self.target = self.target
            #     return (train_model, DCH, used_model, state, action,
            #             self.step_comms_cost, self.step_move_cost, self.energy_harvested)
            #
            # elif self.is_charging and self.state[0][2] == self.max_energy * 1_000:
            #     self.is_charging = False
            """
            New Charging: Dynamic Charging
            """
            # if self.p_cycle < 1.0:
            #     self.p_cycle = 30
            #     """
            #     Dynamic determination of charging time.
            #     1) Emergency Charge when below threshold to 1.5 * needed
            #     2) If tour would result in emergency situation
            #     """
            #     if self.state[0][2] < 0.2 * self.max_energy * 1_000:
            #         self.p_count = min(20,
            #                            round(1.5 * (dist / self.maxSpd) * (self.flight_discharge / self.charge_rate)))
            #
            #     elif ((self.state[0][2] -
            #           1.25 * (dist / self.maxSpd) * (1_000 * self.max_energy / (self.flight_discharge * 60 * 60))) <=
            #           0.2 * self.max_energy * 1_000):
            #         self.p_count = min(20,
            #                            round(1.5 * (dist / self.maxSpd) * (self.flight_discharge / self.charge_rate)))
            #
            #     else:
            #         if peak <= 240 and self.state[0][2] < 0.8 * self.max_energy * 1_000:
            #             self.p_count = min(20,round(0.5*(dist/self.maxSpd)*(self.flight_discharge/self.charge_rate)))
            #         else:
            #             self.p_count = 0
            #
            #     if self.p_count > 0:
            #         self.is_charging = True
            #     else:
            #         self.is_charging = False
            """
            Model Charging
            """
            if action_p < self.p_count:
                self.p_count = math.ceil((self.p_count + action_p) / 2)

            if self.p_cycle < 1.0:
                self.p_cycle = 30
                self.p_count = action_p

                train_p = True
                self.is_charging = True

                if self.p_count > 0:
                    self.is_charging = True
                else:
                    self.is_charging = False
            """
            END
            """

            if changed_transit and self.p_count < 1.0:
                self.p_cycle = 0

            elif self.is_charging and (self.p_count < 1.0 or self.state[0][2] > 0.8 * self.max_energy * 1_000):
                self.p_count = 0
                self.is_charging = False



            if self.is_charging:
                used_model = False
                self.target = self.target
                return (train_model, DCH, used_model, train_p, state, action, action_p, p_state,
                        self.step_comms_cost, self.step_move_cost, self.energy_harvested)

            else:
                self.no_hold = True
                self.is_charging = False
                if used_model:
                    self.model_transit = True


                if dest.type == 1:
                    """
                    Old targeting
                    """
                    # self.tour = []
                    # self.tour_iter = 1
                    """
                    New targeting
                    """
                    self.tour = tour
                    self.tour_iter = 1

                    self.target = dest
                    self.target_x = dest.id_x
                    self.target_y = dest.id_y

                    return (train_model, DCH, used_model, train_p, state, action, action_p, p_state,
                            self.step_comms_cost, self.step_move_cost, self.energy_harvested)

                else:
                    if dest.cluster_num == self.targetSerial:
                        self.targetType = True
                        self.bad_target = 1
                    else:
                        self.targetType = False
                        self.bad_target = 0

                    if changed_transit:
                        train_model = True

                    self.target = dest
                    self.targetHead = dest
                    self.targetSerial = self.targetHead.cluster_num
                    self.target_x = dest.id_x
                    self.target_y = dest.id_y

                    return (train_model, DCH, used_model, train_p, state, action, action_p, p_state,
                            self.step_comms_cost, self.step_move_cost, self.energy_harvested)

        return (train_model, DCH, used_model, train_p, self.state, self.targetHead.cluster_num, action_p, p_state,
                self.step_comms_cost, self.step_move_cost, self.energy_harvested)