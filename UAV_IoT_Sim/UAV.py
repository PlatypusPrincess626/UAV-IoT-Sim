# Dependencies to Import
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
            "LoRa_Max_Distance_m": 5000,
            "LoRa_Bit_Rate_bit/s": 24975,
            "LoRa_Current_A": 5400,  # micro-amps transmitting
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

        # Positioning
        self.indX = int(X)
        self.indY = int(Y)
        self.maxH = 20
        self.h = 1

        # Trajectory
        self.target = None
        self.targetHead = None
        self.last_Head = None
        self.targetSerial = 0
        self.tour = None
        self.tour_iter = 0

        self.targetX = int(X)
        self.targetY = int(Y)
        self.lat = lat
        self.long = long
        self.last_AoI = 0
        self.bad_target = False

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
        self.targetType = 0

        self.origin_state = None
        self.origin_action = None

        # Pandas version of state used for environment comparisons
        ch_list = []
        sensor_list = []
        ch_list.append(self)
        sensor_list.append(len(CHList))

        for ch in range(len(CHList)):
            ch_list.append(CHList[ch][0])
            sensor_list.append(CHList[ch][1])

        self.full_sensor_list = pd.DataFrame(np.array(ch_list))
        self.sensor_count = pd.DataFrame(np.array(sensor_list))
        self.full_sensor_list.rename(
            columns={0: "Device"},
            inplace=True
        )

        # Battery Usage
        self.max_energy = 6_800  # 6800 mAh
        self.cpu_pow = 3.7  # milli-watts/ 2 micro Joule
        self.cpu_amps = 1_000 # micro-amps

        self.charge_rate = 1/3  # 20 min charging time
        self.flight_discharge = 1 # 45 min flight time
        self.amp = self.max_energy / self.charge_rate  # Roughly 2.72 A optimal current
        self.stored_energy = self.max_energy * 1_000  # Initialize at full battery
        self.is_charging = False
        self.launch_cost = 18.889 # mA

        # State used for model
        self.state = [[0, 0, 0, 0] for _ in range(len(CHList) + 1)]

        self.state[0][0], self.state[0][1], self.state[0][2] = -1, 0, self.max_energy * 1_000
        count = 0
        for row in range(len(self.state) - 1):
            self.state[row + 1][0] = count
            count += 1

        self.step_comms_cost = 0
        self.step_move_cost = 0
        self.energy_harvested = 0

    def reset(self):
        # Reset Flags
        self.stored_energy = self.max_energy * 1_000

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
        self.tour = None
        self.tour_iter = 0
        self.bad_target = False
        self.targetType = 0

        self.step_move_cost = 0
        self.step_comms_cost = 0
        self.energy_harvested = 0
        self.last_AoI = 0
        self.p_cycle = 0
        self.p_count = 0

        # Reset State
        self.state[0][0], self.state[0][1], self.state[0][2] = -1, 0, self.max_energy * 1_000

        for row in range(len(self.state) - 1):
            self.state[row + 1][1] = 0
            self.state[row + 1][2] = 0
            self.state[row + 1][3] = 0

    # Internal UAV Mechanics
    def navigate_step(self, env: object):
        self.step_move_cost = round(self.cpu_amps)
        self.step_comms_cost = round(self._comms.get("AmBC_Current_A") + self._comms.get("Lora_Upkeep_A"))
        self.energy_harvested = 0

        power_upkeep = round(self.cpu_amps + self._comms.get("AmBC_Current_A") +
                        self._comms.get("Lora_Upkeep_A"))
        self.stored_energy -= power_upkeep

        maxDist = math.sqrt(pow(self.indX - self.targetX, 2) + pow(self.indY - self.targetY, 2))

        if maxDist < 1.0:
            if self.h == 1 and self.target.type == 2:
                self.h = 0
                self.energy_cost(0, 0, 1)

        elif self.stored_energy > (1_000 * self.max_energy / (self.flight_discharge * 60)):
            if self.h == 0:
                self.h = 1
                self.energy_cost(0, 0, 1)

            if maxDist < self.maxSpd * 60:
                env.moveUAV(round(self.indX), round(self.indY), round(self.targetX), round(self.targetY))
                self.indX = self.targetX
                self.indY = self.targetY
                time = maxDist / self.maxSpd
                if self.target.type == 2:
                    self.h = 0
                    self.energy_cost(0, 0, 1)

            else:
                time = 60
                vectAngle = math.atan(abs(self.targetY - self.indY) / max(abs(self.targetX - self.indX), 1))  # Returns radians
                directionX = (self.targetX - self.indX) / max(abs(self.targetX - self.indX), 1)
                directionY = (self.targetY - self.indY) / max(abs(self.targetY - self.indY), 1)
                env.moveUAV(round(self.indX), round(self.indY),
                            math.floor(self.indX + directionX * self.maxSpd * 60 * math.cos(vectAngle)),
                            math.floor(self.indY + directionY * self.maxSpd * 60 * math.sin(vectAngle))
                            )
                self.indX += math.floor(directionX * self.maxSpd * 60 * math.cos(vectAngle))
                self.indY += math.floor(directionY * self.maxSpd * 60 * math.sin(vectAngle))

            self.energy_cost(time, 0, 0)

        else:
            self.crash = True

    def energy_cost(self, flight: float = 0.0, lora: float = 0.0, launch: float = 0.0):
        total_cost = 0
        # Cost of air travel
        total_cost += round(flight * 1_000 * self.max_energy / (self.flight_discharge * 60 * 60))
        # Cost of LoRa
        total_cost += round((lora/max(lora, 1)) * self._comms.get("LoRa_Current_A"))
        # Cost to Launch
        total_cost += round(launch * self.launch_cost * 1_000)

        self.step_move_cost += round(flight * 1_000 * self.max_energy / (self.flight_discharge * 60 * 60) +
                                     round(launch * self.launch_cost * 1_000))
        self.step_comms_cost += round(self._comms.get("LoRa_Current_A"))

        self.stored_energy -= total_cost
        self.state[0][2] = self.stored_energy

    # Finish with battery drain
    # UAV-IoT Communication
    def receive_data(self, step):
        totalData = 0
        device = self.target
        train_model = False
        change_archives = False

        if self.target.type == 1:
            dataReturn = max(0, device.ws_upload_data(int(self.indX), int(self.indY)))

            if math.sqrt(pow((self.indX - self.target.indX), 2) + pow((self.indY - self.target.indY), 2)) < \
                    self._comms.get("AmBC_Max_Distance_m"):

                totalData += dataReturn

                if self.tour_iter < len(self.tour):
                    self.target = self.tour[self.tour_iter]
                    self.tour_iter += 1
                else:
                    self.target = self.targetHead

                self.targetX = self.targetHead.indX
                self.targetY = self.targetHead.indY
                self.inRange = True
                train_model, change_archives = True, True

            else:
                self.inRange = False

            totalTime = totalData / self._comms.get("AmBC_Bit_Rate_bit/s")
            self.energy_cost(0, totalTime, 0)

            self.state[self.last_Head + 1][2] = step - self.last_AoI
            self.state[self.last_Head + 1][1] += totalData
            self.state[0][1] += totalData
            self.targetSerial = self.targetHead.headSerial


        else:
            if math.sqrt(pow((self.indX - self.target.indX), 2) + pow((self.indY - self.target.indY), 2)) < \
                    self._comms.get("LoRa_Max_Distance_m"):

                self.inRange = True

                dataReturn, self.last_AoI, avg_AoI = device.ch_upload(int(self.indX), int(self.indY))
                dataReturn = max(0, dataReturn)
                totalData += dataReturn

                totalTime = totalData / self._comms.get("LoRa_Bit_Rate_bit/s")
                self.energy_cost(0, totalTime, 0)

                # ADF 2
                self.state[self.targetSerial + 1][2] = step - self.last_AoI
                self.state[self.targetSerial + 1][3] = step - avg_AoI
                self.state[self.targetSerial + 1][1] += totalData
                self.state[0][1] += totalData

            else:
                self.inRange = False

        for CH in range(len(self.full_sensor_list) - 1):
            if device.type == 1:
                self.state[CH + 1][2] += 1
                self.state[CH + 1][3] += 1
            elif device.headSerial != self.full_sensor_list.iat[CH + 1, 0].headSerial:
                self.state[CH + 1][2] += 1
                self.state[CH + 1][3] += 1
            elif not self.inRange:
                self.state[CH + 1][2] += 1
                self.state[CH + 1][3] += 1

        return train_model, change_archives

    def receive_energy(self):
        excess_percent = 1.0
        if self.target.type == 2 and self.h == 0:
            t = self.target.charge_time(int(self.indX), int(self.indY), self.is_charging)

            if (self.is_charging and abs(self.indX-self.targetX) < 1.0 and
                    abs(self.indY-self.targetY) < 1.0 and t < 1.0):
                self.no_hold = False

            if self.stored_energy < (0.8 * 1_000 * self.max_energy):
                self.stored_energy += round(2 * t * 1_000 * (self.max_energy / ((self.charge_rate) * 60 * 60)))
            else:
                self.stored_energy += round(0.5 * t * 1_000 * (self.max_energy / ((self.charge_rate) * 60 * 60)))

            excess_percent = (self.stored_energy / (self.max_energy*1_000))
            if self.stored_energy > self.max_energy * 1_000:
                excess_percent = 0
                self.stored_energy = self.max_energy * 1_000

            self.energy_harvested += round(t * 1_000 * (self.max_energy / (self.charge_rate * 60 * 60)))
            self.state[0][2] = self.stored_energy

        return excess_percent

    def set_dest(self, model, model_p, step, _=None):
        train_model = False
        used_model = False
        train_p = False
        action_p = 0
        p_state = [0, 0, 0, 0]

        if self.targetHead is not None:
            self.last_Head = self.targetHead.headSerial

        if self.target is None:
            minDist = 10_000.0
            minCH = self.full_sensor_list.iat[1, 0]
            for CH in range(len(self.full_sensor_list.index) - 1):
                dist = math.sqrt(pow((self.indX - self.full_sensor_list.iat[CH + 1, 0].indX), 2)
                                 + pow((self.indY - self.full_sensor_list.iat[CH + 1, 0].indY), 2))
                if dist < minDist:
                    minDist = dist
                    minCH = self.full_sensor_list.iat[CH + 1, 0]

            self.target = minCH
            self.targetHead = minCH
            self.targetX = minCH.indX
            self.targetY = minCH.indY
            self.targetSerial = self.targetHead.headSerial

        elif not self.inRange:
            self.target = self.target

        elif self.target.type == 1:
            self.target = self.target

        else:
            ### TargetType > sensor
            self.targetType = 1 - self.targetType
            if self.bad_target:
                self.targetType = 1

            self.force_change = False
            # True, True, sensor, CHstate, action, action_p
            used_model, changed_transit, dest, state, action, action_p, p_state = \
                self.target.get_dest(self.state, self.full_sensor_list, model, model_p, step, self.p_count,
                                     self.targetType, self.targetSerial)

            self.p_cycle -= 1
            self.action = action
            if action_p < self.p_count:
                self.p_count = math.ceil((self.p_count + action_p) / 2)

            if changed_transit and self.p_count < 1.0:
                self.p_cycle = 0

            if self.bad_target:
                self.bad_target = False

            if self.targetType == 1:
                if dest.headSerial == self.targetSerial:
                    self.bad_target = True

            if self.model_transit and changed_transit:
                train_model = True
                self.model_transit = False

            if self.is_charging and (self.p_count < 1.0 or self.state[0][2] == self.max_energy * 1_000):
                self.is_charging = False

            if self.h == 0:
                self.p_count -= 1

            if self.p_cycle < 1.0 and self.p_count < 1.0 and not self.bad_target:
                self.p_cycle = 30
                self.p_count = action_p
                train_p = True
                used_model = False
                self.is_charging = True
                self.target = self.target
                return (train_model, used_model, train_p, state, action, action_p, p_state,
                        self.step_comms_cost, self.step_move_cost, self.energy_harvested)

            elif self.is_charging:
                used_model = False
                self.target = self.target
                return (train_model, used_model, train_p, state, action, action_p, p_state,
                        self.step_comms_cost, self.step_move_cost, self.energy_harvested)

            else:
                self.no_hold = True
                self.is_charging = False
                if used_model:
                    self.model_transit = True

                if dest.type == 1:
                    self.tour = self.target.tour
                    self.tour_iter = 1
                    self.target = dest
                    self.targetX = dest.indX
                    self.targetY = dest.indY

                    return (train_model, used_model, train_p, state, action, action_p, p_state,
                            self.step_comms_cost, self.step_move_cost, self.energy_harvested)

                else:
                    self.target = dest
                    self.targetHead = dest
                    self.targetSerial = self.targetHead.headSerial
                    self.targetX = dest.indX
                    self.targetY = dest.indY
                    return (train_model, used_model, train_p, state, action, action_p,  p_state,
                            self.step_comms_cost, self.step_move_cost, self.energy_harvested)

        return (train_model, used_model, train_p, self.state, self.targetHead.headSerial, action_p,  p_state,
                self.step_comms_cost, self.step_move_cost, self.energy_harvested)

