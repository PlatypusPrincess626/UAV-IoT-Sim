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

            "AmBC_Max_Distance_m": 500,
            "AmBC_Bit_Rate_bit/s": 1592,
            "AmBC_Power_W": 0.00259,  # upkeep
            "AmBC_Voltage_V": 3.3,
            "AmBC_Current_A": 785  # micro-amps upkeep
        }

        # Positioning
        self.indX = int(X)
        self.indY = int(Y)
        self.maxH = 20

        # Trajectory
        self.target = None
        self.targetHead = None
        self.targetSerial = 0
        self.targetX = int(X)
        self.targetY = int(Y)
        self.lat = lat
        self.long = long
        self.last_AoI = 0

        # Movement
        self.maxSpd = 15  # 15 m/s max speed cap
        self.maxTurn = 200  # 200 degree per sec max yaw
        self.maxClimb = 3  # 3 m/s ascent and descent
        self.inRange = True

        # State
        self.crash = False
        self.model_transit = False
        self.no_hold = True
        self.force_change = False
        self.force_count = 0

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

        self.charge_rate = 1  # 60 min charging time
        self.flight_discharge = 0.5  # 30 min flight time
        self.amp = self.max_energy / self.charge_rate  # Roughly 2.72 A optimal current
        self.stored_energy = self.max_energy * 1_000  # Initialize at full battery
        self.is_charging = False

        # State used for model
        #   ADF 2.0
        # self.state = [[0, 0, 0] for _ in range(len(CHList) + 6)]
        #   ADF 1.0
        self.state = [[0, 0, 0] for _ in range(len(CHList) + 1)]

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

        self.target = None
        self.targetHead = None
        self.step_move_cost = 0
        self.step_comms_cost = 0
        self.energy_harvested = 0
        self.last_AoI = 0

        # Reset State
        self.state[0][0], self.state[0][1], self.state[0][2] = -1, 0, self.max_energy * 1_000

        for row in range(len(self.state) - 1):
            self.state[row + 1][1] = 0
            self.state[row + 1][2] = 0

    # Internal UAV Mechanics
    def navigate_step(self, env: object):
        self.step_move_cost = round(self.cpu_amps)
        self.step_comms_cost = round(self._comms.get("AmBC_Current_A") + self._comms.get("Lora_Upkeep_A"))
        self.energy_harvested = 0

        power_upkeep = round(self.cpu_amps + self._comms.get("AmBC_Current_A") +
                        self._comms.get("Lora_Upkeep_A"))
        self.stored_energy -= power_upkeep

        maxDist = math.sqrt(pow(self.indX - self.targetX, 2) + pow(self.indY - self.targetY, 2))

        if abs(self.targetX - self.indX) < 1.0 and abs(self.targetY - self.indY) < 1.0:
            self.energy_cost(0, 0)

        elif self.stored_energy > (1_000 * self.max_energy / (self.flight_discharge * 60)):

            if maxDist <= self.maxSpd * 60:
                env.moveUAV(round(self.indX), round(self.indY), round(self.targetX), round(self.targetY))
                self.indX = self.targetX
                self.indY = self.targetY
                time = maxDist / (self.maxSpd * 60)

            else:
                vectAngle = math.atan((self.targetY - self.indY) / (self.targetX - self.indX))  # Returns radians
                direction = (self.targetX - self.indX) / abs(self.targetX - self.indX)
                env.moveUAV(round(self.indX), round(self.indY),
                            math.floor(self.indX + direction * self.maxSpd * 60 * math.cos(vectAngle)),
                            math.floor(self.indY + direction * self.maxSpd * 60 * math.sin(vectAngle))
                            )
                self.indX += direction * self.maxSpd * 60 * math.cos(vectAngle)
                self.indY += direction * self.maxSpd * 60 * math.sin(vectAngle)
                time = maxDist / (self.maxSpd * 60)

            self.energy_cost(time, 0)

        else:
            self.crash = True

    def energy_cost(self, flight: float = 0.0, lora: float = 0.0):
        total_cost = 0
        # Cost of air travel
        total_cost += round(flight * 1_000 * (self.max_energy / (self.flight_discharge * 60 * 60)))
        # Cost of LoRa
        total_cost += round(lora * self._comms.get("LoRa_Current_A"))

        self.step_move_cost += round(flight * 1_000 * (self.max_energy / (self.flight_discharge * 60 * 60)))
        self.step_comms_cost += round(lora * self._comms.get("LoRa_Current_A"))

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
                self.target = self.targetHead
                self.targetX = self.targetHead.indX
                self.targetY = self.targetHead.indY
                self.inRange = True
                train_model, change_archives = True, True

            else:
                self.inRange = False

            totalTime = totalData / self._comms.get("AmBC_Bit_Rate_bit/s", 0.0)
            self.energy_cost(0, totalTime)

            self.state[self.targetSerial + 1][2] = self.last_AoI
            self.state[self.targetSerial + 1][1] += totalData
            self.state[0][1] += totalData
            self.targetSerial = self.targetHead.headSerial


        else:
            if math.sqrt(pow((self.indX - self.target.indX), 2) + pow((self.indY - self.target.indY), 2)) < \
                    self._comms.get("AmBC_Max_Distance_m"):

                self.inRange = True

                dataReturn, self.last_AoI = device.ch_upload(int(self.indX), int(self.indY))
                dataReturn = max(0, dataReturn)
                totalData += dataReturn

                totalTime = totalData / self._comms.get("LoRa_Bit_Rate_bit/s")
                self.energy_cost(0, totalTime)

                self.state[self.targetSerial + 1][2] = self.last_AoI
                self.state[self.targetSerial + 1][1] += totalData
                self.state[0][1] += totalData

            else:
                self.inRange = False

        return train_model, change_archives

    def receive_energy(self):
        if self.target.type == 2:
            t = self.target.charge_time(int(self.indX), int(self.indY), self.is_charging)

            if self.is_charging and t < 60.0:
                self.no_hold = False

            self.stored_energy += round(t * 1_000 * (self.max_energy / (self.charge_rate * 60 * 60)))
            if self.stored_energy >= self.max_energy * 1_000:
                self.stored_energy = self.max_energy * 1_000

            self.energy_harvested += round(t * 1_000 * (self.max_energy / (self.charge_rate * 60 * 60)))
            self.state[0][2] = self.stored_energy

    def set_dest(self, model, step, _=None):
        print(self.state)
        train_model = False
        used_model = False
        print(self.force_change)

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

        elif self.stored_energy < (self.max_energy * .30 * 1_000) and self.no_hold:
            self.is_charging = True
            self.target = self.targetHead

        elif self.is_charging and self.stored_energy > (self.max_energy * .60 * 1_000) and self.no_hold:
            self.is_charging = False
            self.target = self.target

        elif self.target.type == 1:
            self.target = self.target

        # Here model_transit will change
        #   ADF 2.0
        # else:
        #     # False, True, full_state.iat[CH + 1, 0], _, state, _, CH, _
        #     used_model, changed_transit, dest1, dest2, state1, state2, action1, action2 = \
        #         self.target.get_dest(self.state, self.full_sensor_list, model, step,
        #                              self.no_hold, self.force_change, self.targetSerial)
        #
        #     self.no_hold = True
        #
        #     if self.model_transit and changed_transit:
        #         train_model = True
        #
        #     self.state = state1
        #     self.action = action1
        #     if used_model:
        #         self.model_transit = True
        #     else:
        #         self.model_transit = False
        #
        #     if dest1.type == 1:
        #         if dest2.headSerial == self.targetSerial and self.inRange:
        #             self.force_count += 1
        #
        #         elif self.force_change:
        #             self.force_change = False
        #             self.force_count = 0
        #
        #         if self.force_count > 30:
        #             self.force_change = True
        #
        #         self.targetSerial = self.targetHead.headSerial
        #         self.targetHead = dest2
        #         self.target = dest1
        #         self.targetX = dest1.indX
        #         self.targetY = dest1.indY
        #         return (train_model, used_model, state1, action1,
        #                 self.step_comms_cost, self.step_move_cost, self.energy_harvested)
        #
        #     else:
        #         if dest1.headSerial == self.targetSerial and self.inRange:
        #             self.force_count += 1
        #
        #         elif self.force_change:
        #             self.force_change = False
        #             self.force_count = 0
        #
        #         if self.force_count > 10:
        #             self.force_change = True
        #
        #         self.target = dest1
        #         self.targetHead = dest1
        #         self.targetSerial = self.targetHead.headSerial
        #         self.targetX = dest1.indX
        #         self.targetY = dest1.indY
        #         return (train_model, used_model, state1, action1,
        #                 self.step_comms_cost, self.step_move_cost, self.energy_harvested)

        #   ADF 1.0
        else:
            used_model, changed_transit, dest, state, action = \
                self.target.get_dest(self.state, self.full_sensor_list, model, step)

            self.no_hold = True
            if self.model_transit and changed_transit:
                train_model = True

            self.state = state
            self.action = action
            if used_model:
                self.model_transit = True
            else:
                self.model_transit = False

            self.target = dest
            self.targetHead = dest
            self.targetX = dest.indX
            self.targetY = dest.indY
            return (train_model, used_model, state, action,
                    self.step_comms_cost, self.step_move_cost, self.energy_harvested)

        return (train_model, used_model, self.state, self.targetHead.headSerial,
                self.step_comms_cost, self.step_move_cost, self.energy_harvested)

