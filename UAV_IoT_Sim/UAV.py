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
            "LoRa_Current_mA": 0.8,
            "LoRa_Voltage_V": 3.7,
            "LoRa_Power_W": 0.8 * 3.7,

            "AmBC_Max_Distance_m": 500,
            "AmBC_Bit_Rate_bit/s": 1592,
            "AmBC_Power_W": 0.00352 / 1000,
            "AmBC_Voltage_V": 3.3,
            "AmBC_Current_mA": 0.00352 / 3.3
        }

        # Positioning
        self.indX = float(X)
        self.indY = float(Y)
        self.maxH = 20

        # Trajectory
        self.target = None
        self.targetHead = None
        self.targetX = float(X)
        self.targetY = float(Y)
        self.lat = lat
        self.long = long

        # Movement
        self.maxSpd = 15  # 15 m/s max speed cap
        self.maxTurn = 200  # 200 degree per sec max yaw
        self.maxClimb = 3  # 3 m/s ascent and descent
        self.inRange = True

        # State
        self.crash = False
        self.model_transit = False
        self.origin_state = None
        self.origin_action = None

        # Pandas version of state used for environment comparisons
        self.uav = np.array([[self, len(CHList), float(0.0), float(0.0)]])
        self.CH_state = np.concatenate((np.array(CHList), np.array([[0, 0]] * (len(CHList)))), axis=1)
        self.full_state = pd.DataFrame(np.concatenate((self.uav, self.CH_state)))

        self.full_state = self.full_state.sort_index()
        self.full_state = self.full_state.reset_index()
        self.full_state.drop('index', axis=1, inplace=True)
        self.full_state.rename(
            columns={0: "Device", 1: "Num_Sensors", 2: "Total_Data", 3: "AoI"},
            inplace=True
        )

        # Battery Usage
        self.max_energy = 6800  # 6800 mAh
        self.charge_rate = 2.5  # 150 min charging time
        self.flight_discharge = 0.5  # 30 min flight time
        self.amp = self.max_energy / self.charge_rate  # Roughly 2.72 A optimal current
        self.stored_energy = self.max_energy  # Initialize at full battery
        self.is_charging = False

        # State used for model
        self.state = [[0, 0, 0] for _ in range(len(CHList) + 6)]
        self.state[0][0], self.state[0][1], self.state[0][2] = -1, 0, self.max_energy
        count = 0
        for row in range(len(self.state) - 1):
            self.state[row + 1][0] = count
            count += 1

        self.step_comms_cost = 0
        self.step_move_cost = 0
        self.energy_harvested = 0

    def reset(self):
        # Reset Flags
        self.stored_energy = self.max_energy
        self.crash = False
        self.model_transit = False

        # Reset State
        self.state = [[0, 0, 0] for _ in range(len(self.state))]
        self.state[0][0], self.state[0][1], self.state[0][2] = -1, 0, self.max_energy
        self.full_state[0, 2] = 0
        self.full_state[0, 3] = 0
        count = 0
        for row in range(len(self.state) - 1):
            self.state[row + 1][0] = count
            self.full_state[row, 2] = 0
            self.full_state[row, 3] = 0
            count += 1

    # Internal UAV Mechanics
    def navigate_step(self, env: object):
        maxDist = math.sqrt(pow(self.indX - self.targetX, 2) + pow(self.indY - self.targetY, 2))
        self.step_move_cost = 0

        if round(self.targetX) == round(self.indX) and round(self.targetY) == round(self.indY):
            pass
        elif self.stored_energy >= ((self.max_energy / self.flight_discharge) / 60):

            if maxDist <= self.maxSpd * 60:
                env.moveUAV(round(self.indX), round(self.indY), round(self.targetX), round(self.targetY))
                self.indX = self.targetX
                self.indY = self.targetY

            else:
                vectAngle = math.atan((self.targetY - self.indY) / (self.targetX - self.indX))  # Returns radians
                direction = (self.targetX - self.indX) / abs(self.targetX - self.indX)
                env.moveUAV(round(self.indX), round(self.indY),
                            math.floor(self.indX + direction * self.maxSpd * 60 * math.cos(vectAngle)),
                            math.floor(self.indY + direction * self.maxSpd * 60 * math.sin(vectAngle))
                            )
                self.indX += direction * self.maxSpd * 60 * math.cos(vectAngle)
                self.indY += direction * self.maxSpd * 60 * math.sin(vectAngle)

            self.energy_cost(60, 0, 0)

        else:
            self.crash = True

    def energy_cost(self, flight: float = 0.0, lora: float = 0.0, ambc: float = 0.0):

        total_cost = 0
        # Cost of air travel
        total_cost += flight * ((self.max_energy / self.flight_discharge) / (60 * 60))
        # Cost of AmBC
        total_cost += lora * self._comms.get("LoRa_Current_mA")
        # Cost of LoRa
        total_cost += ambc * self._comms.get("AmBC_Current_mA")

        self.step_move_cost = flight * ((self.max_energy / self.flight_discharge) / (60 * 60))
        self.step_comms_cost += lora * self._comms.get("LoRa_Current_mA") + ambc * self._comms.get("AmBC_Current_mA")

        self.stored_energy -= total_cost
        self.state[0][2] = self.stored_energy
        self.full_state.iloc[0, 2] = self.stored_energy

    # Finish with battery drain
    # UAV-IoT Communication
    def receive_data(self, step):
        totalData = 0
        device = self.target
        train_model = False
        change_archives = False
        self.step_comms_cost = 0

        if self.target.type == 1:
            dataReturn = max(0, device.ws_upload_data(int(self.indX), int(self.indY)))

            if math.sqrt(pow((self.indX - self.target.indX), 2) + pow((self.indY - self.target.indY), 2)) < \
                    self._comms.get("AmBC_Max_Distance_m"):
                totalData += dataReturn
                self.target = self.targetHead
                self.targetX = self.targetHead.indX
                self.targetY = self.targetHead.indY
                train_model, change_archives = True, True

            else:
                self.inRange = False

            totalTime = totalData / self._comms.get("AmBC_Bit_Rate_bit/s", 0.0)
            self.energy_cost(0, totalTime, totalTime)

            self.update_state(self.targetHead.headSerial + 1, step, totalData)
            self.state[self.targetHead.headSerial + 1][2] = step
            self.state[self.targetHead.headSerial + 1][1] += totalData / 1000
            self.state[0][1] += totalData / 1000


        else:
            dataReturn = max(0, device.ch_upload(int(self.indX), int(self.indY)))

            if (self._comms.get("LoRa_Bit_Rate_bit/s") * 56) > dataReturn >= 0:
                if math.sqrt(pow((self.indX - self.target.indX), 2) + pow((self.indY - self.target.indY), 2) < \
                             self._comms.get("LoRa_Max_Distance_m")):
                    totalData += max(0, dataReturn)
                else:
                    self.inRange = False
            else:
                totalData += max(0, dataReturn)
                self.inRange = True

            totalTime = totalData / self._comms.get("LoRa_Bit_Rate_bit/s")
            self.energy_cost(0, totalTime, 0)

            self.update_state(device.headSerial + 1, step, totalData)
            self.state[device.headSerial + 1][2] = step
            self.state[device.headSerial + 1][1] += totalData / (1000000 * 8)
            self.state[0][1] += totalData / (1000000 * 8)

        return train_model, change_archives

    def receive_energy(self):
        self.energy_harvested = 0
        if self.target.type == 2:
            t = self.target.charge_time(int(self.indX), int(self.indY))

            self.stored_energy += t * (self.max_energy / (self.charge_rate * 60 * 60))
            self.energy_harvested = t * (self.max_energy / (self.charge_rate * 60 * 60))
            self.state[0][2] = self.stored_energy
            self.full_state.iloc[0, 3] = self.stored_energy

    def set_dest(self, model, step, _=None):
        train_model = False
        used_model = False

        if self.target is None:
            minDist = 10000
            minCH = self.full_state.iloc[1, 0]
            for CH in range(len(self.full_state) - 1):
                dist = math.sqrt(pow((self.indX - self.full_state.iloc[CH + 1, 0].indX), 2) \
                                 + pow((self.indY - self.full_state.iloc[CH + 1, 0].indY), 2))
                if dist < minDist:
                    minDist = dist
                    minCH = self.full_state.iloc[CH + 1, 0]

            self.target = minCH
            self.targetHead = minCH
            self.targetX = minCH.indX
            self.targetY = minCH.indY

        elif self.target.type == 1:
            self.target = self.target

        elif self.stored_energy < (self.max_energy * .30):
            self.is_charging = True
            self.target = self.targetHead

        elif self.is_charging and self.stored_energy > (self.max_energy * .80):
            self.is_charging = False
            self.target = self.target


        # Here model_transit will change
        # Investigate what is needed on return
        else:
            used_model, changed_transit, dest1, dest2, state1, state2, action1, action2 = \
                self.target.get_dest(self.state, self.full_state, model, step)

            if self.model_transit and changed_transit:
                train_model = True

            self.state = state1
            self.action = action1
            if used_model:
                self.model_transit = True
            else:
                self.model_transit = False

            if dest1.type == 1:
                self.targetHead = dest2
                self.target = dest1
                self.targetX = dest1.indX
                self.targetY = dest1.indY
                return (train_model, used_model, state1, action1, \
                        self.step_comms_cost, self.step_move_cost, self.energy_harvested)

            else:
                self.target = dest1
                self.targetHead = dest1
                self.targetX = dest1.indX
                self.targetY = dest1.indY
                return (train_model, used_model, state1, action1, \
                        self.step_comms_cost, self.step_move_cost, self.energy_harvested)

        return (train_model, used_model, self.state, self.targetHead.headSerial, \
                self.step_comms_cost, self.step_move_cost, self.energy_harvested)

    def update_state(self, device, step, data):
        self.full_state.iloc[device, 3] = step
        self.full_state.iloc[device, 2] += data
        self.full_state.iloc[0, 2] += data
