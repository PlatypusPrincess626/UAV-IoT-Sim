# Dependecies to Import
import pandas as pd
import numpy as np

# Simulation Files to Import
from UAV_IoT_Sim import Environment, IoT_Device, UAV


class make_env:
    def __init__(
        self,
        scene: str = "test",
        num_sensors: int = 50,
        num_ch: int = 5,
        num_uav: int = 1,
        max_num_steps: int = 720
    ):
        self.scene = scene
        self._env = Environment.sim_env(scene, num_sensors, num_uav, num_ch, max_num_steps)
        self.dim = self._env.dim

        self.sensX = self._env.xPts
        self.sensY = self._env.yPts
        self.chX = self._env.chX
        self.chY = self._env.chY
        self.uavX = 0
        self.uavY = 0

        self._num_sensors = num_sensors
        self.num_ch = num_ch
        self._num_uav = num_uav
        self.max_num_steps = max_num_steps
        self.dims = self._env.dim
        
        self.curr_step = 0
        self.last_action = 0
        self.archived_action = 0
        self.archived_paction = 0
        self.deciding_CH = 0

        self.full_reward = 0.0
        self.accum_reward = 0.0
        self.accum_steps = 0.0
        self.track_reward = False
        self.total_average = 0

        self.curr_state = [[0, 0, 0, 0] for _ in range(self.num_ch + 1)]
        self.archived_state = [[0, 0, 0, 0] for _ in range(self.num_ch + 1)]
        self.rewards = [0.0, 0.0, 0.0]
        self.accum_rewards = [0.0, 0.0, 0.0]
        self.archived_rewards = [0.0, 0.0, 0.0]

        self.full_reward2 = 0.0
        self.accum_reward_p = 0.0
        self.accum_steps_p = 0.0
        self.track_reward_p = False

        self.curr_pstate = [0, 0, 0, 0]
        self.archived_pstate = [0, 0, 0, 0]
        self.accum_rewardsp = [0.0, 0.0, 0.0]
        self.archived_rewardsp = [0.0, 0.0, 0.0]

        self.ch_sensors = [0 for _ in range(self.num_ch)]
        for CH in range(self.num_ch):
            self.ch_sensors[CH] = self._env.CHTable.iat[CH, 0].num_sensors

        self.curr_reward = 0
        self.reward2 = 0

        self.curr_info = {
            "Last_Action": None,
            "Reward_Change": 0.0,          # -> Change in reward at step
            "Avg_Age": 0.0,               # -> avgAoI
            "Peak_Age": 0.0,              # -> peakAoI
            "Data_Distribution": 0.0,     # -> Distribution of Data
            "Total_Data_Change": 0.0,      # -> Change in Total Data
            "Total_Data": 0.0,            # -> Total Data
            "Crashed": False,
            "Truncated": False
        }

        self._aoi_threshold = 120
        self.truncated = False
        self.terminated = False
        self._curr_total_data = 0
    
    def reset(self):
        if self.scene == "test":
            for sensor in range(self._num_sensors):
                self._env.sensorTable.iat[sensor, 0].reset()
            for CH in range(self.num_ch):
                self._env.CHTable.iat[CH, 0].reset()
            for uav in range(self._num_uav):
                self._env.UAVTable.iat[uav, 0].reset()
            # self._env.initInterference()

            self.curr_step = 0
            self.last_action = 0
            self.archived_action = 0
            self.archived_paction = 0
            self.deciding_CH = 0

            self.full_reward = 0.0
            self.accum_reward = 0.0
            self.accum_steps = 0.0
            self.track_reward = False
            self.total_average = 0

            self.curr_state = [[0, 0, 0, 0] for _ in range(self.num_ch + 1)]
            self.archived_state = [[0, 0, 0, 0] for _ in range(self.num_ch + 1)]
            self.rewards = [0.0, 0.0, 0.0]
            self.accum_rewards = [0.0, 0.0, 0.0]
            self.archived_rewards = [0.0, 0.0, 0.0]

            self.full_reward2 = 0.0
            self.accum_reward_p = 0.0
            self.accum_steps_p = 0.0
            self.track_reward_p = False

            self.curr_pstate = [0, 0, 0, 0]
            self.archived_pstate = [0, 0, 0, 0]
            self.accum_rewardsp = [0.0, 0.0, 0.0]
            self.archived_rewardsp = [0.0, 0.0, 0.0]

            self.curr_reward = 0
            self.reward2 = 0

            self.curr_info = {
                "Last_Action": None,
                "Reward_Change": 0.0,          # -> Change in reward at step
                "Avg_Age": 0.0,               # -> avgAoI
                "Peak_Age": 0.0,              # -> peakAoI
                "Data_Distribution": 0.0,     # -> Distribution of Data
                "Total_Data_Change": 0.0,      # -> Change in Total Data
                "Total_Data": 0.0,            # -> Total Data
                "Crashed": False, 
                "Truncated": False
            }

            self.truncated = False
            self.terminated = False
            self._curr_total_data = 0
            return self._env
        return None

    def step(self, model, model_p):
        train_model = False
        train_CH = 0
        old_action = 0
        comms, move, harvest = 0, 0, 0
        self.total_average = 0
        last_CH = None
        curr_CH = None

        train_p = False
        old_paction = 0
        old_pstate = [0, 0, 0, 0]
        self.reward2 = 0
        bad_target = 0

        old_state = [[0, 0, 0, 0] for _ in range(self.num_ch + 1)]

        if not self.terminated:
            self.truncated = False
            if self.curr_step < self.max_num_steps:
                x = self.curr_step/60 + 2
                alpha = abs(104 - 65 * x + 47 * pow(x, 2) - 12 * pow(x, 3) + pow(x, 4))

                for sens in range(self._num_sensors):
                    self._env.sensorTable.iat[sens, 0].harvest_energy(alpha, self._env, self.curr_step)
                    self._env.sensorTable.iat[sens, 0].harvest_data(self.curr_step)

                for CH in range(self.num_ch):
                    self._env.CHTable.iat[CH, 0].harvest_energy(alpha, self._env, self.curr_step)
                    self._env.CHTable.iat[CH, 0].ch_download(self.curr_step)

                for uav in range(self._num_uav):
                    uav = self._env.UAVTable.iat[uav, 0]
                    bad_target = uav.bad_target
                    train_model, DCH, used_model, train_p, state, action, action_p, p_state, comms, move, harvest = \
                        (uav.set_dest(model, model_p, self.curr_step))
                    uav.navigate_step(self._env)
                    self.uavX = uav.indX
                    self.uavY = uav.indY

                    train_model2, change_archives = uav.receive_data(self.curr_step)
                    excess_energy = uav.receive_energy()

                    self.reward(excess_energy, bad_target)

                    # train_model = True
                    if train_model or train_model2:
                        train_model = True
                        train_CH = self.deciding_CH
                        self.full_reward = self.accum_reward / max(self.accum_steps, 1)
                        self.archived_rewards = np.array([(1 - bad_target) * self.rewards[0],
                                                          (1 - bad_target) * self.rewards[1],
                                                          (1 - bad_target) * self.rewards[2]])

                    self.curr_state = uav.state
                    self.last_action = action
                    self.terminated = uav.crash
                    self.curr_pstate = p_state

                    old_state = self.archived_state
                    old_action = self.archived_action
                    old_pstate = self.archived_pstate
                    old_paction = self.archived_paction

                    if used_model:
                        self.archived_state = state
                        self.archived_action = action
                        self.deciding_CH = DCH
                        self.track_reward = True
                        self.accum_rewards = self.rewards

                    """Train Dual Model"""
                    if train_p:
                        if self.curr_step < 15:
                            train_p = False
                        else:
                            self.full_reward2 = np.array(self.accum_reward_p / max(self.accum_steps_p, 1))
                            self.archived_rewardsp = np.array([self.rewards[0],
                                                               self.rewards[1],
                                                               self.rewards[2]])
                            if self.terminated:
                                self.archived_rewardsp = [-1, -1, -1]
                        self.archived_pstate = p_state
                        self.archived_paction = action_p
                        self.track_reward_p = True
                        self.accum_rewardsp = self.rewards

                if self.track_reward:
                    self.accum_steps += 1
                    self.accum_reward += self.curr_reward

                if self.track_reward_p:
                    if self.terminated:
                        self.accum_reward_p = 0
                    else:
                        self.accum_steps_p += 1
                        self.accum_reward_p += self.reward2

                self.curr_step += 1

            else:
                self.truncated = True
                self.curr_info = {
                    "Last_Action": self.last_action,
                    "Reward_Change": 0,        # -> Change in reward at step
                    "Avg_Age": 0,                   # -> avgAoI
                    "Peak_Age": 0,                 # -> peakAoI
                    "Data_Distribution": 0,     # -> Distribution of Data
                    "Total_Data_Change": 0,      # -> Change in Total Data
                    "Total_Data": self._curr_total_data, # -> Total Data
                    "Crashed": self.terminated,         # -> True if UAV is crashed
                    "Truncated": self.truncated         # -> Max episode steps reached
                }
        else:
            self.curr_reward = 0
            self.curr_info = {
                "Last_Action": self.last_action,
                "Reward_Change": 0,        # -> Change in reward at step
                "Avg_Age": 0,                   # -> avgAoI
                "Peak_Age": 0,                 # -> peakAoI
                "Data_Distribution": 0,     # -> Distribution of Data
                "Total_Data_Change": 0,      # -> Change in Total Data
                "Total_Data": self._curr_total_data, # -> Total Data
                "Crashed": self.terminated,         # -> True if UAV is crashed
                "Truncated": self.truncated         # -> Max episode steps reached
            }

        return train_model, train_p, old_state, old_action, old_paction, old_pstate, comms, move, harvest

    def reward(self, excess_energy, bad_target):
        '''
        Distribution of Data
        Average AoI
        Peak AoI
        Total Data
        '''
        totalAge = 0
        peakAge = 0
        minAge = self.curr_step

        for index in range(len(self.curr_state) - 1):
            peak_age = self.curr_state[index + 1][2]
            average_age = self.curr_state[index + 1][3]

            totalAge += average_age
            if peak_age > peakAge:
                peakAge = peak_age
            if peak_age < minAge:
                minAge = peak_age
        avgAge = float(totalAge) / float(self.num_ch)
       
        dataChange = 0
        maxColl = 0.0
        minColl = 1.0
        index: int

        for index in range(len(self.curr_state)):
            if index > 0:
                val = self.curr_state[index][1] / (max(self._curr_total_data, 1))
                if val > maxColl:
                    maxColl = val
                if val < minColl:
                    minColl = val
            else:
                dataChange = max(0, self.curr_state[index][1] - self._curr_total_data)
                self._curr_total_data += abs(round(dataChange))
        
        distOffset = maxColl - minColl

        rewardDist = 1 - distOffset

        rewardPeak = (1-peakAge/self._aoi_threshold) if (1-peakAge/self._aoi_threshold) > 0 else 0

        rewardAvgAge = 1 - avgAge/max(self.curr_step, 1)


        rewardDataChange = dataChange / 1_498_500

        rewardChange = (1 - bad_target) * (0.7 * rewardPeak + 0.3 * rewardAvgAge)

        reward_energy = max(excess_energy, 0)
        reward2Change = 0.25 * rewardPeak + 0.25 * rewardAvgAge + 0 * rewardDataChange + 0.5 * reward_energy

        if self.terminated:
            print("Died")
            rewardChange = 0
        
        self.curr_info = {
            "Last_Action": max(self.last_action, -1),
            "Reward_Change": rewardChange,        # -> Change in reward at step
            "Avg_Age": avgAge,                   # -> avgAoI
            "Peak_Age": peakAge,                 # -> peakAoI
            "Data_Distribution": distOffset,     # -> Distribution of Data
            "Total_Data_Change": dataChange,      # -> Change in Total Data
            "Total_Data": self._curr_total_data, # -> Total Data
            "Crashed": self.terminated,         # -> True if UAV is crashed
            "Truncated": self.truncated         # -> Max episode steps reached
        }
        
        self.curr_reward = rewardChange
        self.reward2 = reward2Change

        self.rewards = [max(rewardPeak, 0), max(rewardAvgAge, 0), max(reward_energy, 0)]