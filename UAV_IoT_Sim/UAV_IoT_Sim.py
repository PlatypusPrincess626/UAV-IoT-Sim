# Dependecies to Import
import pandas as pd

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
        self._num_sensors = num_sensors
        self.num_ch = num_ch
        self._num_uav = num_uav
        self._max_steps = max_num_steps
        
        self.curr_step = 0
        self.curr_state = [[0, 0, 0] for _ in range(self.num_ch + 6)]
        self.last_action = 0

        self.archived_state = [[0, 0, 0] for _ in range(self.num_ch + 6)]
        self.archived_action = 0

        self.curr_reward = 0
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

        self._aoi_threshold = 60
        self.truncated = False
        self.terminated = False
        self._curr_total_data = 0
    
    def reset(self):
        if self.scene == "test":
            for sensor in range(self._num_sensors):
                self._env.sensorTable.iloc[sensor,0].reset()
            for CH in range(self.num_ch):
                self._env.CHTable.iloc[CH, 0].reset()
            for uav in range(self._num_uav):
                self._env.UAVTable.iloc[uav, 0].reset()
            self._env.initInterference()
            
            self.curr_step = 0
            self.curr_state = [[0, 0, 0] for _ in range(self.num_ch + 6)]
            self.last_action = 0

            self.archived_state = [[0, 0, 0] for _ in range(self.num_ch + 6)]
            self.archived_action = 0

            self.curr_reward = 0
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
    
    def step(self, model):
        train_model = False
        old_state = [[0, 0, 0] for _ in range(self.num_ch + 6)]
        old_action = 0
        comms, move, harvest = 0, 0, 0

        if not self.terminated:
            if self.curr_step < self._max_steps:
                x = self.curr_step/60 + 2
                alpha = 1.041834 - 0.6540587 * x + 0.4669073 * pow(x, 2) - 0.1225805 * pow(x, 3) + 0.0137882 * pow(x, 4) - 0.0005703625 * pow(x, 5)
            	
                for sens in range(self._num_sensors):
                    self._env.sensorTable.iloc[sens, 0].harvest_energy(alpha, self._env, self.curr_step)
                    self._env.sensorTable.iloc[sens, 0].harvest_data(self.curr_step)

                for CH in range(self.num_ch):
                    self._env.CHTable.iloc[CH, 0].harvest_energy(alpha, self._env, self.curr_step)
                    self._env.CHTable.iloc[CH, 0].ch_download(self.curr_step)

                for uav in range(self._num_uav):
                    uav = self._env.UAVTable.iloc[uav, 0]
                    train_model, used_model, state, action, comms, move, harvest = uav.set_dest(model, self.curr_step)
                    uav.navigate_step(self._env)
                    train_model, change_archives = uav.receive_data(self.curr_step)
                    uav.receive_energy()

                    self.curr_state = uav.state
                    self.last_action = action
                    self.terminated = uav.crash

                    old_state = self.archived_state
                    old_action = self.archived_action

                    if change_archives:
                        for Iter in range(5):
                            self.archived_state[len(uav.full_state) + Iter][1], \
                                self.archived_state[len(uav.full_state) + Iter][2] = 0, 0
                        self.archived_action = action

                    if used_model:
                        self.archived_state = state
                        self.archived_action = action

                self.reward()
                self.curr_step += 1
            else:
                self.truncated = True
                self.curr_info={
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

        return train_model, old_state, old_action, comms, move, harvest

    def reward(self):
        '''
        Distribution of Data
        Average AoI
        Peak AoI
        Total Data
        '''
        totalAge = 0
        peakAge = 0
        minAge = self._aoi_threshold
        for index in range(len(self.curr_state) - 6):
            age = self.curr_step - self.curr_state[index + 1][2]
            if age > self._aoi_threshold:
                age = self._aoi_threshold
            totalAge += age
            if age > peakAge:
                peakAge = age
            if age < minAge:
                minAge = age
        avgAge = totalAge/self.num_ch
       
        dataChange = 0
        maxColl = 0.0
        minColl = 1.0
        index: int
        for index in range(len(self.curr_state) - 5):
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
        
        rewardDist = 1 - 2 * distOffset
        rewardPeak = 1 - 2 * (peakAge / self._aoi_threshold)
        rewardAvgAge = 1 - 2 * (peakAge - avgAge)/self._aoi_threshold
        rewardDataChange = dataChange / (max(self._curr_total_data/(self.curr_step+1), 1))
        rewardChange = 0.5 * rewardDist + 2 * rewardPeak + 2 * rewardAvgAge + 0.1 * rewardDataChange

        if self.terminated:
            rewardChange -= 100
        
        self.curr_info = {
            "Last_Action": self.last_action,
            "Reward_Change": rewardChange,        # -> Change in reward at step
            "Avg_Age": avgAge,                   # -> avgAoI
            "Peak_Age": peakAge,                 # -> peakAoI
            "Data_Distribution": distOffset,     # -> Distribution of Data
            "Total_Data_Change": dataChange,      # -> Change in Total Data
            "Total_Data": self._curr_total_data, # -> Total Data
            "Crashed": self.terminated,         # -> True if UAV is crashed
            "Truncated": self.truncated         # -> Max episode steps reached
        }
        
        self.curr_reward += rewardChange
