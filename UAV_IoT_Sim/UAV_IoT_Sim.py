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
        self._num_ch = num_ch
        self._num_uav = num_uav
        self._max_steps = max_num_steps
        
        self._curr_step = 0
        self._curr_state = [0,0,0]*(self._num_ch+1)
        self._curr_reward = 0
        self._curr_info = {
            "Last_Action": 0,
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
        self._truncated = False
        self._terminated = False
        self._curr_total_data = 1
    
    def reset(self):
        if self.scene == "test":
            for sensor in range(self._num_sensors):
                self._env.sensorTable.iloc[sensor,0].reset()
            for CH in range(self._num_ch):
                self._env.CHTable.iloc[CH, 0].reset()
            for uav in range(self._num_uav):
                self._env.UAVTable.iloc[uav, 0].reset()
            
            self._curr_step = 0
            self._curr_state = [0,0,0]*(self._num_ch+1)
            self._curr_reward = 0
            self._curr_info = {
                "Last_Action": 0,
                "Reward_Change": 0.0,          # -> Change in reward at step
                "Avg_Age": 0.0,               # -> avgAoI
                "Peak_Age": 0.0,              # -> peakAoI
                "Data_Distribution": 0.0,     # -> Distribution of Data
                "Total_Data_Change": 0.0,      # -> Change in Total Data
                "Total_Data": 0.0,            # -> Total Data
                "Crashed": False, 
                "Truncated": False
            }
            self._truncated = False
            self._terminated = False
            self._curr_total_data = 1
            return self._env
    
    def step(self, model):
        if not self._terminated:
            if self._curr_step < self._max_steps:
                for sensor in range(self._num_sensors):
                    self._env.sensorTable.iloc[sensor, 0].harvest_energy(self._env, self._curr_step)
                    self._env.sensorTable.iloc[sensor, 0].harvest_data()

                for CH in range(self._num_ch):
                    self._env.CHTable.iloc[CH, 0].harvest_energy(self._env, self._curr_step)
                    self._env.CHTable.iloc[CH, 0].ch_download(self._curr_step)

                for uav in range(self._num_uav):
                    uav = self._env.UAVTable.iloc[uav, 0]
                    self.last_action = uav.set_dest(model)
                    uav.navigate_step(self._env)
                    uav.recieve_data(self._curr_step)
                    uav.recieve_energy()
                    self._curr_state = uav.state
                    self._terminated = uav.crash

                self.reward()
                self._curr_step += 1
            else:
                self._truncated = True
                self._curr_info={
                    "Last_Action": self.last_action,
                    "Reward_Change": 0,        # -> Change in reward at step
                    "Avg_Age": 0,                   # -> avgAoI
                    "Peak_Age": 0,                 # -> peakAoI
                    "Data_Distribution": 0,     # -> Distribution of Data
                    "Total_Data_Change": 0,      # -> Change in Total Data
                    "Total_Data": self._curr_total_data, # -> Total Data
                    "Crashed": self._terminated,         # -> True if UAV is crashed
                    "Truncated": self._truncated         # -> Max episode steps reached
                }
        else:
            self._curr_reward = 0
            self._curr_infor={
                "Last_Action": self.last_action,
                "Reward_Change": 0,        # -> Change in reward at step
                "Avg_Age": 0,                   # -> avgAoI
                "Peak_Age": 0,                 # -> peakAoI
                "Data_Distribution": 0,     # -> Distribution of Data
                "Total_Data_Change": 0,      # -> Change in Total Data
                "Total_Data": self._curr_total_data, # -> Total Data
                "Crashed": self._terminated,         # -> True if UAV is crashed
                "Truncated": self._truncated         # -> Max episode steps reached
            }

        return self._curr_state, self._curr_reward, self._terminated, self._truncated, self._curr_info
           
    def reward(self):
        '''
        Distribution of Data
        Average AoI
        Peak AoI
        Total Data
        '''
        totalAge = 0
        peakAge = 0
        minAge = self._max_steps
        for index in range(len(self._curr_state)-1):
            age = self._curr_step - self._curr_state[index+1][2]
            if age > self._aoi_threshold:
                age= self._aoi_threshold
            totalAge += age
            if age > peakAge:
                peakAge = age
            if age < minAge:
                minAge = age
        avgAge = totalAge/len(self._curr_state)
       
        dataChange = 0
        maxColl = 0.0
        minColl = 1.0
        for index in range(len(self._curr_state)):
            if index > 0:
                val = self._curr_state[index][1] / self._curr_total_data
                if val > maxColl:
                    maxColl = val
                if val < minColl:
                    minColl = val
            else:
                dataChange = self._curr_state[index][1] - self._curr_total_data
                self._curr_total_data += self._curr_state[index][1]
        
        distOffset = maxColl - minColl
        
        rewardDist = (1 - distOffset)
        rewardPeak = (1 - 2 * (self._aoi_threshold - peakAge)/self._aoi_threshold)
        rewardAvgAge = (1 - 2 * (peakAge - avgAge)/self._aoi_threshold)
        rewardDataChange = dataChange/10 # Maximum = 10
        rewardChange = 10*rewardDist + 10*rewardPeak + 10*rewardAvgAge + 4*rewardDataChange
        if self._terminated:
            rewardChange -= 100
        
        self._curr_info = {
            "Last_Action": self.last_action,
            "Reward_Change": rewardChange,        # -> Change in reward at step
            "Avg_Age": avgAge,                   # -> avgAoI
            "Peak_Age": peakAge,                 # -> peakAoI
            "Data_Distribution": distOffset,     # -> Distribution of Data
            "Total_Data_Change": dataChange,      # -> Change in Total Data
            "Total_Data": self._curr_total_data, # -> Total Data
            "Crashed": self._terminated,         # -> True if UAV is crashed
            "Truncated": self._truncated         # -> Max episode steps reached
        }
        
        self._curr_reward += rewardChange
