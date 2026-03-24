import setup_path
import airsim
import random
import numpy as np
import math
import time
import csv
import os
import datetime
from argparse import ArgumentParser
import atexit


import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv

class Drone():
    def __init__(self):
       self.speed = 20              #m/s
       self.position = [0, 0, 0]
       self.velocity = [0, 0, 0]
       self.prev_pos = [0, 0, 0]
       self.acceleration = 5        #m/s^2

    def get_position():
        pass

    def set_position(self, x, y, z):
       self.position = [x, y, z]

    def set_velocity(self, x, y, z):
       self.velocity = [x, y, z]

    def set_prev(self, x, y, z):
       self.prev_pos = [x, y, z]

    def move_to_point(self, x, y, z):
        pass 

    def move_by_velocity(self, x, y, z, t):
        self.set_velocity(x, y, z)
        self.prev_pos = self.position
        self.position = [
            self.prev_pos[0] + t * self.velocity[0],
            self.prev_pos[1] + t * self.velocity[1],
            self.prev_pos[2] + t * self.velocity[2]
            ]




class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape):
        self.mem = image_shape[0] - 1
        #shape_list = list(image_shape)
        #shape_list[0] += 1
        #self.image_shape = tuple(shape_list)
        self.image_shape = image_shape

        super().__init__(image_shape)
        #-------------------------------------------------------------------------------
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        csv_str = (".csv")
        date_time = datetime.datetime.now()

        # Individual file name
        stats_log = ("final_stats_" + date_time.strftime("%d") + "_" +
                        date_time.strftime("%m") + csv_str)
        stat_logfile = os.path.join(log_dir, stats_log)
        

        # open once, append mode; newline='' avoids blank lines on Windows
        self.stat_file = open(stat_logfile, mode='a', newline='', encoding='utf-8')
        self.stat_writer = csv.writer(self.stat_file, delimiter='|')

        # write header only if file is empty
        if os.path.getsize(stat_logfile) == 0:
            self.stat_writer.writerow(["ep", "reward", "avg_reward", "time_sec", "distance_m", "reward_dist", "P_used"])
            self.stat_file.flush()

        self.episode_idx = 0
        # --- end CSV setup ------------------------------------------------------------

        self.step_length = step_length
        self.time = 0
        self.step_size = 0.1 #seconds
    
        self.clock_speed = 10
        self.max_drone_speed = 10  #m/s
        self.max_aggro_speed = 5  #m/s
        self.target_dist = 20  #m
        self.time_limit = 200 #s
        self.length = 500 #m
        self.height = 100 #m
        self.dist_limit = 500 #m
        self.height_limit = -100 #m

        self.reward = 0
        self.avg_reward = 0
        self.dist = 0
        
        self.reward_dist = 0
        #self.reward_vert = 0
        self.reward_vel = 0
        self.reward_time = 0
        self.speed = 0
        self.height_penalty = 0
        
        self.location = 0
        self.quad_pt = 0
        

        #dist weight
        self.alpha = 0.90
        #vel weight
        self.gamma = 0.10
        
        self.energy_expended = 0
        self.path = 4
        
        self.drone = Drone()

        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
        }
        #self.obs = np.zeros(self.image_shape)
        self.obs = np.zeros((self.mem, 3, 1), dtype=np.float32)
        self.stored = 0
        self.drone_origin = self.drone.position
        self.action_space = spaces.Discrete(7)
        self._setup_flight()

    def __del__(self):

        self.drone.set_position(self.drone_origin[0], self.drone_origin[1], self.drone_origin[2])
        self.drone.set_velocity(0, 0, 0)
        self.drone.set_prev(self.drone_origin[0], self.drone_origin[1], self.drone_origin[2])
        
        # Close files
        try:
            self.stat_file.close()
        except Exception:
            pass
        atexit.register(lambda: self.stat_file and not self.stat_file.closed and self.stat_file.close())


    def _setup_flight(self):

        self.drone.set_position(self.drone_origin[0], self.drone_origin[1], self.drone_origin[2])
        self.drone.set_velocity(0, 0, 0)
        self.drone.set_prev(self.drone_origin[0], self.drone_origin[1], self.drone_origin[2])

        #self.path = random.randint(1, 7)
        if self.path == 1:
            print()
            print("square")
        elif self.path == 2:
            print()
            print("zig zag")
        elif self.path == 3:
            print()
            print("circle")
        elif self.path == 4:
            print()
            print("oval")
        elif self.path == 5:
            print()
            print("teardrop")
        elif self.path == 6:
            print()
            print("s offset circ")
        else:
            print()
            print("∞")

        self.obs = np.zeros(self.image_shape)
        self.stored = 0
        self.reward = 0
        self.dist = 0
        self.time = 0


    def _find_aggro(self):
        aggro_start = 0     
        pts1_square = [
            np.array([300,  -300, -40]),
            np.array([300,   300, -50]),
            np.array([-300,  300, -60]),
            np.array([-300, -300, -50])
        ]
        pts2_zig_zag = [
            np.array([300,  -300, -40]),
            np.array([200,   300, -50]),
            np.array([100,  -300, -60]),
            np.array([0,     300, -50]),
            np.array([-100, -300, -40]),
            np.array([-200,  300, -30]),
            np.array([-300, -300, -40])
        ]
        pts3_circle = [
            np.array([300.0,     0.0,  -30.0]),   #0
            np.array([259.8,   150.0,  -40.0]),  
            np.array([150.0,   259.8,  -50.0]),   
            np.array([75.0,    300.0,  -60.0]),  #90
            np.array([-150.0,  259.8,  -70.0]),  
            np.array([-259.8,  150.0,  -80.0]),         
            np.array([-300.0,    0.0,  -80.0]),   #180
            np.array([-259.8, -150.0,  -70.0]),  
            np.array([-150.0, -259.8,  -60.0]),  
            np.array([75.0,   -300.0,  -50.0]),   #270
            np.array([150.0,  -259.8,  -40.0]),
            np.array([259.8,  -150.0,  -30.0])
        ]
        pts4_oval = [
            np.array([300.0,  -300.0,  -40.0]),
            np.array([334.6,  -219.8,  -45.0]),
            np.array([318.2,  -106.1,  -50.0]),
            np.array([253.4,    23.8,  -55.0]),
            
            np.array([150.0,   150.0,  -55.0]),
            np.array([ 23.8,   253.4,  -50.0]),
            np.array([-106.1,  318.2,  -45.0]),
            np.array([-219.8,  334.6,  -40.0]),

            np.array([-300.0,  300.0,  -40.0]),            
            np.array([-334.6,  219.8,  -45.0]),
            np.array([-318.2,  106.1,  -50.0]),
            np.array([-253.4,  -23.8,  -55.0]),
            
            np.array([-150.0, -150.0,  -55.0]),           
            np.array([ -23.8, -253.4,  -50.0]),
            np.array([106.1,  -318.2,  -45.0]),
            np.array([219.8,  -334.6,  -40.0]),
        ]
        pts5_teardrop = [
            np.array([300.0,    0.0,   -40.0]), #tip
            np.array([150.0,   66.7,   -42.5]),   
            np.array([  0.0,  133.3,   -45.0]),   

            np.array([-103.3,  164.3,  -50.0]),
            np.array([-200.0,  133.3,  -60.0]),
            np.array([-250.0,   66.7,  -70.5]),

            np.array([-265.8,    0.0,  -72.5]), #crest
            np.array([-250.0,  -66.7,  -70.0]), 
            np.array([-200.0, -133.3,  -60.0]), 

            np.array([-150.0,  -200.0,  -50.0]),    
            np.array([   0.0,  -133.3,  -43.5]),    
            np.array([ 150.0,   -66.7,  -40.0])     
        ]
        pts6_SmallOffCirc = [ 
            np.array([   0.0,  300.0,  -40.0]),   #right
            np.array([-112.5,  269.9,  -50.0]),   
            np.array([-194.9,  187.5,  -65.0]),   
            np.array([-225.0,   75.0,  -80.0]),  #bottom
            np.array([-194.9,  -37.5,  -85.0]),   
            np.array([-112.5, -119.9,  -82.5]),   
            np.array([   0.0, -150.0,  -80.0]),   #left
            np.array([ 112.5, -119.9,  -70.0]),   
            np.array([ 194.9,  -37.5,  -55.0]), 
            np.array([ 225.0,   75.0,  -45.0]),  #top
            np.array([ 194.9,  187.5,  -42.5]),     
            np.array([ 112.5,  269.9,  -40.0])    
        ]
        pts7_figure_eight = [
            np.array([-150.0, -300.0, -40.0]),  #left
            
            np.array([ -43.9, -256.1, -42.5]),  
            np.array([   0.0, -150.0, -45.0]),  #left top
            np.array([ -43.9,  -43.9, -50.0]),  

            np.array([-150.0,    0.0, -65.0]),  #intersect
            
            np.array([-256.1,  43.91, -80.0]),  
            np.array([-300.0,  150.0, -85.0]),  #right bottom
            np.array([-256.1,  256.1, -85.0]),

            np.array([-150.0,  300.0, -80.0]),  #right
            
            np.array([ -43.9,  256.1, -65.0]), 
            np.array([   0.0,  150.0, -50.0]),  #right top
            np.array([ -43.9,   43.9, -35.0]),  

            np.array([-150.0,    0.0, -32.5]),  #intersect

            np.array([-256.1,  -43.9, -33.5]), 
            np.array([-300.0, -150.0, -39.0]),  #left bottom
            np.array([-256.1, -256.1, -40.0])
        ]

        if self.path == 1:
            pts = pts1_square    
        elif self.path == 2:
            pts = pts2_zig_zag
        elif self.path == 3:
            pts = pts3_circle           
        elif self.path == 4:
            pts = pts4_oval            
        elif self.path == 5:
            pts = pts5_teardrop
        elif self.path == 6:
            pts = pts6_SmallOffCirc
        else:
            pts = pts7_figure_eight
 

        # Keep track of which target Aggro is traveling to and when they started this "leg"
        # Calculate intermediate location using difference in system time multiplied by clock speed and "velocity"
        # End value should be an intermediate point with injected noise
        traveled = self.time * self.max_aggro_speed
        distance = np.linalg.norm(pts[aggro_start] - pts[aggro_start+1])

        while traveled > distance:
            traveled -= distance
            aggro_start += 1
            if aggro_start < len(pts) - 1:
                distance = np.linalg.norm(pts[aggro_start] - pts[aggro_start + 1])
            elif aggro_start == len(pts):
                aggro_start = 0
                distance = np.linalg.norm(pts[aggro_start] - pts[aggro_start + 1])
            else:
                distance = np.linalg.norm(pts[aggro_start] - pts[0])

        if aggro_start < len(pts) - 1:
            unit_vector = ((pts[aggro_start + 1] - pts[aggro_start]) /
                           np.linalg.norm(pts[aggro_start] - pts[aggro_start + 1]))
        else:
            unit_vector = ((pts[0] - pts[aggro_start]) /
                           np.linalg.norm(pts[aggro_start] - pts[0]))

        return pts[aggro_start] + traveled * unit_vector


    def _get_obs(self):
        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone.position
        self.state["velocity"] = self.drone.velocity
        collision = False
        self.state["collision"] = collision

            #here
        quad_pt = np.array(
            list(
                (
                    self.state["position"][0],
                    self.state["position"][1],
                    self.state["position"][2],
                )
            )
        )
        aggro_location = self._find_aggro()
        directional_vec = np.array([aggro_location[0] - quad_pt[0], 
                                  aggro_location[1] - quad_pt[1], 
                                  aggro_location[2] - quad_pt[2]])
        location = 255 * ((np.divide(directional_vec, np.array([self.length, self.length, self.height]))
                         + np.random.normal(0, 0.01, 3) + np.ones(3)) / 2)
        if self.stored < self.mem:
            self.obs[self.stored, 0, 0] = location[0].astype(np.uint8)
            self.obs[self.stored, 1, 0] = location[1].astype(np.uint8)
            self.obs[self.stored, 2, 0] = location[2].astype(np.uint8)
            self.stored += 1
        else:
            i = 0
            while i < (self.mem - 1):
                self.obs[i, 0, 0] = self.obs[i + 1, 0, 0]
                self.obs[i, 1, 0] = self.obs[i + 1, 1, 0]
                self.obs[i, 2, 0] = self.obs[i + 1, 2, 0]
                i += 1
            self.obs[self.mem - 1, 0, 0] = location[0].astype(np.uint8)
            self.obs[self.mem - 1, 1, 0] = location[1].astype(np.uint8)
            self.obs[self.mem - 1, 2, 0] = location[2].astype(np.uint8)
         
            #here
        quad_vel = self.drone.velocity
        self.obs[self.mem, 0, 0] = np.uint8( 255 * ((quad_vel[0] / self.max_drone_speed) + 1) / 2)
        self.obs[self.mem, 1, 0] = np.uint8( 255 * ((quad_vel[1] / self.max_drone_speed) + 1) / 2)
        self.obs[self.mem, 2, 0] = np.uint8( 255 * ((quad_vel[2] / self.max_drone_speed) + 1) / 2)

        return self.obs

    def _do_action(self, action):
        quad_offset = self.interpret_action(action)
        quad_vel = self.drone.velocity 
            #here
        vel_vec = np.array([quad_vel[0] + quad_offset[0],
                            quad_vel[1] + quad_offset[1],
                            0 + 0])
        vel = np.linalg.norm(vel_vec)
        
        unit_vel = vel_vec / max(1e-5, np.linalg.norm(vel_vec))

        command_duration = self.step_size #every action commands velocity for step_size seconds
        if vel < self.max_drone_speed:
            self.drone.move_by_velocity(
                vel_vec[0],
                vel_vec[1],
                vel_vec[2],
                command_duration,
            )
        else:
            self.drone.move_by_velocity(
                self.max_drone_speed * unit_vel[0],
                self.max_drone_speed * unit_vel[1],
                self.max_drone_speed * unit_vel[2],
                command_duration,
            )

    def _compute_reward(self):
        quad_pt = np.array(
            list(
                (
                    self.state["position"][0],
                    self.state["position"][1],
                    self.state["position"][2],
                )
            )
        )
        
        location = self._find_aggro()


        # Calculate reward for distance
        scale = max(self.length, self.height, self.dist_limit)
        dist  = math.sqrt((location[0] - quad_pt[0]) ** 2 + (location[1] - quad_pt[1]) ** 2)
        reward_dist = 1.0 - min(dist / scale, 1.0)

        # Calculate reward for velocity
        # 1 <= vel_max, decays to 0 when > vel_max
        
            #here
        quad_vel = self.drone.velocity
        speed = np.linalg.norm(np.array([quad_vel[0], quad_vel[1]], dtype=float))
        vel_max = float(self.max_drone_speed)
        reward_vel = speed / vel_max

        z_axis = quad_pt[2]
        
        if z_axis < self.height_limit:
            self.height_penalty = 1
        else:
            self.height_penalty = 0
        
        reward = (
            self.alpha * reward_dist + self.gamma * reward_vel - self.height_penalty
            #+ self.delta * reward_time + self.beta  * reward_vert
        )

        done = 0
        if dist <= self.target_dist:
            reward = self.alpha + self.gamma * reward_vel # + self.delta * reward_time
            print("#####CAUGHT#####CAUGHT#####CAUGHT#####CAUGHT#####CAUGHT#####CAUGHT#####CAUGHT#####CAUGHT#####CAUGHT#####CAUGHT#####CAUGHT")
            done = 1
        elif self.state["collision"] or self.time > self.time_limit or dist > self.dist_limit or speed > vel_max: #or quad_pt[2] < self.height_limit:
            reward = -1.0
            done = 1 
       
        #calculate energy used
        P_used = 131.1 / (self.time / 3600)
        self.energy_expended = P_used

        self.dist = dist
        #compute as local variable for printing
        self.reward_dist = float(reward_dist)   
        #self.reward_vert = float(reward_vert)
        self.reward_vel = reward_vel
        #self.reward_time = float(reward_time)
        self.location = location
        self.quad_pt = quad_pt
        self.speed = speed
        return reward, done

    def step(self, action):
 
        self.time += self.step_size
        self._do_action(action) #timestep =0.1020512580871582
        obs = self._get_obs()
        reward, done = self._compute_reward()
        self.reward = reward
        self.avg_reward = (self.avg_reward + reward) / 2
        return obs, reward, done, self.state

    def reset(self):
        # Save final reward, time, and distance
        run_stats = [self.episode_idx, self.reward, self.avg_reward, self.time, self.dist, self.reward_dist, self.energy_expended, self.location, self.quad_pt]
        self.stat_writer.writerow(run_stats)
        self.stat_file.flush()
 

        print(f"A_Loc{self.location}")
        print(f"D_Loc{self.quad_pt}")
        print(f"time={self.time} dist={self.dist} speed={self.speed} P_used={self.energy_expended}")
        print(f"R={self.reward} avg_R={self.avg_reward} R_dist={self.reward_dist} R_vel={self.reward_vel}")

        self._setup_flight()
        return self._get_obs()
        
    def interpret_action(self, action):
        if action == 0:
            quad_offset = (self.step_length, 0, 0)
        elif action == 1:
            quad_offset = (0, self.step_length, 0)
        elif action == 2:
            quad_offset = (0, 0, -self.step_length)
        elif action == 3:
            quad_offset = (-self.step_length, 0, 0)
        elif action == 4:
            quad_offset = (0, -self.step_length, 0)
        elif action == 5:
            quad_offset = (0, 0, self.step_length)
        else:
            quad_offset = (0, 0, 0)

        return quad_offset
