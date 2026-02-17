"""
File: environment.py
Author: Mason Conkel
Creation Date: 2026-02-10
Description: This script establishes the environment
"""
# Import Dependencies
from pvlib import spectrum, solarposition, irradiance, atmosphere
import pandas as pd
from numpy.typing import NDArray
import numpy as np
import random
from sklearn.cluster import KMeans
from scipy import signal
from scipy.signal import windows
import math
import datetime

# Custom Packages
from sim_mods import devices_iot, devices_uav


def gaussian_kernel(n, std, normalised=False):
    '''
    Generates a n x n matrix with a centered gaussian
    of standard deviation std centered on it. If normalised,
    its volume equals 1.
    '''
    gaussian1D = windows.gaussian(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    if normalised:
        gaussian2D /= (2 * np.pi * (std ** 2))
    return gaussian2D


def dist(pt1: NDArray[np.int32], pt2: NDArray[np.int32]):
    assert pt1.shape == (2,)
    assert pt2.shape == (2,)
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


class sim_env:
    def __init__(self, scene, num_sensors, num_uav, num_ch, max_num_steps):

        self.uav_table = None
        self.ch_table = None
        self.sensor_table = None
        self.total_sensors = num_sensors
        self.total_uav = num_uav
        self.total_clusterheads = num_ch
        self.max_num_steps = max_num_steps

        self.chX = [0] * self.total_clusterheads
        self.chY = [0] * self.total_clusterheads
        self.sensor_pts = []
        self.ch_pts = []

        if (scene == "test"):
            # Test scene will be Yellowstone National Park
            self.lat_center = 44.424  # Latitude
            self.long_center = -110.589  # Longitude
            self.stp = 0.000009  # 1 degree lat/long is ~111km
            self.pressure = 101253  # Sea Level is 1013.25 mb, Average Pressure in Yellowstone is +4.09mb
            self.water_vapor_content = 0.35  # Roughly 0.35 cm in Yellowstone
            self.tau500 = 0.75  # Aerosol Turbidity 500nm
            self.ozone = 0.23  # Ozone in atm-cm
            self.albedo = 0.2  # Bare Ground and Grassy
            self.dim = 5_000  # Map dimension n x n
            self.numObst = 500  # Number of obstacles decided
            self.stepSize = 'min'  # Frequency of time steps
            self.times = pd.date_range('2021-01-01 8:00', freq=self.stepSize, periods=self.max_num_steps, tz="MST")
            random.seed('2021-01-01 8:00')

        # insert interference creation loop
        checkpoints = int(720 / 30)
        shadow_array = []
        for checkpoint in range(checkpoints):
            shadows = self.init_interference()
            shadow_array.append(shadows)
            # if checkpoint == 0:
            #     self.obfuscation_array = np.expand_dims(shadows, axis=0)
            # else:
            #     np.append(self.obfuscation_array, np.expand_dims(shadows, axis=0), axis=0)

        self.obfuscation_array = np.array(shadow_array)

        self.envMap = self.makeMap()

    # Create Map and Grid
    def makeMap(self):

        if self.dim % 2 == 0:
            self.dim += 1

        flg_done = self.place_devices()

        return flg_done

    def init_interference(self):

        envStaticInter = [0.0] * (self.dim * self.dim)

        shadows = int(self.dim)
        print("Making Happy Trees")
        for shadow in range(shadows):

            place = random.randint(0, self.dim * self.dim - 1)
            shadeSize = random.randint(int(8), int(50))
            intensity = random.randint(int(0.25 * shadeSize), int(0.75 * shadeSize))
            data2D = gaussian_kernel(shadeSize, intensity, normalised=False)
            count = 0

            for pixel in range(shadeSize * shadeSize):
                try:
                    envStaticInter[place] += data2D.flatten()[pixel]
                    if envStaticInter[place] > 1:
                        envStaticInter[place] = 1
                except:
                    # In the case in which variable "place" would hit out-of-bounds
                    break
                place += 1
                count += 1
                if count % shadeSize == 0:
                    place += (self.dim - shadeSize)
                    count = 0

        return np.array(envStaticInter)

    def move_dest(self, x: int, y: int, timestep: int):
        checkpoint = int(timestep / 720)
        alpha = 0.5
        O_best = 10
        x_best, y_best = [0, 0]
        for x_1 in range(2):
            for y_1 in range(2):
                O1 = self.obfuscation_array[checkpoint, (y + y_1 - 1) * self.dim + (x + x_1 - 1)]
                O2 = 0
                for x_2 in range(2):
                    for y_2 in range(2):
                        O2 += self.obfuscation_array[
                            checkpoint + 1, (y + y_1 + y_2 - 2) * self.dim + (x + x_1 + x_2 - 2)]
                O2 = O2 / 9
                if (alpha * O1 + (1 - alpha) * O2) < O_best:
                    O_best = alpha * O1 + (1 - alpha) * O2
                    x_best, y_best = [(x + x_1 - 1), (y + y_1 - 1)]

        return x_best, y_best

    # Place obstructions and devices in initial positions
    def place_devices(self) -> list:
        """
        self.dim -> dimension of environment
        self.total_sensors -> number of sensors in the environment
        max_dist_ambc = 800
        """
        max_dist_ambc = 800
        sensor_pts = np.array([[0, 0]] * self.total_sensors, np.int32)
        sensor_list = []

        print("Placing Sensors")
        for sensor in range(self.total_sensors):
            position = random.randint(0, self.dim * self.dim - 1)

            sensor_pts[sensor] = [int(position % self.dim), int(position / self.dim)]
            sensor_list.append([devices_iot.Sensor(int(position % self.dim),
                                                   int(position / self.dim),
                                                   self.lat_center + self.stp * (int(position % self.dim) - self.dim),
                                                   self.long_center + self.stp * (int(position / self.dim) - self.dim),
                                                   random.randint(1, 3))])

        K = 1
        while True:
            err_dist = False
            k_means = KMeans(n_clusters=K, random_state=0, n_init=10).fit(sensor_pts)
            for pt in range(self.total_sensors):
                cluster = k_means.cluster_centers_[k_means.predict(np.expand_dims(sensor_pts[pt], axis=0)).item()]
                if abs(dist(sensor_pts[pt], cluster)) > max_dist_ambc:
                    err_dist = True
            if not err_dist:
                K += 1
            else:
                break

        centroids = k_means.cluster_centers_
        labels = k_means.predict(sensor_pts)
        self.total_clusterheads = K

        ch_list = []
        ch_count = 0
        for index, cluster in enumerate(centroids):
            pt_max_dist = 0
            num_sensors = 0
            ch_sensor_list = []
            for pt_index, true_pt in enumerate(sensor_pts):
                if labels[pt_index] == index:
                    num_sensors += 1
                    ch_sensor_list.append(sensor_list[pt_index])
                    pt_dist = dist(true_pt, cluster)
                    if pt_dist > pt_max_dist:
                        pt_max_dist = pt_dist

            r_move = max_dist_ambc - pt_max_dist
            ch_long = self.lat_center + self.stp * (cluster[0] - self.dim)
            ch_lat = self.long_center + self.stp * (cluster[1] - self.dim)
            ch_list.append(devices_iot.ClusterHead(self,
                                                   cluster[0],
                                                   cluster[1],
                                                   ch_long,
                                                   ch_lat,
                                                   ch_sensor_list,
                                                   r_move,
                                                   ch_count))
            ch_count += 1

        uav_list = []
        count = 0
        for uav in range(self.total_uav):
            place = random.randint(0, self.dim * self.dim + self.dim - 1)
            uav_list.append([devices_uav.QuadUAV(int(place % self.dim),
                                                 int(place / self.dim),
                                                 self.lat_center + self.stp * (int(place % self.dim) - self.dim),
                                                 self.long_center + self.stp * (int(place / self.dim) - self.dim),
                                                 count,
                                                 ch_list)])
            count += 1

        self.sensor_pts = sensor_pts
        self.ch_pts = centroids

        self.sensor_table = pd.DataFrame(sensor_list)
        self.sensor_table.rename(
            columns={0: "Sensor"},
            inplace=True
        )
        self.ch_table = pd.DataFrame(ch_list)
        self.ch_table.rename(
            columns={0: "CH"},
            inplace=True
        )
        self.uav_table = pd.DataFrame(uav_list)
        self.uav_table.rename(
            columns={0: "UAV"},
            inplace=True
        )

        return True

    def get_spectrum(self, lat, long, tilt, azimuth, step):
        solpos = solarposition.get_solarposition(self.times[step], lat, long)
        aoi = irradiance.aoi(tilt, azimuth, solpos.apparent_zenith, solpos.azimuth)
        relative_airmass = atmosphere.get_relative_airmass(solpos.apparent_zenith, model='kasten1966')
        spectra = spectrum.spectrl2(
            apparent_zenith=solpos.apparent_zenith,
            aoi=aoi,
            surface_tilt=tilt,
            ground_albedo=self.albedo,
            surface_pressure=self.pressure,
            relative_airmass=relative_airmass,
            precipitable_water=self.water_vapor_content,
            ozone=self.ozone,
            aerosol_turbidity_500nm=self.tau500,
        )

        return spectra

    def get_obfuscation(self, x: int, y: int, step):
        return self.obfuscation_array[int(step / 30), int(y * self.dim + x)]

