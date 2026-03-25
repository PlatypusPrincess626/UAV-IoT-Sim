from pvlib import spectrum, solarposition, irradiance, atmosphere
import pandas as pd
from moving_cells_utils import devices_ugv as ugv
import random
import numpy as np
from scipy.signal import windows
import os
import datetime
import csv
import atexit


def gaussian_kernel(n, std, normalised=False):
    """
    Generates a n x n matrix with a centered gaussian
    of standard deviation std centered on it. If normalised,
    its volume equals 1.
    """
    gaussian_1D = windows.gaussian(n, std)
    gaussian_2D = np.outer(gaussian_1D, gaussian_1D)
    if normalised:
        gaussian_2D /= (2 * np.pi * (std ** 2))
    return gaussian_2D


class SingleUGVEnv:
    def __init__(self, max_steps: int=720, max_dim: int=800, chkpt_div: int=15):
        """
        Need environment variables required by PVLib set up obfuscation table
        """
        self.dim = max_dim
        self.step_size, self.max_num_steps, self.chkpt_div = 'min', max_steps, chkpt_div
        self.times = pd.date_range('2021-01-01 8:00', freq=self.step_size, periods=self.max_num_steps, tz="MST")

        self.latitude, self.longitude, self.res = 44.424, -110.589, 0.000009
        self.pressure, self.wvc, self.tau500, self.ozone, self.albedo = 101253, 0.35, 0.75, 0.23, 0.2

        checkpoints = int(720 / self.chkpt_div)
        shadow_array = [[0.0] * (1 + 2 * self.dim) * (1 + 2 * self.dim)] * checkpoints
        for checkpoint in range(checkpoints):
            print("Making Happy Trees:", checkpoint)
            shadows = self.init_interference()
            shadow_array[checkpoint] = shadows
        self.obfuscation_array = np.array(shadow_array)

        # Set directory path
        log_dir = "moving_cells_utils/logs"
        os.makedirs(log_dir, exist_ok=True)
        csv_str = ".csv"
        date_time = datetime.datetime.now()
        # Shade logging
        shade_log = ("solver_obfuscation_" + date_time.strftime("%d") + "_" +
                     date_time.strftime("%m") + csv_str)
        shade_logfile = os.path.join(log_dir, shade_log)
        # open once, append mode; newline='' avoids blank lines on Windows
        self.shade_file = open(shade_logfile, mode='a', newline='', encoding='utf-8')
        self.shade_writer = csv.writer(self.shade_file, delimiter='|')
        # write header only if file is empty
        if os.path.getsize(shade_logfile) == 0:
            self.shade_writer.writerow(["shade"])
            self.shade_file.flush()
        for timepoint in self.obfuscation_array:
            self.shade_writer.writerow("Checkpoint")
            self.shade_file.flush()
            self.shade_writer.writerows(timepoint)
            self.shade_file.flush()
        try:
            self.shade_file.close()
        except Exception:
            pass
        atexit.register(lambda: self.shade_file and not self.shade_file.closed and self.shade_file.close())

        self.ugv = ugv.DeviceUGV(self, random.sample(self.dim, k=1)[0])

        self.current_step = 0


    def get_constraints(self):
        return self.max_num_steps


    def init_interference(self):
        env_static_interference = [[0.0] * (1 + 2 * self.dim) * (1 + 2 * self.dim)]
        shadows = int(self.dim)
        for shadow in range(shadows):
            place = random.randint(0, self.dim * self.dim - 1)
            size = random.randint(int(8), int(50))
            intensity = random.randint(int(0.25 * size), int(0.75 * size))
            data_2D = gaussian_kernel(size, intensity, normalised=False)
            count = 0
            for pixel in range(size * size):
                if place < len(env_static_interference[0]):
                    env_static_interference[0][place] += data_2D.flatten()[pixel]
                    if env_static_interference[0][place] > 1:
                        env_static_interference[0][place] = 1
                else:
                    # In the case in which variable "place" would hit out-of-bounds
                    break
                place += 1
                count += 1
                if count % size == 0:
                    place += (self.dim - size)
                    count = 0
        return np.array(env_static_interference)


    def get_spectrum(self, x: int, y: int, time: int):
        """
        Return spectrum from PVLib
        """
        solpos = solarposition.spa_python(self.times[time], self.latitude+x*self.res,
                                                 self.longitude+y*self.res, pressure=self.pressure)
        relative_airmass = atmosphere.get_relative_airmass(solpos.apparent_zenith, model='kasten1966')
        spectra = spectrum.spectrl2(
            apparent_zenith=solpos.apparent_zenith,
            aoi=0,
            surface_tilt=solpos.apparent_zenith,
            ground_albedo=self.albedo,
            surface_pressure=self.pressure,
            relative_airmass=relative_airmass,
            precipitable_water=self.wvc,
            ozone=self.ozone,
            aerosol_turbidity_500nm=self.tau500,
        )
        return spectra


    def get_obfuscation(self, x: int, y: int, step: int):
        """
        Return obfuscation from table
        """
        return self.obfuscation_array[int(step / self.chkpt_div), int(y * self.dim + x)]


    def step(self, step: int):
        """
        Move environment at step
        """
        return self.ugv.step(step)