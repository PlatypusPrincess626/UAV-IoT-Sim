from pvlib import spectrum, solarposition, irradiance, atmosphere
import pandas as pd
import devices_ugv as ugv
import random
import numpy as np
from scipy.signal import windows


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
    def __init__(self, max_steps: int=720, max_dim: int=800, chkpt_div: int=5):
        """
        Need environment variables required by PVLib set up obfuscation table
        """
        self.dim = max_dim
        self.step_size, self.max_num_steps, self.chkpt_div = 'min', max_steps, chkpt_div
        self.times = pd.date_range('2021-01-01 8:00', freq=self.step_size, periods=self.max_num_steps, tz="MST")

        self.latitude, self.longitude, self.res = 44.424, -110.589, 0.000009
        self.pressure, self.wvc, self.tau500, self.ozone, self.albedo = 101253, 0.35, 0.75, 0.23, 0.2

        checkpoints = int(720 / self.chkpt_div)
        shadow_array = []
        for checkpoint in range(checkpoints):
            shadows = self.init_interference()
            shadow_array.append(shadows)
        self.obfuscation_array = np.array(shadow_array)
        self.ugv = ugv.DeviceUGV(self, random.sample(range(int(self.dim*9/10)), k=1))

        self.current_step = 0


    def get_constraints(self):
        return self.max_num_steps


    def init_interference(self):
        env_static_interference = [0.0] * (self.dim * self.dim)
        shadows = int(self.dim)
        print("Making Happy Trees")
        for shadow in range(shadows):
            place = random.randint(0, self.dim * self.dim - 1)
            size = random.randint(int(8), int(50))
            intensity = random.randint(int(0.25 * size), int(0.75 * size))
            data_2D = gaussian_kernel(size, intensity, normalised=False)
            count = 0
            for pixel in range(size * size):
                if place < len(env_static_interference):
                    env_static_interference[place] += data_2D.flatten()[pixel]
                    if env_static_interference[place] > 1:
                        env_static_interference[place] = 1
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
        solpos = solarposition.get_solarposition(self.times[time], self.latitude+x*self.res,
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