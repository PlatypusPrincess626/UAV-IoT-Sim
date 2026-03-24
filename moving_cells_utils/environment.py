from pvlib import spectrum, solarposition, irradiance, atmosphere
import pandas as pd
import devices_ugv as ugv
import random


class single_ugv_environment:
    def __init__(self):
        """
        Need environment variables required by PVLib set up obfuscation table
        """
        self.step_size = 'min'
        self.times = pd.date_range('2021-01-01 8:00', freq=self.step_size, periods=self.max_num_steps, tz="MST")

        self.latitude, self.longitude, self.res = 44.424, -110.589, 0.000009
        self.pressure, self.wvc, self.tau500, self.ozone, self.albedo = 101253, 0.35, 0.75, 0.23, 0.2

        self.ugv = ugv.device_ugv(self, random.sample(range(700), k=1))

        self.current_step = 0


    def make_map(self):
        """
        Make environment map
        """
        pass


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


    def get_obfuscation(self):
        """
        Return obfuscation from table
        """
        pass


    def step(self):
        e_harvest, e_move, e_overhead =