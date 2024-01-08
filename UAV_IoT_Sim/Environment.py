# Import Dependencies
from pvlib import spectrum, solarposition, irradiance, atmosphere
import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
from scipy import signal
import math

# Custom Packages
from UAV_IoT_Sim import IoT_Device, UAV


class sim_env:
    def __init__(self, scene, num_sensors, num_uav, num_clusterheads, max_num_steps):
        
        self.total_sensors = num_sensors
        self.total_uav = num_uav
        self.total_clusterheads = num_clusterheads
        
        
        if(scene == "test"):
            # Test scene will be Upemba National Park in Kabanga, Congo
            self.lat_center = -9.00           # Latitude
            self.long_center = 26.75          # Longitude
            self.pressure = 101709            # Sea Level is 1013.25 mb, Average Pressure in Kabanga is +4.09mb
            self.water_vapor_content = 0.35   # Roughly 0.35 cm in Kabanga
            self.tau500 = 0.75                # Aerosol Turbidity 500nm
            self.ozone = 0.23                 # Ozone in atm-cm
            self.albedo = 0.2                 # Bare Ground and Grassy
            self.dim = 10000                  # Map dimension n x n
            self.numObst = 500                # Number of obstacles decided
            self.stepSize = 'T'               # Frequency of time steps
            self.times = pd.date_range('2021-01-01 8:00', freq=self.stepSize, periods=max_num_steps, tz="Africa/Lubumbashi")
            random.seed('2021-01-01 8:00')
            
            
        self.envMap = self.makeMap()    
    
    # Create Map and Grid
    def makeMap(self):
        
        if(self.dim%2==0):
            self.dim += 1
        
        stp = 0.000009   # 1 degree lat/long is ~111km
        minLat = self.long_center - self.dim/2 * stp
        maxLat = self.long_center + self.dim/2 * stp
        minLong = self.lat_center - self.dim/2 * stp
        maxLong = self.lat_center + self.dim/2 * stp
        
        envLats = np.arange(start = minLat, stop = maxLat, step = stp, dtype=float)
        envLongs = np.arange(start = minLong, stop = maxLong, step = stp, dtype=float)
        
        
        envMapTemp = []
        for y in envLats:
            for x in envLongs:
                envMapTemp.append([x, y])
        
        self.envMap = pd.DataFrame(envMapTemp)
        
        envObj= self.placeObjects()
        self.initInterference()
        
        return pd.concat([self.envMap, pd.DataFrame(envObj)], axis = 1)
    
    # Place obstructions and devices in initial positions
    def placeObjects(self) -> list:
        
        dims = self.dim
        envObj = [0] * (dims*dims)
        
        for obst in range(self.numObst):
            obstType = random.randint(-3,-1)
            while obstType < 0:
                place = random.randint(0, dims*dims-1)
                if envObj[place]==0:
                    envObj[place] = obstType
                    obstType = 0
        
        envSensors = [0] * (dims*dims)
        sensorList = []
        # Random Sensor Placement for now
        for sensor in range(self.total_sensors):
            place = random.randint(0, dims*dims-1)
            obstType = 1
            while obstType > 0:
                place = random.randint(0, dims*dims-1)
                if envObj[place]==0:
                    sensorList.append([IoT_Device.IoT_Device(int(place/dims), int(place%dims), obstType,\
                                                           self.envMap.iat[place,0], self.envMap.iat[place,1])])
                    envObj[place] = obstType
                    envSensors[place] = obstType
                    obstType = 0
        
        self.sensorTable = pd.DataFrame(sensorList)
        
        np.reshape(envSensors,(dims, dims))
        sensCoord = []
        for sensor in sensorList:
            sensCoord.append([sensor[0].indX, sensor[0].indY])
        data = np.array(sensCoord, dtype = 'int')
        kmeans = KMeans(n_clusters=self.total_clusterheads, random_state=0, n_init=10).fit(data)
        centroids = kmeans.cluster_centers_
        head_assignment = [sensCoord, kmeans.predict(sensCoord)]
        
        uavCHList = []
        clusterheadList = []
        countCH = 0
        for centroid in centroids:
            row = int(centroid[0])
            column = int(centroid[1])
            place = row * dims + column - 1
            obstType = 2
            while obstType > 0:
                if envObj[place]==0:
                    clusterheadList.append([IoT_Device.IoT_Device(int(place/dims), int(place%dims), obstType,\
                                                           self.envMap.iat[place,0], self.envMap.iat[place,1], countCH), []])
                    countCH += 1
                    envObj[place] = obstType
                    obstType = 0
                else:
                    if column < dims/2:
                        if row < dims/2:
                            place += 1
                        else:
                            place -= 1
 
        for location in head_assignment:
            X, Y = location[0], location[1]
            for sensor in sensorList:
                sensorX, sensorY = sensor[0].getIndicies()
                if sensorX == X and sensorY == Y:
                    clusterheadList[location[2]][1].append(sensor)
                    sensor.setHead(location[2], len(clusterheadList[location[2]][1]))
                    break  
                    
        for CH in clusterheadList:
            uavCHList.append([CH[0], len(CH[1])])
        
        for CH in clusterheadList:
            CH[0].set_sensor_data(CH[1])
            
        self.clusterTable = pd.DataFrame(clusterheadList)
        
        
        uavList = []
        count = 0
        for uav in range(self.total_uav):
            obstType = 3
            while obstType > 0:
                place = random.randint(0, dims*dims-1)
                if envObj[place]==0:
                    uavList.append([UAV.QuadUAV(int(place/dims), int(place%dims), self.envMap.iat[place,0],\
                                                        self.envMap.iat[place,1], count, uavCHList)])
                    envObj[place] = obstType
                    obstType = 0
        
        self.uavTable = pd.DataFrame(uavList)
        return envObj
    
    # Update Environment Variables
    def gaussian_kernel(self, n, std, normalised=False):
        '''
        Generates a n x n matrix with a centered gaussian 
        of standard deviation std centered on it. If normalised,
        its volume equals 1.
        '''
        gaussian1D = signal.gaussian(n, std)
        gaussian2D = np.outer(gaussian1D, gaussian1D)
        if normalised:
            gaussian2D /= (2*np.pi*(std**2))
        return gaussian2D
    
    def initInterference(self):
        dims = self.dim
        envStaticInter = [0.0] * (dims*dims)
        
        # Create Stoatic 
        shadows = int(math.sqrt(dims))
        for shadow in range(shadows):
            place = random.randint(0, dims*dims-1)
            shadeSize = random.randint(int(dims/25), int(2*dims/25))
            intensity = random.randint(int(0.25*shadeSize), int(0.75*shadeSize))
            data2D = self.gaussian_kernel(shadeSize, intensity, normalised = True)
            count=0
            for pixel in range(shadeSize*shadeSize):
                try:
                    envStaticInter[place] += data2D.flatten()[pixel]
                    if envStaticInter[place] > 1:
                        envStaticInter[place]=1
                except:
                    # In the case in which variable "place" would hit out-of-bounds
                    break
                place += 1
                count += 1
                if count%shadeSize==0:
                    place += (dims-shadeSize)
                    count = 0
        self.dataStaticInter = np.array(envStaticInter, dtype='float')
                    
    # Interactions with devices
    def moveUAV(self, X: int, Y: int, newX: int, newY: int): # Estimated Position of UAV (nearest meter)
        place = X * self.dims + Y
        newPlace = newX * self.dims + newY
        if self.envMap.iat[place, 2] == 3:
            self.envMap.loc[place, 2] = 0
        elif self.envMap.iat[place, 2] == 9:
            self.envMap.loc[place, 2] = 1
        else:
            self.envMap.loc[place, 2] /= 3
            
        if self.envMap.iat[newPlace, 2] == 0:
            self.envMap.loc[newPlace, 2] = 3
        elif self.envMap.iat[newPlace, 2] == 1:
            self.envMap.loc[newPlace, 2] = 9
        else:
            self.envMap.loc[newPlace, 2] *= 3
    
    def getIrradiance(self, lat, long, tilt, azimuth, step):
        solpos = solarposition.get_solarposition(self.time[step], lat, long)
        aoi = irradiance.aoi(tilt, azimuth, solpos.apparent_zenith, solpos.azimuth)
        relative_airmass = atmosphere.get_relative_airmass(solpos.apparent_zenith, model='kasten1966')
        spectra = spectrum.spectr12(
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
    
    def getInterference(self, X, Y):
        return self.dataStaticInter[X*self.dim+Y] + 0.001 * np.random.random()
        
        