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


def poolingOverlap(mat, f, stride=None, method='max', pad=False,
                   return_max_pos=False):
    '''Overlapping pooling on 2D or 3D data.
    Args:
        mat (ndarray): input array to do pooling on the first 2 dimensions.
        f (int): pooling kernel size.
    Keyword Args:
        stride (int or None): stride in row/column. If None, same as <f>,
            i.e. non-overlapping pooling.
        method (str): 'max for max-pooling,
                      'mean' for average-pooling.
        pad (bool): pad <mat> or not. If true, pad <mat> at the end in
               y-axis with (f-n%f) number of nans, if not evenly divisible,
               similar for the x-axis.
        return_max_pos (bool): whether to return an array recording the locations
            of the maxima if <method>=='max'. This could be used to back-propagate
            the errors in a network.
    Returns:
        result (ndarray): pooled array.
    See also unpooling().
    '''
    m, n = mat.shape[:2]
    if stride is None:
        stride = f
    _ceil = lambda x, y: x // y + 1
    if pad:
        ny = _ceil(m, stride)
        nx = _ceil(n, stride)
        size = ((ny - 1) * stride + f, (nx - 1) * stride + f) + mat.shape[2:]
        mat_pad = np.full(size, 0)
        mat_pad[:m, :n, ...] = mat
    else:
        mat_pad = mat[:(m - f) // stride * stride + f, :(n - f) // stride * stride + f, ...]
    view = asStride(mat_pad, (f, f), stride)
    if method == 'max':
        result = np.nanmax(view, axis=(2, 3), keepdims=return_max_pos)
    else:
        result = np.nanmean(view, axis=(2, 3), keepdims=return_max_pos)
    if return_max_pos:
        pos = np.where(result == view, 1, 0)
        result = np.squeeze(result)
        return result, pos
    else:
        return result


def asStride(arr, sub_shape, stride):
    '''Get a strided sub-matrices view of an ndarray.
    Args:
        arr (ndarray): input array of rank 2 or 3, with shape (m1, n1) or (m1, n1, c).
        sub_shape (tuple): window size: (m2, n2).
        stride (int): stride of windows in both y- and x- dimensions.
    Returns:
        subs (view): strided window view.
    See also skimage.util.shape.view_as_windows()
    '''
    s0, s1 = arr.strides[:2]
    m1, n1 = arr.shape[:2]
    m2, n2 = sub_shape[:2]
    view_shape = (1 + (m1 - m2) // stride, 1 + (n1 - n2) // stride, m2, n2) + arr.shape[2:]
    strides = (stride * s0, stride * s1, s0, s1) + arr.strides[2:]
    subs = np.lib.stride_tricks.as_strided(
        arr, view_shape, strides=strides, writeable=False)
    return subs

class sim_env:
    def __init__(self, scene, num_sensors, num_uav, num_clusterheads, max_num_steps):
        
        self.dataStaticInter = None
        self.total_sensors = num_sensors
        self.total_uav = num_uav
        self.total_clusterheads = num_clusterheads
        
        
        if(scene == "test"):
            # Test scene will be Yellowstone National Park
            self.lat_center = 44.424           # Latitude
            self.long_center = -110.589        # Longitude
            self.pressure = 101253            # Sea Level is 1013.25 mb, Average Pressure in Yellowstone is +4.09mb
            self.water_vapor_content = 0.35   # Roughly 0.35 cm in Yellowstone
            self.tau500 = 0.75                # Aerosol Turbidity 500nm
            self.ozone = 0.23                 # Ozone in atm-cm
            self.albedo = 0.2                 # Bare Ground and Grassy
            self.dim = 10000                  # Map dimension n x n
            self.numObst = 500                # Number of obstacles decided
            self.stepSize = 'min'               # Frequency of time steps
            self.times = pd.date_range('2021-01-01 8:00', freq=self.stepSize, periods=max_num_steps, tz="MST")
            random.seed('2021-01-01 8:00')
            
            
        self.envMap = self.makeMap()    
    
    # Create Map and Grid
    def makeMap(self):
        
        if self.dim % 2 == 0:
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
        
        envObj = self.placeObjects()
        self.initInterference()
        
        return pd.concat([self.envMap, pd.DataFrame(envObj)], axis = 1)
    
    # Place obstructions and devices in initial positions
    def placeObjects(self) -> list:
        
        dims = self.dim
        envObj = [0] * (dims*dims)

        print("Placing Obstuctions")
        for obst in range(self.numObst):
            obstType = random.randint(-3,-1)
            while obstType < 0:
                place = random.randint(0, dims*dims-1)
                if envObj[place] == 0:
                    envObj[place] = obstType
                    obstType = 0

        print("Placing Sensors")
        envSensors = [0] * (dims*dims)
        sensorList = []
        # Random Sensor Placement for now
        for sensor in range(self.total_sensors):
            obstType = 1
            while obstType > 0:
                place = random.randint(0, dims*dims-1)
                if envObj[place] == 0:
                    sensorList.append([IoT_Device.IoT_Device(int(place % dims), math.floor(place/dims), obstType,
                                                             self.envMap.iat[place, 0], self.envMap.iat[place, 1])])
                    envObj[place] = obstType
                    envSensors[place] = obstType
                    obstType = 0
        
        np.reshape(envSensors,(dims, dims))
        sensCoord = []
        for sensor in sensorList:
            X, Y = sensor[0].get_indicies()
            sensCoord.append([X, Y])
        data = np.array(sensCoord, dtype = 'int')
        kmeans = KMeans(n_clusters=self.total_clusterheads, random_state=0, n_init=10).fit(data)
        centroids = kmeans.cluster_centers_
        heads = kmeans.predict(sensCoord)
        head_assignment = [sensCoord[i] + [heads[i]-1] for i in range(len(sensCoord))]

        uavCHList = []
        clusterheadList = []
        countCH = 0
        for centroid in centroids:
            row = int(centroid[0])
            column = int(centroid[1])
            place = row * dims + column - 1
            obstType = 2
            while obstType > 0:
                if envObj[place] == 0:
                    clusterheadList.append([IoT_Device.IoT_Device(int(place % dims), math.floor(place/dims), obstType,
                                                                  self.envMap.iat[place, 0], self.envMap.iat[place, 1],
                                                                  countCH), []])
                    countCH += 1
                    envObj[place] = obstType
                    obstType = 0
                else:
                    if column < dims/2:
                        if row < dims/2:
                            place += 1
                        else:
                            place -= 1

        print("Placing Clusterheads") 
        for location in head_assignment:
            X, Y = location[0], location[1]
            for sensor in sensorList:
                sensorX, sensorY = sensor[0].get_indicies()
                if sensorX == X and sensorY == Y:
                    clusterheadList[location[2]][1].append(sensor)
                    sensor[0].set_head(location[2], len(clusterheadList[location[2]][1]))
                    break
                    
        for CH in clusterheadList:
            uavCHList.append([CH[0], len(CH[1])])
            CH[0].set_sensor_data(CH[1])
        
        uavList = []
        count = 0
        for uav in range(self.total_uav):
            obstType = 3
            while obstType > 0:
                place = random.randint(0, dims*dims-1)
                if envObj[place]==0:
                    uavList.append([UAV.QuadUAV(int(place%dims), math.floor(place/dims), self.envMap.iat[place,0],
                                                        self.envMap.iat[place,1], count, uavCHList)])
                    envObj[place] = obstType
                    obstType = 0
        
        self.sensorTable = pd.DataFrame(sensorList)
        self.sensorTable.rename(
             columns={0:"Sensor"},
             inplace=True
        )
        self.CHTable = pd.DataFrame(clusterheadList)
        self.CHTable.rename(
             columns={0:"CH",1:"Sensor_List"},
             inplace=True
        )
        self.UAVTable = pd.DataFrame(uavList)
        self.UAVTable.rename(
             columns={0:"UAV"},
             inplace=True
        )
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
        
        # Create Static 
        shadows = int(math.sqrt(dims*dims))
        print("Making Happy Trees")
        for shadow in range(shadows):
        
            place = random.randint(0, dims*dims-1)
            shadeSize = random.randint(int(8), int(30))
            intensity = random.randint(int(0.75*shadeSize), int(0.95*shadeSize))
            data2D = self.gaussian_kernel(shadeSize, intensity, normalised = False)
            count = 0
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
                if count % shadeSize == 0:
                    place += (dims-shadeSize)
                    count = 0


        temp = poolingOverlap(np.array(envStaticInter, dtype='float').reshape(dims, dims), \
                                              9, stride=1, method='mean', pad=True, return_max_pos=False)

        self.dataStaticInter = temp.flatten()

    # Interactions with devices
    def moveUAV(self, X: int, Y: int, newX: int, newY: int): # Estimated Position of UAV (nearest meter)
        place = Y * self.dim + X
        newPlace = newY * self.dim + newX
        if self.envMap.iloc[place, 2] == 3:
            self.envMap.iloc[place, 2] = 0
        elif self.envMap.iloc[place, 2] == 9:
            self.envMap.iloc[place, 2] = 1
        else:
            self.envMap.iloc[place, 2] /= 3
            
        if self.envMap.iloc[newPlace, 2] == 0:
            self.envMap.iloc[newPlace, 2] = 3
        elif self.envMap.iloc[newPlace, 2] == 1:
            self.envMap.iloc[newPlace, 2] = 9
        else:
            self.envMap.iloc[newPlace, 2] *= 3
    
    def getIrradiance(self, lat, long, tilt, azimuth, step):
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
    
    def getInterference(self, X, Y, type):
        if type == 1:
            return self.dataStaticInter[X*self.dim+Y]
        else:
            return 0
