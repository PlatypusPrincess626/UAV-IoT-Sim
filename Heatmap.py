import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from pvlib import spectrum, solarposition, irradiance, atmosphere
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import windows
import math

def main():
    lat_center = 44.424  # Latiitude
    long_center = -110.589  # Longitude
    dim = 100  # Map dimension n x n

    if (dim % 2 == 0):
        dim += 1

    stp = 0.000009  # 1 degree lat/long is ~111km
    minLat = long_center - dim / 2 * stp
    maxLat = long_center + dim / 2 * stp
    minLong = lat_center - dim / 2 * stp
    maxLong = lat_center + dim / 2 * stp

    envLats = np.arange(start=minLat, stop=maxLat, step=stp, dtype=float)
    envLongs = np.arange(start=minLong, stop=maxLong, step=stp, dtype=float)

    envMapTemp = []
    for y in envLats:
        for x in envLongs:
            envMapTemp.append([x, y])
    
    stepSize = 'min'  # Frequency of time steps
    hours = 8
    periods = 60 * hours # 60 minutes times number of hours
    times = pd.date_range('2021-01-01 8:00', freq=stepSize, periods=periods, tz="MST")
    #steps = [15, 120, 240, 465]
    steps = [465]
    
    #discounts = [0.75, 1, 0.93, 0.40]
    discounts = [0.40]
    for hour in range(1):
    	envInterference = init_interference(dim)
    	alpha = discounts[hour]
    	envPower, powMax = getPower(envMapTemp, envInterference, dim, steps[hour], times, alpha)
    	
    	finalPower = envPower.reshape(dim, dim)
    	# Add average pooling
    	po = poolingOverlap(finalPower, 9, stride=1, method='mean', pad=True, return_max_pos=False)
    	
    	sizes = np.shape(po)
    	fig = plt.figure()
    	fig.set_size_inches(1. * sizes[0] / sizes[1], 1, forward = False)
    	ax = plt.Axes(fig, [0., 0., 1., 1.])
    	ax.set_axis_off()
    	fig.add_axes(ax)
    	plt.imshow(po, cmap='turbo', interpolation='none', vmin = 0, vmax = powMax)
    	plt.savefig("output.png")

    return 0
    
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
    _ceil = lambda x, y: x//y + 1
    if pad:
        ny = _ceil(m, stride)
        nx = _ceil(n, stride)
        size = ((ny-1)*stride+f, (nx-1)*stride+f) + mat.shape[2:]
        mat_pad = np.full(size, 0)
        mat_pad[:m, :n, ...] = mat
    else:
        mat_pad = mat[:(m-f)//stride*stride+f, :(n-f)//stride*stride+f, ...]
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
    view_shape = (1+(m1-m2)//stride, 1+(n1-n2)//stride, m2, n2)+arr.shape[2:]
    strides = (stride*s0, stride*s1, s0, s1)+arr.strides[2:]
    subs = np.lib.stride_tricks.as_strided(
        arr, view_shape, strides=strides, writeable=False)
    return subs

def getPower(envMap, envStaticInter, dim, step, times, alpha):

    powMap = [0] * (dim*dim)

    for pix in range(len(powMap)):
        powMap[pix], vmax = getIrradiance(envMap[pix][0], envMap[pix][1], envStaticInter[pix], step, times, alpha)

    return np.array(powMap), vmax

def gaussian_kernel(n, std, normalised=False):
        '''
        Generates a n x n matrix with a centered gaussian 
        of standard deviation std centered on it. If normalised,
        its volume equals 1.
        '''
        gaussian1D = windows.gaussian(n, std)
        gaussian2D = np.outer(gaussian1D, gaussian1D)
        if normalised:
            gaussian2D /= (2*np.pi*(std**2))
        return gaussian2D

def init_interference(dim):

	envStaticInter = [0.0] * (dim*dim)
	
	shadows = int(math.sqrt(dim*dim))
	print("Making Happy Trees")
	for shadow in range(shadows):
	
            place = random.randint(0, dim*dim-1)
            shadeSize = random.randint(int(8), int(30))
            intensity = random.randint(int(0.75*shadeSize), int(1*shadeSize))
            data2D = gaussian_kernel(shadeSize, intensity, normalised = False)
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
                    place += (dim-shadeSize)
                    count = 0
	
	return envStaticInter

def getIrradiance(lat, long, interference, step, times, alpha, tilt = 45, azimuth = 180):
    pressure = 101253  # Sea Level is 1013.25 mb, Average Pressure in Yellowstone is +4.09mb
    water_vapor_content = 0.35  # Roughly 0.35 cm in Yellowstone
    tau500 = 0.75  # Aerosol Turbidity 500nm
    ozone = 0.23  # Ozone in atm-cm
    albedo = 0.2  # Bare Ground and Grassy
    
    random.seed('2021-01-01 12:00')

    solpos = solarposition.get_solarposition(times[step], lat, long)
    aoi = irradiance.aoi(tilt, azimuth, solpos.apparent_zenith, solpos.azimuth)
    relative_airmass = atmosphere.get_relative_airmass(solpos.apparent_zenith, model='kasten1966')
    spectra = spectrum.spectrl2(
        apparent_zenith=solpos.apparent_zenith,
        aoi=aoi,
        surface_tilt=tilt,
        ground_albedo=albedo,
        surface_pressure=pressure,
        relative_airmass=relative_airmass,
        precipitable_water=water_vapor_content,
        ozone=ozone,
        aerosol_turbidity_500nm=tau500,
    )
    f1 = InterpolatedUnivariateSpline(spectra['wavelength'], spectra['poa_global'])
    
    solpos = solarposition.get_solarposition(times[120], lat, long)
    aoi = irradiance.aoi(tilt, azimuth, solpos.apparent_zenith, solpos.azimuth)
    relative_airmass = atmosphere.get_relative_airmass(solpos.apparent_zenith, model='kasten1966')
    spectra = spectrum.spectrl2(
        apparent_zenith=solpos.apparent_zenith,
        aoi=aoi,
        surface_tilt=tilt,
        ground_albedo=albedo,
        surface_pressure=pressure,
        relative_airmass=relative_airmass,
        precipitable_water=water_vapor_content,
        ozone=ozone,
        aerosol_turbidity_500nm=tau500,
    )

    f2 = InterpolatedUnivariateSpline(spectra['wavelength'], spectra['poa_global'])
    
    powDensity = alpha * (1 - interference) * f1.integral(-math.inf, math.inf)

    return powDensity, f2.integral(-math.inf, math.inf)

if __name__=="__main__":
    main()

