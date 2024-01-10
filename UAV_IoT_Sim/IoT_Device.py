# Import Dependencies
import random
import numpy
import math
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline

class IoT_Device:
    def __init__(self, X:int, Y:int, devType:int, long:float, lat:float, clusterheadNum:int=None):
        self.indX = X
        self.indY = Y
        self.lat = lat
        self.long = long
        
        # Communication Specifications
        self.LoRaDistmin = 5000         # 5 km conservative LoRa comms distance
        self.maxAmBCDist = 2000         # 2 km AmBC comms distance
        self.maxAmBCDistCharge = 20     # 10 m distance for AmBC charging
        self.commCost = 0.00007         # 70 microAmp current necessary
        self.AmBCBER = 0.001            # Bit Error Rate
        self.transSpdAmBC = 1000 * (1 - self.AmBCBER)
        
        if devType == 1:
            self.azimuth = 0                # 0 for North
            self.tilt = 0                   # 0 for no horizontal tilt
            self.type = 1        
            self.typeStr = "Wireless Sensor"
            self.storedData = random.randint(0, 250)
            
            # Sensor Specifications
            self.minVolt = 20               # 20 mV minimum voltage 
            self.maxColRate = 16            # 16 bits per sample
            self.sampleFreq = 2             # 15 minutes/ timesteps
            self.maxData = 250              # 250 kB maximum data storage
            self.sensRes = 5000             # 5k Ohm sensor resistance
            self.currDraw = self.minVolt/self.sensRes
            
            # BPW34 Solar Cell Specifications
            self.solarArea = 50 * 50        # 50 mm x mm solar panel assumption
            self.V_mp = 2                   # 2 V peak in ideal conditions (Vmp)
            self.spctrlLow = 0              # Spectral bandwidth for power calculations
            self.spctrlHigh = numpy.inf
            self.panelRes = 11.44           # Internal resistance of solar panels
            
            self._C = .470                   # 470 mF capacitance
            self._Q = (self._C * self.V_mp)           # C * V = Q
            self.capRes = .156              # Internal resistance of capacitor
            self.serRes = 0
            
            self.Ah = 1                     # 1 A*h battery rating
            self.rate = 10                  # 10 hours discharge/charge rate
            self.battAmp = self.Ah/self.rate
            self.storedEnergy=self.Ah 
            
        else:
            self.type = 2
            self.headSerial = clusterheadNum
            self.typeStr = "Clusterhead"
            self.azimuth = 180
            self.tilt = 45
            self.h = 0
            self.storedData = random.randint(0, 25000)
            
            # Clusterhead-specific communications
            self.numChannelsAmBC = 20
            self.LoRaIdle = 0.0000016       # 1.6 microA idle LoRa
            self.LoRaTrans = 0.0389         # 38.9 mA transmit LoRa
            self.LoRaRec = 0.01422          # 14.22 mA receive LoRa
            self.LoRaBER = 0.001
            self.transSpdLoRa = 25000 * (1 - self.LoRaBER)
            
            # BPW34 Solar Cell Specifications
            self.numPanels = 2
            self.solarArea = 50*50*self.numPanels  # Total surface area
            self.V_mp = 2*self.numPanels           # 2 V * number of panels in series (Vmp)
            self.spctrlLow = 0                     # Spectral bandwidth for power calculations
            self.spctrlHigh = numpy.inf
            self.panelRes = 11.44*self.numPanels   # Internal resistance of solar panels
            
            self._C = .470                         # 470 mF capacitance
            self._Q = (self._C * self.V_mp)                  # C * V = Q
            self.capRes = .156                     # Internal resistance of capacitor
            self.serRes = 0
            
            self.Ah = 2                            # 2 A*h battery rating
            self.rate = 10                         # 10 hours discharge/charge rate
            self.battAmp = self.Ah/self.rate
            self.storedEnergy=self.Ah 
            
    def reset(self):
        self.storedEnergy = self.Ah
        if self.type == 1:
            self.storedData = random.randint(0, 250)
        else:
            self.storedData = random.randint(0, 25000)

    # Call location
    def getIndicies(self):
        return self.indX, self.indY
    
    def getPlace(self):
        return self.long, self.lat
    
    def setHead(self, head: int, queue:int):
        self.head = head
        self.queue = queue
    
    # Harvesting
    def ws_charge(self,xtraCurr):
        if xtraCurr > self.battAmp:
            self.storedEnergy += self.battAmp/60
            if self.storedEnergy > self.Ah:
                self.storedEnergy = self.Ah
            
    def discharge(self, currCost):
        self.storedEnergy -= currCost
    
    def harvest_energy(self, env, step):
        spectra = env.getIrradiance(self.lat, self.long, self.tilt, self.azimuth, step)
        interference = env.getInterference(self.indX, self.indY)
        f = InterpolatedUnivariateSpline(spectra['wavelength'], spectra['poa_global'])
        powDensity = (1 - interference) * f.integral(self.spctrlLow,self.spctrlHigh)
        power = powDensity * self.solarArea/(1000*1000)
        
        if power>0:
            self.isActive = True
            currAmps = (power / (self.V_mp)) * 60
            if self.type == 1:
                if step % self.sampleFreq == 0:  # Data harvesting scheduling
                    self.harvest_data()
                    currAmps -= self.currDraw * 10
                if step % math.ceil(self.queue/2) == 0:    # Transmission Scheduling
                    self.tryComms()
                    currAmps -= self.commCost * 30
            if currAmps > 0:
                self.storedEnergy += currAmps
            else:
                self.discharge(currAmps)
            
                    
        elif self.storedEnergy > 0:
            self.isActive = True
            maxAmps = self.battAmp * 60
            currCost = 0
            if self.type == 1:
                if step % self.sampleFreq == 0:
                    if maxAmps > self.currDraw:
                        self.harvest_data()
                        currCost += self.currDraw * 10
                if step % math.ceil(self.queue/2):
                    if maxAmps > self.commCost:
                        self.upload_Data()
                        currCost += self.commCost * 30
            self.discharge(currCost)
                 
        else:
            self.isActive = False
                
       
    
    def harvest_data(self) -> float:
        if self.isActive:
            if self.storedData < self.maxData:
                self.storedData += self.maxColRate * 10 # Assume 10 samples taken every collection period
                if self.storedData > self.maxData:
                    self.storedData = self.maxData
        else:
            self.storedData += 0
    
    # Uploading data from a sensor
    def ws_upload_data(self, X:int, Y:int, commsDistance:int, device:object, h:int=0):
        if self.isActive:
            if self.storedData > 0:
                self.storedData -= self.transSpdAmBC * 28 * (device.type-1) # Assume 10 seconds to connect to clusterhead/uav
                sentData = self.transSpdAmBC * 28 * (device.type-1)
                if self.storedData < 0:
                    sentData += self.storedData 
                    self.storedData = 0
                return sentData
            
        elif math.sqrt(pow((self.indX - X),2) + pow((self.indY - Y),2) \
                         + pow(h,2)) <= commsDistance:
            if self.storedData > 0:
                self.storedData -= self.transSpdAmBC * 28 * (device.type-1)# Assume 10 seconds to connect to clusterhead/uav
                sentData = self.transSpdAmBC * 28 * (device.type-1)
                if self.storedData < 0:
                    sentData += self.storedData
                    self.storedData = 0
                return sentData
        
        else:
            return -1
    
    # Clusterhead-Specific Tasks
    def set_sensor_data(self, sensList: list):
        sensActive = [True] * (len(sensList))
        self.sensTable = pd.concat([pd.DataFrame(sensList), pd.DataFrame(sensActive)], axis=1)
        self.sensTable.rename(
            columns = {0:"Sensor", 1:"Connection_Status"},
            inplace = True
        )
    
    def ch_download(self, step):
        rotations = math.ceil(len(self.sensTable.index)/2)
        rotation = step % rotations
        sensor = rotation * 2
        recData = 0
        activeChannels = []
        sensor1 = self.sensTable.iloc[sensor, 0]
        activeChannels.append(sensor1.ws_upload_data(self.indX, self.indY, self.maxAmBCDist, self, self.h))
        if rotation < (rotations-1) or len(self.sensTable.index)%2 == 0:
            sensor2 = self.sensTable.iloc[sensor+1, 0]
            activeChannels.append(sensor2.ws_upload_data(self.indX, self.indY, self.maxAmBCDist, self, self.h))
        else:
            activeChannels.append(-2)

        totalChannels = 0
        for channel in range(len(activeChannels)):
            if activeChannels[channel] == -1:
                self.sensTable.iloc[sensor, 1] = False
                totalChannels += 1
            elif activeChannels[channel] >=  0:
                self.sensTable.iloc[sensor+1, 1] = True
                self.storedData += recData
                totalChannels += 1
            
        self.storedEnergy -= self.commCost * 30 * totalChannels
    
    def ch_upload(self, X:int, Y:int, h:int=0):
        if math.sqrt(pow((self.indX - X),2) + pow((self.indY - Y),2) \
                         + pow(h,2)) <= self.LoRaDistmin:
            if self.storedData > 0:
                self.storedData -= self.transSpdLoRa * 56
                sentData = self.transSpdLoRa * 56
                if self.storedData < 0:
                    sentData += self.storedData
                    self.storedData = 0

                self.storedEnergy -= self.LoRaTrans * (sentData/self.transSpdLoRa + 4)
                self.storedEnergy -= self.LoRaIdle * (1-(sentData/self.transSpdLoRa + 4))
                return sentData
            
            else:
                self.storedEnergy -= self.LoRaIdle * 60
                return 0
        
        else:
            self.storedEnergy -= self.LoRaIdle * 60
            return -1
    
    def chargeTime(self, X, Y, h, climb):
        if self.indX == X and self.indY == Y:
            if h > 0:
                timeDock = h/climb
                timeCharge = 60.0 - timeDock
                return 0, timeCharge, timeDock
            else:
                timeDock = 0
                timeCharge = 60.0
                return 0, timeCharge, timeDock
        else:
            return h, 0, 0
    
    def getDest(self, state, full_state, model):
        unserviced = full_state.iloc[:,3].isin([0])
        for CH in range(unserviced.size-1):
            if unserviced.iloc[CH+1, 3]:
                return full_state.iloc[CH+1, 0]
            
        # Insert code for independent sensors
        
        action = model.get(state)
        return full_state.iloc[action+1, 0]
