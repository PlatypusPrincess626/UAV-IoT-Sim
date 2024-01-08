# Dependencies to Import
import math
import pandas as pd

class QuadUAV:
    def __init__(self, X:int, Y:int, long:float, lat:float, uavNum:int, CHList: list):
        self.type = 3
        self.serial = uavNum
        self.typeStr = "UAV"
        
        # AmBC Comms
        self.maxAmBCDist = 2000         # 2 km AmBC comms distance
        self.maxAmBCDistCharge = 20     # 10 m distance for AmBC charging
        self.commCost = 0.00007         # 70 microAmp current necessary
        self.AmBCBER = 0.001            # Bit Error Rate
        self.transSpdAmBC = 1000 * (1 - self.AmBCBER)
        
        # LoRa Comms
        self.LoRaDistmin = 5000         # 5 km conservative LoRa comms distance
        self.LoRaIdle = 0.0000016       # 1.6 microA idle LoRa
        self.LoRaTrans = 0.0389         # 38.9 mA transmit LoRa
        self.LoRaRec = 0.01422          # 14.22 mA receive LoRa
        self.LoRaBER = 0.001
        self.transSpdLoRa = 25000 * (1 - self.LoRaBER)
        
        # Positioning
        self.indX = float(X)
        self.indY = float(Y)
        self.maxH = 20
        self.h = 20
        self.target = None
        self.targetX = float(X)
        self.targetY = float(Y)
        self.lat = lat
        self.long = long
        self.maxSpd = 15                # 15 m/s max speed cap
        self.maxTurn = 200              # 200 degree per sec max yaw 
        self.maxClimb = 3               # 3 m/s ascent and descent
        self.inRange = True
        self.uav = [self, len(CHList), float(0.0), float(0.0)]
        self.full_state = pd.concat([pd.DataFrame(CHList), pd.DataFrame([[0, 0]]*(len(CHList)))], axis = 1)
        self.full_state.loc[-1] = self.uav
        self.full_state = self.full_state.sort_index()
        self.full_state = self.full_state.reset_index()
        self.full_state.drop('index', axis=1, inplace=True)
        self.full_state.rename(
            columns = {0:"Device", 1:"Num_Sensors", 2:"Total_Data", 3:"AoI"},
            inplace = True
        )
        self.crash = False
        
        # Battery Usage
        self.cap = 6800                 # 6800 mAh 
        self.rate = 2.5                 # 150 min charging time
        self.flight = 0.5               # 30 min flight time
        self.Amp = self.cap/self.rate   # Roughly 2.72 A optimal current
        self.storedBatt = self.cap/1000 # Initialize at full battery
        
        # State used for model
        self.state = [[0]*3 for _ in range(len(self.full_state))]
        self.state[0][0], self.state[0][1], self.state[0][2] = -1, 0, self.cap
        count = 0
        for row in range(len(self.state)-1):
            self.state[row+1][0] = count
            count += 1
        
    # Internal UAV Mechanics
    def set_target(self, X:int, Y:int):
        self.targetX = float(X)
        self.targetY = float(Y)
        
    def navigate_step(self, env: object):
        maxDist = math.sqrt(pow(self.indX - self.targetX, 2) + pow(self.indY - self.targetY, 2))
        if self.storedBatt >= 60 * ((self.cap/self.flight)/(60*60)):
            if maxDist <= self.maxSpd:
                timeLeft = 60*(1 - maxDist/self.maxSpd)
                self.indX = self.targetX
                self.indY = self.targetY
                self.energy_cost(60, 0, 0, 0, 0)
            else:
                vectAngle = math.atan((self.targetY - self.indY)/(self.targetX - self.indX)) # Returns radians
                env.moveUAV(math.round(self.indX), math.round(self.indY), \
                            math.round(self.maxSpd * math.cos(vectAngle)), math.round(self.maxSpd * math.sin(vectAngle)))
                self.indX = self.maxSpd * math.cos(vectAngle)
                self.indY = self.maxSpd * math.sin(vectAngle)
                self.energy_cost(60, 0, 0, 0, 0)
        else:
            self.crash = True
    
    def energy_cost(self, timeAir:float=0, timeAmBC:float=0, timeLoRaTrans:float=0,\
                    timeLoRaRec:float=0, timeLoRaIdle:float=0):
        totalCost = 0
        # Cost of air travel
        totalCost += timeAir * ((self.cap/self.flight)/(60*60))
        # Cost of AmBC
        totalCost += timeAmBC * self.commCost
        # Cost of LoRa
        totalCost += timeLoRaTrans * self.LoRaTrans
        totalCost += timeLoRaRec * self.LoRaRec
        totalCost += timeLoRaIdle * self.LoRaIdle
        
        self.storedBatt -= self.storedBatt
        
    # Finish with battery drain
    # UAV-IoT Communication
    def recieve_data(self, step):
        commsDone = 3
        totalData = 0
        totalTime = 0.0
        if self.target.type == 1:
            while commsDone > 0:
                dataReturn = self.target.ws_upload_data(self.indX, self.indY, self.h, self.maxAmBCDist)
                if dataReturn < (self.transSpdAmBC * 56):
                    
                    if commsDone == 2:
                        totalData += dataReturn
                        commsDone = 0
                    elif math.sqrt(pow((self.indX - self.target.indX),2) + pow((self.indY - self.target.indY),2) \
                         + pow(self.h,2)) < self.maxAmBCDistCharge:
                        totalData += dataReturn
                        commsDone = 0
                    else:
                        totalData += dataReturn
                        commsDone = 0
                        self.inRange = False
                else:
                    totalData += dataReturn
                    self.inRange = True
                    commsDone = 2
                    
            totalTime = totalData/self.transSpdSmBC
            self.energy_cost(0, totalTime, 0, 0, 60)
            
            
        else:
            dataReturn = self.target.ch_upload(self.indX, self.indY, self.h)
            while commsDone > 0:
                if dataReturn < (self.transSpdLoRa * 56):
                    if math.sqrt(pow((self.indX - self.target.indX),2) + pow((self.indY - self.target.indY),2) \
                         + pow(self.h,2)) < self.LoRaDistmin:
                        totalData += dataReturn
                        commsDone = 0
                    else:
                        totalData += dataReturn
                        commsDone = 0
                        self.inRange = False
                else:
                    totalData += dataReturn
                    self.inRange = True
                    commsDone = 2
            totalTime = totalData/self.LoRaRec
            self.energy_cost(0, 0, 0, totalTime, 60-totalTime)
            self.update_state(self.target.headSerial+1, step, totalData)
    
    def recieve_energy(self):
        if self.target.type == 2:
            h, tC, tD = self.target.chargeTime(self.indX, self.indY, self.h, self.maxClimb)
            self.h = h
            self.energy_cost(tD, 0, 0, 0, 0)
            self.storedBatt += tD * (self.cap/(self.rate*60*60))
            self.target.discharge(tD * (self.cap/(self.rate*60*60)))
    
    def set_dest(self, model):
        if self.target == None:
            minDist = 10000
            minCH = self.full_state.loc[0, "CH"]
            for CH in range(len(self.full_state)):
                dist = math.sqrt(pow((self.indX+self.full_state.loc[CH, "CH"].indX), 2)\
                                 + pow((self.indY+self.full_state.loc[CH, "CH"].indY), 2))
                if dist < minDist:
                    minDist = dist
                    minCH = self.full_state[CH,"CH"]
            self.target = minCH
            self.targetX = minCH.indX
            self.targetY = minCH.indY
        elif self.targetX == self.indX and self.targetY == self.indY:
            dest = self.target.getDest(self.state, self.full_state, model)
            self.target = dest
            self.targetX = dest.indX
            self.targetY = dest.indY
        else:
            # No change to destination
            self.target = self.target
            
        try:
            return self.target.headSerial
        except:
            return None
    
    def update_state(self, device, step, data):
        self.full_state.loc[device, "AoI"] =  step
        self.full_state.loc[device, "Total_Data"] += data
        self.full_state.loc[0, "Total_Data"] += data
        
        self.state[device][2] = step
        self.state[device][1] += round(data/1000)
        self.state[0][1] += round(data/1000)
        