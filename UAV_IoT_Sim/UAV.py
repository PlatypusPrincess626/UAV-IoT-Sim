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
        self.LoRaIdle = 0.016       # 1.6 microA idle LoRa
        self.LoRaTrans = 38.9         # 38.9 mA transmit LoRa
        self.LoRaRec = 14.22          # 14.22 mA receive LoRa
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
        self.storedBatt = self.cap      # Initialize at full battery
        
        # State used for model
        self.state = [[0]*3 for _ in range(len(self.full_state))]
        self.state[0][0], self.state[0][1], self.state[0][2] = -1, 0, self.cap
        count = 0
        for row in range(len(self.state)-1):
            self.state[row+1][0] = count
            count += 1

    def reset(self):
        self.state = [[0]*3 for _ in range(len(self.full_state))]
        self.state[0][0], self.state[0][1], self.state[0][2] = -1, 0, self.cap
        self.full_state[0, 2] = 0
        self.full_state[0, 3] = 0
        count = 0
        self.storedBatt = self.cap
        self.crash = False
        for row in range(len(self.state)-1):
            self.state[row+1][0] = count
            self.full_state[row, 2] = 0
            self.full_state[row, 3] = 0
            count += 1
 
    # Internal UAV Mechanics
    def set_target(self, X:int, Y:int):
        self.targetX = float(X)
        self.targetY = float(Y)
        
    def navigate_step(self, env: object):
        maxDist = math.sqrt(pow(self.indX - self.targetX, 2) + pow(self.indY - self.targetY, 2))
        if self.storedBatt >=((self.cap/self.flight)/(60)):
            if maxDist <= self.maxSpd * 60:
                timeLeft = 60*(1 - maxDist/self.maxSpd)
                env.moveUAV(round(self.indX), round(self.indY), round(self.targetX), round(self.targetY))
                self.indX = self.targetX
                self.indY = self.targetY
                self.energy_cost(60, 0, 0, 0, 0)
            else:
                vectAngle = math.atan((self.targetY - self.indY)/(self.targetX - self.indX)) # Returns radians
                direction = (self.targetX - self.indX)/abs(self.targetX - self.indX)
                env.moveUAV(round(self.indX), round(self.indY), math.floor(self.indX + direction * self.maxSpd * 60 * math.cos(vectAngle)), math.floor(self.indY + direction * self.maxSpd * 60 * math.sin(vectAngle)))
                self.indX += direction * self.maxSpd * 60 * math.cos(vectAngle)
                self.indY += direction * self.maxSpd * 60 * math.sin(vectAngle)
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
        
        self.storedBatt -= totalCost
        self.state[0][2] = self.storedBatt
        self.full_state.iloc[0,2] = self.storedBatt
        
    # Finish with battery drain
    # UAV-IoT Communication
    def recieve_data(self, step):
        commsDone = 3
        totalData = 0
        totalTime = 0.0
        device = self.target
        if self.storedBatt < (self.cap*.40):
            self.energy_cost(0, 0, 0, 0, 60)

        elif self.target.type == 1:
            dataReturn = device.ws_upload_data(self.indX, self.indY, self.maxAmBCDist, self, self.h)
            if dataReturn < (self.transSpdAmBC * 56) and dataReturn >= 0:

                if commsDone == 2:
                    totalData += dataReturn
                elif math.sqrt(pow((self.indX - self.target.indX),2) + pow((self.indY - self.target.indY),2) \
                     + pow(self.h,2)) < self.maxAmBCDistCharge:
                    totalData += dataReturn
                else:
                    totalData += dataReturn
                    self.inRange = False
            else:
                totalData += dataReturn
                self.inRange = True

            totalTime = totalData/self.transSpdAmBC
            self.energy_cost(0, totalTime, 0, 0, 60)


        else:
            dataReturn = device.ch_upload(self.indX, self.indY, self.h)
            if dataReturn < (self.transSpdLoRa * 56) and dataReturn >= 0:
                if math.sqrt(pow((self.indX - self.target.indX),2) + pow((self.indY - self.target.indY),2) \
                     + pow(self.h,2)) < self.LoRaDistmin:
                    totalData += dataReturn
                else:
                    self.inRange = False
            else:
                totalData += dataReturn
                self.inRange = True

            totalTime = totalData/self.transSpdLoRa
            self.energy_cost(0, 0, 0, totalTime, 60-totalTime)
            self.update_state(device.headSerial+1, step, totalData)
            self.state[device.headSerial+1][2] = step
            self.state[device.headSerial+1][1] += round(totalData/1000)
            self.state[0][1] += round(totalData/1000)
     
    def recieve_energy(self):
        if self.target.type == 2:
            h, tC, tD = self.target.chargeTime(self.indX, self.indY, self.h, self.maxClimb)
            self.h = h
            self.energy_cost(tD, 0, 0, 0, 0)
            self.storedBatt += tC * (self.cap/(self.rate*60*60))
            self.target.discharge(tC * (self.cap/(self.rate*60*60)))
            self.state[0][2] = self.storedBatt
            self.full_state.iloc[0, 3] = self.storedBatt
    
    def set_dest(self, model):
        if self.target == None:
            minDist = 10000
            minCH = self.full_state.iloc[1, 0]
            for CH in range(len(self.full_state)-1):
                dist = math.sqrt(pow((self.indX-self.full_state.iloc[CH+1, 0].indX), 2)\
                                 + pow((self.indY-self.full_state.iloc[CH+1, 0].indY), 2))
                if dist < minDist:
                    minDist = dist
                    minCH = self.full_state.iloc[CH+1, 0]
            self.target = minCH
            self.targetX = minCH.indX
            self.targetY = minCH.indY

        elif self.storedBatt < (self.cap * .60):
            self.target = self.target

        else:
            dest = self.target.getDest(self.state, self.full_state, model)
            self.target = dest
            self.targetX = dest.indX
            self.targetY = dest.indY

        try:
            return self.target.headSerial
        except:
            return None

    def update_state(self, device, step, data):
        self.full_state.iloc[device, 3] =  step
        self.full_state.iloc[device, 2] += data
        self.full_state.iloc[0, 2] += data
