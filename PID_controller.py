import sys
import scipy as sc
import xpc
import numpy as np


class PID:
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.error_sum = 0.0
        self.last_error = 0.0

    def update(self, desired_value, actual_value):
        error = desired_value - actual_value
        self.error_sum += error * self.dt
        delta_error = error - self.last_error
        self.last_error = error
        output = self.Kp * error + self.Ki * self.error_sum + self.Kd * delta_error / self.dt
        return output





def monitor():

    iter = 1
    x = np.zeros(1000000)
    y = np.zeros(1000000)
    x1 = np.zeros(1000000)
    y1 = np.zeros(1000000)
    z= np.zeros(1000000)
    z1=np.zeros(1000000)
    z2=z1
    for i in range(len(z)):
        z[i]=0.0
        z1[i]=-0.0
    print(len(z))
    Rctrl = np.zeros(6)
    with xpc.XPlaneConnect() as client:
        posi = client.getPOSI()

        posx = 55.5916
        posy = 37.2616

        pos = [posx, posy, 500, 0, 0, 0, 1]
        client.sendPOSI(pos)


        while True:
            if iter-10<0:
                num=1
            else:
                num=10

            posi = client.getPOSI()
            ctrl = client.getCTRL()
            if iter>len(x):
                break
            else:
                None

            if abs(y[iter])>180:
                y[iter]*=-1

            elif(y[iter]<0):
                y[iter]*=-1

            for i in range(3,6):
                Rctrl[i]=ctrl[i]
            Rctrl[3]=0.7

            yaw=PID(0.01,0.001,0.01,1/10)
            roll=PID(0.01,0.001,0.01,1/10)
            pitch=PID(0.01,0.0001,0.01,1/10)

            Rctrl[2] = yaw.update(0, posi[5])
            Rctrl[1]=roll.update(0.0,posi[4])
            Rctrl[0]=pitch.update(0,posi[3])

            client.sendCTRL(Rctrl)

            print("Loc: (%4f, %4f, %4f) Aileron:%2f Elevator:%2f Rudder:%2f\n"\
               % (posi[3], posi[4], posi[2], ctrl[1], ctrl[0], ctrl[2]))

if __name__ == "__main__":
    monitor()