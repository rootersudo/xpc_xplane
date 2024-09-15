from time import sleep
import xpc as xpc
def monitor():
    with xpc.XPlaneConnect() as client:
        while True:
            posi = client.getPOSI()
            ctrl = client.getCTRL()

            print("Aileron:%2f Elevator:%2f Rudder:%2f HZCHE:%2f\n"\
               % (ctrl[1], ctrl[0], ctrl[2], ctrl[3]))


while True:
    monitor()