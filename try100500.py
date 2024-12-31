import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import xpc
import numpy as np
import time

# Шаг 1: Прочитайте данные из файла
file_path = 'Data1.txt'
with open(file_path, 'r') as file:
    data = file.readlines()

# Шаг 2: Извлеките нужные параметры
headers = data[0].strip().split('|')
rows = [line.strip().split('|') for line in data[1:]]

# Создайте DataFrame из данных
df = pd.DataFrame(rows, columns=headers)


# Шаг 3: Подготовьте данные для обучения нейросети
for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')
#df['   __alt,ftmsl '][:1500]=1700
print(df['   __alt,ftmsl '])
# PID-регулятор
class PID:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0
    def changeKs(self,Kp,Ki,Kd):
        self.Kp=Kp
        self.Ki=Ki
        self.Kd=Kd
    def update(self, measured_value):
        error = self.setpoint - measured_value
        self.integral += error
        derivative = error - self.previous_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        return output
Kp1=0.5
Kp2=0.01
Kp3=0.005

# Пример использования PID-регулятора
pid_thrust = PID(0.3, 0.001, 0.5)
pid_roll = PID(0.5,0.00001,0.005)
pid_pitch = PID(Kp1,Kp2,Kp3)

# Пример функции для управления самолетом
def control_airplane(current_state, setpoint):
    speed, altitude, heading = current_state
    pid_thrust.setpoint = setpoint['speed']
    pid_roll.setpoint = setpoint['heading']
    pid_pitch.setpoint = setpoint['altitude']

    thrust = pid_thrust.update(speed)
    roll = pid_roll.update(heading)
    pitch = pid_pitch.update(altitude)

    return thrust, roll/100, pitch/200-0.09


# Подключение к X-Plane
with xpc.XPlaneConnect() as client:
    for index, row in df.iterrows():
        ctr = client.getCTRL()
        ctr1 = np.zeros(len(ctr))
        for i in range(len(ctr)):
            ctr1[i]=ctr[i]
        pos = client.getPOSI()
        # Получение текущих состояний самолета
        speed = client.getDREF("sim/flightmodel/position/indicated_airspeed")[0]

        altitude = pos[2]*3.2


        heading = pos[5]
        print(altitude - row['   __alt,ftmsl '])
        print()
        current_state = [speed, altitude, heading]
        setpoint = {'speed': row['   _Vind,__mph '], 'altitude': row['   __alt,ftmsl '], 'heading': 180}  # Заданные значения из файла

        # Управление с использованием нейросети
        thrust, roll, pitch = control_airplane(current_state, setpoint)
        if thrust<=0.3:
            thrust=0.3
        if thrust>=0.6:
            thrust=0.6
        if roll >= 0.5:
            roll = 0.5
        if roll <= -0.5:
            roll = -0.5
        if pitch <= -0.6:
            pitch = -0.6
        if pitch >= -0.3:
            pitch = -0.3
        print(f"PID Control: Thrust={thrust}, Roll={roll}, Pitch={pitch}")



        ctr1[0]=pitch
        ctr1[1]=roll
        ctr1[3]=thrust

        if altitude<1800:
            pid_pitch.changeKs(0.15,0.001,0.01)
            #ctr1[0] += 0.15
        if altitude<1600:
            pid_pitch.changeKs(0.25, 0.001, 0.01)
            if ctr1[5]<=0.33:
                ctr1[5]=0.33

        if altitude<1400:
            pid_pitch.changeKs(0.25, 0.001, 0.01)
            if ctr1[5]<=0.66:
                ctr1[5]=0.66

        if altitude < 900:
            pid_pitch.changeKs(0.35, 0.001, 0.005)
            ctr1[5] = 1
            #ctr1[0]-=0.14
        if altitude <500:
            ctr1[3]=0.3
        if altitude<450:
            #ctr1[0]+=0.25
            ctr1[3]=0
            ctr1[6]=-0.5
            # Установка значений управления в X-Plane
        client.sendCTRL(ctr1)

        # Задержка для обновления состояния
        time.sleep(0.05)