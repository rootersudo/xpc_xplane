import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



class PID:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0

    def update(self, measured_value):
        error = self.setpoint - measured_value
        self.integral += error
        derivative = error - self.previous_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        return output


# Создание модели с той же архитектурой, что и обученная модель
model = Sequential()
model.add(Dense(256, input_dim=6, activation='tanh'))
model.add(Dense(256, activation='tanh'))
model.add(Dense(3))  # Выходной слой с количеством нейронов, равным количеству углов отклонения рулей и тяги

# Загрузка весов модели
model.load_weights('model_weights.h5')

import xpc
import time

# Создание PID-регуляторов для управления рулями
pid_aileron = PID(Kp=0.001, Ki=0.001, Kd=0.005)
pid_elevator = PID(Kp=0.001, Ki=0.001, Kd=0.005)
pid_thrust = PID(Kp=0.01, Ki=0.00001, Kd=0.005)
# Управление самолетом через XPCS
# Подключение к X-Plane
client = xpc.XPlaneConnect()

# Функция для получения данных из X-Plane


# Функция для получения данных из X-Plane
def get_xplane_data():
    ctr = client.getCTRL()
    pos = client.getPOSI()
    roll_angle = pos[4]
    pitch_angle = pos[3]
    thrust = ctr[3]
    heading_deviation = pos[5]
    altitude_deviation = pos[2]*3.2
    speed = client.getDREF("sim/flightmodel/position/indicated_airspeed")[0]

    return [roll_angle, pitch_angle, thrust, heading_deviation, altitude_deviation, speed]

# Функция для управления самолетом
def control_airplane(data):
    # Предсказание углов отклонения рулей и тяги
    predictions = model.predict(np.array([data]))

    # Извлечение предсказанных значений
    aileron_deflection, elevator_deflection, thrust = predictions[0]
    # Применение PID-регуляторов
    aileron_deflection += pid_aileron.update(data[0])  # Крен
    elevator_deflection += pid_elevator.update(data[1])  # Тангаж
    rudder_deflection = pid_thrust.update(data[2])  # Рыскание

    client.sendCTRL([aileron_deflection,elevator_deflection, 0, thrust])

# Основной цикл управления
while True:
    # Получение данных из X-Plane
    xplane_data = get_xplane_data()

    # Управление самолетом
    control_airplane(xplane_data)

    # Задержка для обновления данных
    time.sleep(0.1)