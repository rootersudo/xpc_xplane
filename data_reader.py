import math
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import xpc
import matplotlib.pyplot as plt


# Шаг 1: Прочитайте данные из файла
file_path = 'Data.txt'
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

pitch_c=PID(1,0.00000,0.00)
roll_c=PID(0.05,0.0001,0.5)
def neural(X,y,ctr):
    client.pauseSim(True)
    # Пример данных (замените на свои реальные данные)
    # X - входные параметры, y - углы отклонения рулей

    # Извлечение входных параметров (X) и выходных параметров (y) из DataFrame


    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Нормализация данных
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print()
    print((X_train))
    # Создание модели нейросети
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1))  # Выходной слой с количеством нейронов, равным количеству углов отклонения рулей

    # Компиляция модели
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Обучение модели
    model.fit(X_train, y_train, epochs=20, batch_size=1, validation_split=0.05)
    if ctr=='pitch':
        # Сохранение весов модели
        model.save_weights('pitch_weights.h5')
    else:
        # Сохранение весов модели
        model.save_weights('throttle_weights.h5')
    # Оценка модели
    loss = model.evaluate(X_test, y_test)
    print(f'Loss: {loss}')

    # Предсказание углов отклонения рулей на тестовых данных
    predictions = model.predict(X_test)
    predictions=predictions
    print(predictions)
    client.pauseSim(False)
    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Реальные данные')
    plt.plot(predictions, label='Предсказания модели')
    plt.xlabel('Индекс')
    plt.ylabel('Угол отклонения рулей')
    plt.title('Сравнение предсказаний и реальных данных')
    plt.legend()
    plt.show()
    return model, scaler

client = xpc.XPlaneConnect()
def get_xplane_data():

    ctr = client.getCTRL()
    pos = client.getPOSI()
    roll_angle = pos[4]
    pitch_angle = pos[3]
    thrust = ctr[3]
    heading_deviation = pos[5]
    altitude_deviation = pos[2]*3.2
    speed = client.getDREF("sim/flightmodel/position/indicated_airspeed")[0]

    return pos,ctr


row = df


def haversine(lat1, lon1, lat2, lon2):
    # Радиус Земли в километрах
    R = 6371.0

    # Преобразование широты и долготы из градусов в радианы
    lat = math.radians(lat1)
    lon = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Разница координат
    dlat = lat2 - lat
    dlon = lon2 - lon

    # Формула гаверсинуса
    a = math.sin(dlat / 2)**2 + math.cos(lat) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Расстояние
    distance =R * c

    return distance





def pitch():
    dt=1
    altitude=[]
    throttle=[]
    distance=[]
    p=[]
    pit=[]
    fl=[]
    gps=[]
    gps1=[]
    spd=[]

    for i in range(0,2290,1):
        j=i+10


        setpoint={ 'altitude': row[i:j]['   __alt,ftmsl '],'thrust': row[i:j]['     _throttle '], 'speed': row[i:j]["   _Vind,_kias "], 'pitch': row[i:j]['   pitch,__deg '],
                   'elevator':row[i:j]['   _elev,yoke1 '], 'flaps': row[i:j]['   _flap,postn '],"latitude_p": row[i:i+1]['   __lat,__deg '],
                   'longtitude_p':row[i:i+1]['   __lon,__deg '],'aileron':row[i:i+1]["   ailrn,yoke1 "]}
        lat1=setpoint['latitude_p'].mean()
        lon1=setpoint['longtitude_p'].mean()

        flaps=setpoint['flaps'].mean()
        alt2=setpoint['altitude'].mean()
        speed=setpoint['speed'].mean()
        pitch=setpoint['pitch'].mean()
        thrust=setpoint["thrust"]
        elevator=setpoint['elevator']
        #data += get_xplane_data()
        pit.append(pitch)
        #T = np.arcsin(speed.mean()*dt/(np.sqrt(alt ** 2 + (speed.mean() * dt) ** 2)))

        #deltaT=(pit-T)
        t=thrust.mean()
        throttle.append(t)
        spd.append(speed)
        altitude.append(alt2)
        distance.append(haversine(lat1, lon1,47.455136, -122.307724))
        p.append(elevator.mean())
        fl.append(flaps.mean())


    # Преобразование данных в числа, если это необходимо


    print(gps)

    print("distance_Uno = " + str(distance))
    altitude = [float(x) for x in altitude]
    throttle = [float(x) for x in throttle]
    p = [float(x) for x in p]
    fl=[float(x) for x in fl]
    pit = [float(x) for x in pit]

    # Убедитесь, что все списки имеют одинаковую длину

    X = np.array([altitude,fl,throttle,distance]).T
    y = np.array(p)
    X1=np.array([altitude,spd,pit,fl]).T
    y1=np.array(throttle)

    print(f"Форма X: {X.shape}")
    print(f"Форма y: {y.shape}")

    model, scaler = neural(X, y,"pitch")
    model1,scaler1=neural(X1,y1,"throttle")

    return  scaler,model,scaler1,model1



scaler,model,scaler1,model1 = pitch()




# Функция для управления самолетом в реальном времени
def control_airplane():
    # Загрузка обученной модели и нормализатора
    model.load_weights('pitch_weights.h5')
    model1.load_weights('throttle_weights.h5')
    print(111111111111)
    time.sleep(3)
    brakes=0
    while True:

        # Получение текущих данных из X-Plane
        pos,ctr = get_xplane_data()
        ctr1 = np.zeros(len(ctr))
        altitude = pos[2]*3.2
        speed = client.getDREF("sim/flightmodel/position/indicated_airspeed")[0]
        throttle = ctr[3]
        flaps= ctr[5]
        heading=pos[5]
        #roll=pos[3]
        pitch=pos[3]
        elevator_yoke = client.getDREF("sim/cockpit2/controls/elevator_trim")[0]
        aileron_yoke = client.getDREF("sim/cockpit2/controls/aileron_trim")[0]
        # Нормализация
        # данных
        distance = haversine(pos[0], pos[1],  47.46112, -122.307724)
        #print(distance)
        input_data = np.array([[altitude,flaps,throttle,distance]])
        input_data = scaler.transform(input_data)
        input_data1=np.array([[altitude,speed,pitch,flaps]])
        input_data1 = scaler1.transform(input_data1)
        # Предсказание углов отклонения рулей
        prediction = model.predict(input_data)
        prediction1 = model1.predict(input_data1)
        pitch_c.setpoint=prediction[0][0]
        pitch_control=pitch_c.update(elevator_yoke)
        roll_c.setpoint=179.5-heading
        roll_control=roll_c.update(aileron_yoke)
        throttle=prediction1[0][0]
        #print([pitch_control,roll_control])


        if pitch_control>=-0.0:
            pitch_control=-0.0
        if pitch_control<=-0.5:
            pitch_control=-0.5
        if throttle>=0.65:
            throttle=0.65
        if throttle<=0.33:
            throttle=0.33
        if altitude<700:
            flaps=1
        elif altitude<1600:
            flaps=0.66
        elif altitude<1800:
            flaps = 0.33

            #pitch_control=-0.07
        if altitude<425:
            pitch_control=0
            throttle = 0

        if roll_control>=0.1:
            roll_control=0.1
        if roll_control<=-0.1:
            roll_control=-0.1


        for i in range(len(ctr)):
            ctr1[i] = ctr[i]
        ctr1[0]=pitch_control
        ctr1[1]=roll_control
        ctr1[6]=brakes
        ctr1[2]=0
        ctr1[3]=throttle
        ctr1[5]=flaps
        # Отправка команд в X-Plane
        client.sendCTRL(ctr1)
        #print(pitch_control)

        # Задержка для обновления данных
        time.sleep(0.15)

# Запуск управления самолетом
control_airplane()