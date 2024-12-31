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

def predict():

    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import LSTM, Dense

    # Пример данных (замените на ваши данные)
    # Предположим, что данные представлены в виде массива numpy
    data = np.random.rand(100, 3)  # 1000 точек данных, 3 измерения

    # Предобработка данных
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Разделение данных на обучающую и тестовую выборки
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    # Создание входных и выходных данных для LSTM
    def create_dataset(data, time_step=1):
        X, Y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), :])
            Y.append(data[i + time_step, :])
        return np.array(X), np.array(Y)

    time_step = 10
    X_train, Y_train = create_dataset(train_data, time_step)
    X_test, Y_test = create_dataset(test_data, time_step)

    # Создание модели LSTM
    model = Sequential()
    model.add(LSTM(400, return_sequences=True, input_shape=(time_step, X_train.shape[2])))
    model.add(LSTM(400))
    model.add(Dense(X_train.shape[2]))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Обучение модели
    model.fit(X_train, Y_train, epochs=300, batch_size=32, validation_data=(X_test, Y_test), verbose=1)

    # Прогнозирование
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Обратное преобразование данных
    train_predict = scaler.inverse_transform(train_predict)
    Y_train = scaler.inverse_transform(Y_train)
    test_predict = scaler.inverse_transform(test_predict)
    Y_test = scaler.inverse_transform(Y_test)

    # Оценка модели
    from sklearn.metrics import mean_squared_error

    train_score = mean_squared_error(Y_train, train_predict)
    test_score = mean_squared_error(Y_test, test_predict)

    print(f'Train Score: {train_score}')
    print(f'Test Score: {test_score}')

    # Визуализация результатов
    import matplotlib.pyplot as plt

    # Пример визуализации для одного измерения
    plt.figure(figsize=(14, 5))
    plt.plot(Y_test[:, 0], label='True Values')
    plt.plot(test_predict[:, 0], label='Predicted Values')
    plt.legend()
    plt.show()
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

        posx = 55.98
        posy = 37.42

        pos = [posx, posy, 1000, 0, 0, 0, 1]
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


monitor()