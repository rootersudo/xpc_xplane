import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import m

# Генерация случайных данных для обучения
np.random.seed(42)

# Количество точек данных
num_samples = 1000

# Генерация данных для обучения
X_train = []
Y_train = []

for i in range(num_samples):
    latitude = np.random.uniform(55.97, 55.99)
    longitude = np.random.uniform(37.41, 37.43)
    altitude = np.random.uniform(800, 1200)
    pitch = np.random.uniform(-5, 5)
    roll = np.random.uniform(-5, 5)
    true_heading = np.random.uniform(0, 360)
    yaw = np.random.uniform(0, 360)
    throttle = np.random.uniform(0, 1)
    additional_features = np.random.uniform(-1, 1, 4)  # Дополнительные признаки

    X_train.append([latitude, longitude, altitude, pitch, roll, true_heading, yaw, throttle] + additional_features.tolist())
    Y_train.append([latitude, longitude, yaw, throttle])

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Создание модели
model = Sequential([
    Dense(64, activation='relu', input_shape=(12,)),
    Dense(64, activation='relu'),
    Dense(4)  # Выходной слой с 4 нейронами для latitude, longitude, yaw, throttle
])

# Компиляция модели
model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
model.fit(X_train, Y_train, epochs=100, batch_size=10)

# Пример данных для тестирования
X_test = [m.approach_points[:][0],m.approach_points[:][1],m.approach_points[:][2]]

# Предсказание
Y_pred = model.predict(X_test)
print("Предсказанные параметры для посадки:", Y_pred)

# Преобразование предсказанных данных в читаемый формат
latitude_pred = Y_pred[0][0]
longitude_pred = Y_pred[0][1]
yaw_pred = Y_pred[0][2]
throttle_pred = Y_pred[0][3]

print(f"Latitude: {latitude_pred:.6f} deg")
print(f"Longitude: {longitude_pred:.6f} deg")
print(f"Yaw: {yaw_pred:.2f} deg")
print(f"Throttle: {throttle_pred:.2f}")