import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Определение среды для управления тангажем самолёта
class AirplanePitchEnv(gym.Env):
    def __init__(self):
        super(AirplanePitchEnv, self).__init__()
        self.action_space = spaces.Box(low=np.array([0.0, 0.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-np.pi]), high=np.array([np.pi]), dtype=np.float32)
        self.state = np.array([0.0])  # Начальный угол наклона самолёта
        self.target_pitch = np.pi / 4  # Целевой угол наклона самолёта
        self.Kp = 0.5
        self.Ki = 0.1
        self.Kd = 0.05
        self.integral = 0.0
        self.previous_error = 0.0

    def reset(self):
        self.state = np.array([0.0])
        self.integral = 0.0
        self.previous_error = 0.0
        return self.state

    def step(self, action):
        self.Kp, self.Ki, self.Kd = action
        error = self.target_pitch - self.state[0]
        self.integral += error
        derivative = error - self.previous_error
        self.previous_error = error

        # PID-регулятор
        control_signal = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        # Обновление состояния системы (упрощенная модель)
        self.state = np.array([self.state[0] + control_signal])

        # Награда как отрицательное расстояние до целевого угла наклона
        reward = -np.abs(self.state[0] - self.target_pitch)
        done = np.isclose(self.state[0], self.target_pitch, atol=0.01)  # Завершение, если достигнута цель
        return self.state, reward, done, {}

# Проверка среды
env = AirplanePitchEnv()
check_env(env)

# Создание и обучение модели
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Сохранение модели
model.save("pid_tuner_airplane_pitch")

# Загрузка модели (если требуется)
# model = PPO.load("pid_tuner_airplane_pitch")

# Использование модели для управления тангажем самолёта
obs = env.reset()
for _ in range(100):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        break

print(f"Optimized PID parameters: Kp={env.Kp}, Ki={env.Ki}, Kd={env.Kd}")
print(f"Final pitch angle: {obs[0]}")