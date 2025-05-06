import numpy as np
import pandas as pd
import torch
import heapq
import random
import csv
from tabulate import tabulate
from torchrl.envs import EnvBase
from tensordict import TensorDictBase, TensorDict
from torchrl.data import DiscreteTensorSpec
from torchrl.data import Bounded, Unbounded, CompositeSpec, Composite, Categorical


class TradingEnv(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, data, **kwargs):
        """
        Среда для обучения торгового агента.

        Args:
            data (numpy)                   Данные для обучения агента
            initial_balance (float)        Начальный баланс агента
            position_size (float)          Размер позиции
            fee (float)                    Размер комиссии
            device (str)                   Устройство для выполнения (cpu/gpu)
            pos_log (str, optional)        Путь к лог файлу позиций
            episode_log (str, optional)    Путь к лог файлу эпизодов
        """

        super().__init__()

        assert len(data) > 500, "Размер данных должен быть больше 500!"
        self.data = data
        self.initial_balance = kwargs.get("initial_balance", 1000)
        self.position_size = kwargs.get("position_size", 0.001)
        self.episode_len = kwargs.get("episode_len", 1000)
        self.fee = kwargs.get("fee", 0.05)  # размер комиссии в центах
        self.device = kwargs.get("device", "cpu")
        self.pos_log = kwargs.get("pos_log", "env_pos_log.csv")
        self.episode_log = kwargs.get("episode_log", "env_episode_log.csv")

        self.balance = self.initial_balance
        self.full_balance = self.initial_balance
        self.position = 0
        self.current_step = 0   # индекс текущего сотояния
        self.episode_reward = 0.0
        self.episodes = 0
        self.episode_step = 0
        self.max_drawdown_threshold = self.initial_balance // 2
        self.all_episode_rewards = []

        # Определение спецификаций среды:
        # 1. Определяем пространство состояний
        self.observation_spec = CompositeSpec({
            "observation": Unbounded(
                shape=torch.Size([self.data.shape[1]-1]),  # Размер наблюдения, кол-во признаков - 1 (Close)
                dtype=torch.float32,  # Тип данных
                device=self.device    # Устройство
            )
        })
        # 2. Определяем пространство действий
        self.action_spec = DiscreteTensorSpec(  # Дискретное пространство действий:
            n=3,  # 0 - удержание, 1 - покупка, 2 - продажа
            shape=torch.Size([1]),  # размерность действия
            dtype=torch.int8,
            device=self.device      # CPU или CUDA
        )
        # 3. Определяем спецификацию наград
        self.reward_spec = Unbounded(shape=torch.Size([1]), dtype=torch.float32, device=self.device)

        # 4. Определяем спецификацию done, terminated
        self.full_done_spec = CompositeSpec(
            done=Bounded(shape=torch.Size([1]), dtype=torch.bool, low=False, high=True, device=self.device),
            terminated=Bounded(shape=torch.Size([1]), dtype=torch.bool, low=False, high=True, device=self.device)
        )

        # Очистка содержимого файла при инициализации и запись заголовка
        with open(self.pos_log, mode='w') as f:
            fields = ["step", "|", "action", "|", "reward", "|", "done", "|", "terminated",
                      "|", "position", "|", "episode_reward", "|", "balance", "|", "net_worth"]
            f.write("".join(fields) + "\n")  # Открытие в режиме 'w' автоматически очищает файл

        with open(self.episode_log, mode='w') as f:
            fields = ["episode", "|", "mean_episode_reward", "|", "sum_episode_reward", "|", "positive_rate",
                      "|", "balance", "|", "net_worth", "|", "full_balance"]
            f.write("".join(fields) + "\n")  # Открытие в режиме 'w' автоматически очищает файл


    def _reset(self, td: TensorDictBase = None, **kwargs) -> TensorDictBase:
        """Сброс среды до начального состояния. """

        self.balance = self.initial_balance  # начальный баланс
        self.episode_reward = 0.0
        self.position = 0.0
        self.episode_step = 0

        if td is not None:
            if td.get('terminated'):
                self.current_step = round(self.current_step, -3) # округление до ближайшей тысячи

        # Устанавливаем текущий шаг
        self.current_step = self.current_step % len(self.data)

        # Получаем состояние
        states = self._get_observation()
        states.update(self.full_done_spec.zero())  # обновление done_spec

        # Сохраняем в лог
        if self.pos_log:
            self._log_step('-', 0.0, False, False)

        return states


    def _get_observation(self):
        # Извлекаем данные для текущего состояния
        states = TensorDict({
            "observation": torch.tensor(
                self.data[self.current_step, 1:], # берем текущую строку признаков, исключая close
                dtype=self.observation_spec["observation"].dtype,
                device=self.device,
            ),
        }, device=self.device)

        # Добавляем в наблюдение баланс и позицию (buy/sell/hold)
        states["observation"] = torch.cat([
            states["observation"],
            torch.tensor([self.position], dtype=torch.float32, device=self.device),
            torch.tensor([self.net_worth], dtype=torch.float32, device=self.device)
        ], dim=0)

        return states


    def _step(self, td: TensorDictBase) -> TensorDictBase:
        """Выполнение действия в среде."""

        # 1. Определяем стоимость активов до выполнения действия
        previous_net_worth = self.net_worth

        # 2. Выполняем действие
        action = td['action'].item()
        self._take_action(action)

        # 3. Переход к слудующему состоянию (именно после выполнения действия)
        self.current_step += 1
        self.episode_step += 1

        # 4. Расчет награды
        reward = self._calculate_reward(previous_net_worth)
        self.episode_reward += reward

        net_worth = self.net_worth * self.initial_balance  # текущая стоимость активов (не нормализованная)

        # 5. Определяем статус завершения эпизода
        is_done, is_terminated = (False, False)
        if self.current_step % self.episode_len == 0:
            is_done, is_terminated = True, False      # эпизод закончен
            self.episodes += 1                        # подсчет эпизодов для лога
            self.full_balance += self.episode_reward  # состояние баланса за все эпизоды
            self.all_episode_rewards.append(self.episode_reward)
            self._log_episode()                       # сохраняем данные эпизода в лог
        elif net_worth < self.max_drawdown_threshold or self.current_step >= self.data.shape[0] - 1:
            is_terminated = True  # эпизод прерван из-за ограничения баланса или конец данных
            is_done = True        # эпизод закончен

        # 6. Формирование нового состояния
        next_state = self._get_observation()
        next_state.set("reward", torch.tensor([reward], dtype=self.reward_spec.dtype, device=self.device))
        next_state.set("done", torch.tensor([is_done], dtype=self.full_done_spec['done'].dtype, device=self.device))
        next_state.set("terminated", torch.tensor([is_terminated], dtype=self.full_done_spec['terminated'].dtype, device=self.device))

        # 7. Запись в лог
        if self.pos_log:
            self._log_step(action, reward, is_done, is_terminated)

        return next_state


    def _take_action(self, action):
        match action:
            case 1:  # Покупка
                if self.position == 0:   # если нет позиции, то открываем buy
                    self.position = self.position_size
                    self.balance -= self.position * self.current_price + self.fee

                elif self.position < 0:  # если имеется позиция sell, то закрываем ее
                    self.balance += abs(self.position) * self.current_price - self.fee
                    self.position = 0.0

            case 2:  # Продажа
                if self.position == 0:  # если нет позиции, то открываем sell
                    self.position = -self.position_size
                    self.balance -= abs(self.position) * self.current_price + self.fee

                elif self.position > 0:  # если имеется позиция buy, то закрываем ее
                    self.balance += self.position * self.current_price - self.fee
                    self.position = 0.0


    def _calculate_reward(self, previous_net_worth):
        current_net_worth = self.net_worth # баланс после выполнения действия
        if current_net_worth - previous_net_worth == 0:
            return 0.0 # -0.001
        return (current_net_worth - previous_net_worth) * self.initial_balance

    @property
    def net_worth(self):
        return (self.balance + abs(self.position) * self.current_price) / self.initial_balance # нормализуем баланс

    @property
    def current_price(self):
        return self.data[self.current_step][0]


    def _log_step(self, action, reward, is_done, is_terminated):
        step_data = {
            "step": self.current_step,
            "line0": "|",
            "action": action,
            "line1": "|",
            "reward": round(reward, 6),
            "line2": "|",
            "done": is_done,
            "line3": "|",
            "terminated": is_terminated,
            "line4": "|",
            "position": self.position,
            "line5": "|",
            "episode_reward": round(self.episode_reward, 6),
            "line6": "|",
            "balance": round(self.balance, 2),
            "line7": "|",
            "net_worth": round(self.net_worth, 4),
        }
        # Логирует строку с данными шага.
        with open(self.pos_log, "a") as f:
            values = [str(v) for v in step_data.values()]
            f.write("".join(values) + "\n")   # Записываем строку данных
            if is_done:
                f.write("\n")                 # Пустая строка для отделения эпизодов

    def _log_episode(self):
        rewards = np.array(self.all_episode_rewards)
        positive_sum = rewards[rewards > 0].sum()
        total_abs_sum = np.abs(rewards).sum()
        positive_rate = positive_sum / total_abs_sum if total_abs_sum > 0 else 0
        step_data = {
            "episode": self.episodes,
            "line0": "|",
            "mean_episode_reward": round(np.mean(self.all_episode_rewards), 4),
            "line1": "|",
            "sum_episode_reward": round(self.episode_reward, 6),
            "line2": "|",
            "positive_rate": round(positive_rate, 2),
            "line3": "|",
            "balance": round(self.balance, 2),
            "line4": "|",
            "net_worth": round(self.net_worth, 4),
            "line5": "|",
            "full_balance": round(self.full_balance, 2),
        }
        # Логирует строку с данными шага.
        with open(self.episode_log, "a") as f:
            values = [str(v) for v in step_data.values()]
            f.write("".join(values) + "\n")                # Записываем строку данных


    def _set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)



# env = TradingEnv(data=data, **params)
#
# env.reset()