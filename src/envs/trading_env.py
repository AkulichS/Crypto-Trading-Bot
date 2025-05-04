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

OBS_START_INDEX = 7  # индекс (номер столбца) начала признаков входящих в observation
PENALTY = 200.0


class TradingEnv(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, data, **kwargs):
        """
        Среда для обучения торгового агента.

        Args:
            data (numpy)               Данные для обучения агента
            initial_balance (float)    Начальный баланс агента
            position_size (float)      Размер позиции
            fee (float)                Размер комиссии
            device (str)               Устройство для выполнения (cpu/gpu)
            log_file (str, optional)   Путь к лог файлу
        """

        super().__init__()

        assert len(data) > 500, "Размер данных должен быть больше 500!"
        self.data = data
        self.initial_balance = kwargs.get("initial_balance", 1000)
        self.position_size = kwargs.get("position_size", 0.001)
        self.episode_len = kwargs.get("episode_len", 1000)
        self.fee = kwargs.get("fee", 0.05)  # размер комиссии в центах
        self.device = kwargs.get("device", "cpu")
        self.log_file = kwargs.get("log_file", "env_log.csv")

        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0   # индекс текущего сотояния
        self.episode_reward = 0.0
        self.max_drawdown_threshold = self.initial_balance // 2

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

        # Очистка содержимого файла при инициализации
        with open(self.log_file, mode='w') as f:
            fields = ["step", "|", "action", "|", "reward", "|", "done", "|", "terminated", "|", "episode_reward", "|", "balance"]
            f.write("".join(fields) + "\n")  # Открытие в режиме 'w' автоматически очищает файл


    def _reset(self, td: TensorDictBase = None, **kwargs) -> TensorDictBase:
        """Сброс среды до начального состояния. """

        self.episode_reward = 0.0
        self.position = 0.0


        if td is not None:
            if td.get('terminated'):
                self.balance = self.initial_balance  # начальный баланс
                self.current_step = round(self.current_step, -3) # округление до ближайшей тысячи

        # Устанавливаем текущий шаг
        self.current_step = self.current_step % len(self.data)

        # Получаем состояние
        states = self._get_observation()
        states.update(self.full_done_spec.zero())  # обновление done_spec

        # Сохраняем в лог
        if self.log_file:
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
            torch.tensor([self.balance / self.initial_balance], dtype=torch.float32, device=self.device)
        ], dim=0)

        return states


    def _step(self, td: TensorDictBase) -> TensorDictBase:
        """Выполнение действия в среде."""

        action = td['action'].item()

        # 1. Определяем стоимость активов до выполнения действия
        previous_net_worth = self.net_worth

        # 2. Выполняем действие
        self._take_action(action)

        # 3. Переход к слудующему состоянию (именно после выполнения действия)
        self.current_step += 1

        # 4. Определяем статус завершения эпизода
        is_done, is_terminated = (False, False)
        if self.current_step % self.episode_len == 0:
            is_done = True  # эпизод закончен
            is_terminated = False
        elif self.balance < self.max_drawdown_threshold or self.current_step >= self.data.shape[0] - 1:
            is_terminated = True  # эпизод прерван из-за ограничения баланса или конец данных
            is_done = True        # эпизод закончен

        # 5. Расчет награды
        reward = self._calculate_reward(previous_net_worth)
        self.episode_reward += reward

        # 6. Формирование нового состояния
        next_state = self._get_observation()
        next_state.set("reward", torch.tensor([reward], dtype=self.reward_spec.dtype, device=self.device))
        next_state.set("done", torch.tensor([is_done], dtype=self.full_done_spec['done'].dtype, device=self.device))
        next_state.set("terminated", torch.tensor([is_terminated], dtype=self.full_done_spec['terminated'].dtype, device=self.device))

        # 7. Запись в лог
        if self.log_file:
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
            return -0.001
        return current_net_worth - previous_net_worth

    @property
    def net_worth(self):
        return (self.balance + self.position * self.current_price) / self.initial_balance # нормализуем баланс

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
            "episode_reward": round(self.episode_reward, 6),
            "line5": "|",
            "balance": round(self.balance, 2),
        }
        # Логирует строку с данными шага.
        with open(self.log_file, "a") as f:
            values = [str(v) for v in step_data.values()]
            f.write("".join(values) + "\n")                # Записываем строку данных


    def _set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)



# env = TradingEnv(data=data, **params)
#
# env.reset()