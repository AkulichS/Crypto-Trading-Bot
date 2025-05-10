import torch
import random
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
from datetime import datetime, timedelta
import backtrader as bt
import backtrader.indicators as btind
import matplotlib.pyplot as plt
from tensordict import TensorDict
# import environment and dataloader
from data.data_loader import TradingDataLoader
from agents.trading_agent import TradingAgent


class CustomPandasData(bt.feeds.PandasData):
    # Добавляем дополнительные линии
    lines = ('volatility', 'close_sma_diff', 'movement_success',  # 'pct_change', 'acceleration',
             'volume_norm', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9')

    # Связываем дополнительные линии с соответствующими столбцами DataFrame
    params = (
        # ('close_norm', None),
        ('volatility', None),
        ('close_sma_diff', None),
        ('movement_success', None),
        # ('pct_change', None),
        # ('acceleration', None),
        ('volume_norm', None),
        ('RSI_14', None),
        ('MACD_12_26_9', None),
        ('MACDh_12_26_9', None),
        ('MACDs_12_26_9', None),
    )

# Backtrader
class PPOAgentStrategy(bt.Strategy):

    def __init__(self, cfg, device):
        self.device = device
        # Создание модели
        self.agent = TradingAgent(cfg, device)
        self.agent.actor_module.eval()  # Перевод сети в режим оценки
        # === Загрузка чекпойнта ===
        checkpoint_path = cfg.test.checkpoint_path
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        actor_weights = {  # Извлечение весов для actor_network
            key.replace("actor_network_params.module.0.", ""): value
            for key, value in checkpoint["loss_module"].items()
            if key.startswith("actor_network_params.module.0.")
        }
        # Загрузка извлеченных весов в actor_net
        self.agent.actor_module.load_state_dict(actor_weights, strict=False)
        self.pos_size = 0.05


    def log(self, txt):
        dtime = (datetime(1, 1, 1) + timedelta(days=self.data.datetime[0] - 1)).strftime('%Y-%m-%d %H:%M:%S')
        print(f'{dtime}:  {txt}')


    def next(self):
        # Формируем входные данные для модели
        current_price = self.datas[0].close[-1]  # текущая цена
        position_size = self.position.size  # текущая позиция (в лотах)
        balance = self.broker.get_cash()  # доступные средства (кэш)
        initial_balance = self.broker.startingcash
        # net_worth = (balance + abs(position) * price) / initial_balance
        net_worth = (balance + abs(position_size) * current_price) / initial_balance

        # Подготовка наблюдения
        obs = np.array([
            # self.datas[0].close_norm[-1],
            self.datas[0].volatility[-1],
            self.datas[0].close_sma_diff[-1],
            self.datas[0].movement_success[-1],
            # self.datas[0].pct_change[-1],
            # self.datas[0].acceleration[-1],
            self.datas[0].volume_norm[-1],
            self.datas[0].RSI_14[-1],
            self.datas[0].MACD_12_26_9[-1],
            self.datas[0].MACDh_12_26_9[-1],
            self.datas[0].MACDs_12_26_9[-1],
            position_size,
            net_worth
        ], dtype=np.float32)

        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)  # батч из одного элемента
        # obs_td = TensorDict({"observation": obs_tensor}, device=self.device)

        # Получить действие от модели
        # with torch.no_grad():
        #     prediction = self.agent(obs_tensor)[1].item()
        prediction = self.agent.select_action(obs_tensor)

        # Действия агента
        if prediction == 1:  # Покупка
            if self.position.size == 0.0:  # Нет позиции, просто покупаем.0
                entry_price = self.data.close[0]
                stop_price = entry_price * 0.99  # стоп-лосс на 2% ниже входа
                entry_order = self.buy(size=self.pos_size)
                self.sl_order = self.sell(
                    size=self.pos_size,
                    exectype=bt.Order.Stop,
                    price=stop_price,
                    parent=entry_order
                )
                print(f"Покупка: цена {self.data.close[0]}")
            elif self.position.size < 0.0:  # Закрываем позицию продажи перед покупкой
                self.close()
                print(f"Закрытие продажи: цена {self.position.price}")
            else:
                if self.data.close[0] <= self.position.price * 0.98:
                    self.close()
                    print(f"Stop-loss покупки: цена {self.position.price}")

        elif prediction == 2:  # Продажа
            if self.position.size == 0.0:  # Нет позиции, просто продаем
                entry_price = self.data.close[0]
                stop_price = entry_price * 1.02  # стоп-лосс на 2% выше входа

                entry_order = self.sell(size=self.pos_size)
                self.sl_order = self.buy(
                    size=self.pos_size,
                    exectype=bt.Order.Stop,
                    price=stop_price,
                    parent=entry_order
                )
                print(f"Продажа: цена {self.data.close[0]}")
            elif self.position.size > 0.0:  # Закрываем позицию покупки перед продажей
                self.close()
                print(f"Закрытие покупки: цена {self.position.price}")
            else:
                if self.data.close[0] >= self.position.price * 1.02:
                    self.close()
                    print(f"Stop-loss продажи: цена {self.position.price}")


    # def notify_order(self, order):
    #     if order.status in [order.Completed]:  # Ордер выполнен
    #         if order.isbuy():
    #             print(f"Куплено по цене {order.executed.price}")
    #         elif order.issell():
    #             print(f"Продано по цене {order.executed.price}")
    #     elif order.status in [order.Canceled, order.Margin, order.Rejected]:  # Проблемы с ордером
    #         print(f"Ордер {order.ref} был отменен/отклонен")



def main(cfg):

    # seed = 42
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    device = "cpu"

    # Load data
    loader = TradingDataLoader("data/BTCUSDT_1d.csv", from_date="2024-01-01 01:30:00", to_date="2025-01-01 01:00:00")
    df = loader.load_data()
    pd.set_option('display.width', 1000)  # Устанавливаем ширину вывода
    # print(df.head())

    # Подготовка данных для Backtrader
    bt_data = CustomPandasData(
        dataname=df,
        datetime=None,  # использовать индекс как datetime,
        open="open",
        high="high",
        low="low",
        close="close",
        # volume="volume",
        # close_norm="close_norm",
        volatility="volatility",
        close_sma_diff="close_sma_diff",
        movement_success="movement_success",
        # pct_change="pct_change",
        # acceleration="acceleration",
        volume_norm="volume_norm",
        RSI_14="RSI_14",
        MACD_12_26_9="MACD_12_26_9",
        MACDh_12_26_9="MACDh_12_26_9",
        MACDs_12_26_9="MACDs_12_26_9",
        # timeframe=bt.TimeFrame.Days,
        timeframe=bt.TimeFrame.Minutes,
        compression=30  # Компрессия: 1 день
    )

    # Инициализация Cerebro
    cerebro = bt.Cerebro()
    cerebro.adddata(bt_data)
    cerebro.addstrategy(PPOAgentStrategy, cfg=cfg, device=device)
    cerebro.broker.setcash(10000)  # Устанавливаем стартовый капитал
    cerebro.broker.setcommission(commission=0.00001)

    # Запуск
    print("Запуск симуляции...")
    cerebro.run()
    cerebro.plot(style='candle', volume=False)  # volume=False
    plt.show()


if __name__ == "__main__":
    # === Загрузка конфигурации из YAML ===
    cfg = OmegaConf.load("config.yaml")
    main(cfg)