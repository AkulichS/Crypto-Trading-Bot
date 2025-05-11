import torch
import random
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
from datetime import datetime, timedelta
import backtrader as bt
import matplotlib.pyplot as plt
from tensordict import TensorDict
# import environment and dataloader
from data.data_loader import TradingDataLoader
from agents.trading_agent import TradingAgent


class CustomPandasData(bt.feeds.PandasData):
    # Добавляем дополнительные линии
    lines = ('volatility', 'close_sma_diff', 'movement_success',
             'volume_norm', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9')

    # Связываем дополнительные линии с соответствующими столбцами DataFrame
    params = (
        # ('close_norm', None),
        ('volatility', None),
        ('close_sma_diff', None),
        ('movement_success', None),
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
        prediction = self.agent.select_action(obs_tensor).item()

        # Действия агента
        if prediction == 1:  # Покупка
            if self.position.size == 0.0:  # Нет позиции, просто покупаем.0
                # self.buy(size=self.pos_size)
                # Выполняем покупку с установкой стоп-лосса и тейк-профита
                self.buy_bracket(
                    size=self.pos_size,
                    # limitprice=self.data.close[0] * 1.1,
                    price=self.data.close[0],
                    stopprice=self.data.close[0] * 0.9,
                    exectype=bt.Order.Market,   # bt.Order.Limit,
                    # exectype=bt.Order.Market  # Исполнение по рыночным ценам
                )
                print(f"Покупка: цена {self.data.close[0]}")
            elif self.position.size < 0.0:  # Закрываем позицию продажи перед покупкой
                self.close()
                print(f"Закрытие продажи: цена {self.position.price}")

        elif prediction == 2:  # Продажа
            if self.position.size == 0.0:  # Нет позиции, просто продаем
                # self.sell(size=self.pos_size)
                # Выполняем продажу с установкой стоп-лосса и тейк-профита
                self.sell_bracket(
                    size=self.pos_size,
                    # limitprice=self.data.close[0] * 0.98,
                    price=self.data.close[0],
                    stopprice=self.data.close[0] * 1.1,
                    exectype=bt.Order.Market,  # bt.Order.Limit,
                )
                print(f"Продажа: цена {self.data.close[0]}")
            elif self.position.size > 0.0:  # Закрываем позицию покупки перед продажей
                self.close()
                print(f"Закрытие покупки: цена {self.position.price}")


    # def notify_order(self, order):
    #     if order.status in [order.Completed]:  # Ордер выполнен
    #         if order.isbuy():
    #             print(f"Куплено по цене {order.executed.price}")
    #         elif order.issell():
    #             print(f"Продано по цене {order.executed.price}")
    #     elif order.status in [order.Canceled, order.Margin, order.Rejected]:  # Проблемы с ордером
    #         print(f"Ордер {order.ref} был отменен/отклонен")



def print_metrics(strat):
    print('\n=== Strategy Performance ===')

    sharpe = strat.analyzers.sharpe.get_analysis()
    print(f"Sharpe Ratio: {sharpe.get('sharperatio', 'n/a'):.3f}")

    dd = strat.analyzers.drawdown.get_analysis()
    print(f"Max Drawdown [%]: {dd['max']['drawdown']:.2f}")
    print(f"Max Drawdown Duration: {dd['max']['len']} bars")

    returns = strat.analyzers.returns.get_analysis()
    print(f"Total Return [%]: {returns['rtot'] * 100:.2f}")
    print(f"Annualized Return [%]: {returns['rnorm'] * 100:.2f}")

    trades = strat.analyzers.trades.get_analysis()
    total = trades.total.get('total', 0)
    win = trades.won.get('total', 0)
    loss = trades.lost.get('total', 0)
    win_rate = (win / (win + loss) * 100) if (win + loss) > 0 else 0
    print(f"Total Trades: {total}")
    print(f"Win Rate [%]: {win_rate:.2f}")

    net_pnl = trades.pnl.net.get('total', 0)
    print(f"Net Profit: {net_pnl:.2f}")



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
        volatility="volatility",
        close_sma_diff="close_sma_diff",
        movement_success="movement_success",
        volume_norm="volume_norm",
        RSI_14="RSI_14",
        MACD_12_26_9="MACD_12_26_9",
        MACDh_12_26_9="MACDh_12_26_9",
        MACDs_12_26_9="MACDs_12_26_9",
        timeframe=bt.TimeFrame.Days,
        compression=1  # Компрессия: 1 день
    )

    # Инициализация Cerebro
    cerebro = bt.Cerebro()
    cerebro.adddata(bt_data)
    cerebro.addstrategy(PPOAgentStrategy, cfg=cfg, device=device)
    cerebro.broker.setcash(10000)  # Устанавливаем стартовый капитал
    cerebro.broker.setcommission(commission=0.0001)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')

    # Запуск
    print("Запуск симуляции...")
    results = cerebro.run()
    strat = results[0]

    print_metrics(strat)

    cerebro.plot(style='candle', volume=False)  # volume=False
    plt.show()


if __name__ == "__main__":
    # === Загрузка конфигурации из YAML ===
    cfg = OmegaConf.load("config.yaml")
    main(cfg)





# from backtesting import Backtest, Strategy
#
# class PPOAgentStrategy(Strategy):
#     def init(self):
#         # Загрузка модели
#         self.device = 'cpu'
#         self.agent = TradingAgent(cfg, device=self.device)
#         self.agent.actor_module.eval()
#         # Загрузка весов модели
#         checkpoint = torch.load(cfg.test.checkpoint_path, map_location=self.device)
#         actor_weights = {
#             key.replace("actor_network_params.module.0.", ""): value
#             for key, value in checkpoint["loss_module"].items()
#             if key.startswith("actor_network_params.module.0.")
#         }
#         self.agent.actor_module.load_state_dict(actor_weights, strict=False)
#
#     def next(self):
#         # Подготовка наблюдения
#         obs = np.array([
#             self.data.volatility[-1],
#             self.data.close_sma_diff[-1],
#             self.data.movement_success[-1],
#             self.data.volume_norm[-1],
#             self.data.RSI_14[-1],
#             self.data.MACD_12_26_9[-1],
#             self.data.MACDh_12_26_9[-1],
#             self.data.MACDs_12_26_9[-1],
#             self.position.size,
#             (self.equity / 10000)  # net_worth
#         ], dtype=np.float32)
#
#         obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)
#
#         # Получение действия от модели
#         with torch.no_grad():
#             prediction = self.agent.select_action(obs_tensor)
#
#         # Выполнение действия
#         if prediction == 1 and not self.position:
#             self.buy()
#         elif prediction == 2 and not self.position:
#             self.sell()
#         elif prediction == 0 and self.position:
#             self.position.close()
#
#
# loader = TradingDataLoader('d:/Neuro_Net/Projects/RL/Trading_Agent/PPO_Trading_Agent/src/data/BTCUSDT_1d.csv', from_date="2024-01-01 01:30:00", to_date="2025-01-01 01:00:00")
# df = loader.load_data()
# cfg = OmegaConf.load("config.yaml")
# df.rename(columns={
#     'open': 'Open',
#     'high': 'High',
#     'low': 'Low',
#     'close': 'Close',
#     'volume': 'Volume'
# }, inplace=True)
#
# bt = Backtest(df, PPOAgentStrategy, cash=10000, commission=0.0001)
# stats = bt.run()
# bt.plot()





# import torch
# import pandas as pd
# import numpy as np
# import vectorbt as vbt
# from omegaconf import OmegaConf
# from datetime import datetime, timedelta
# from data.data_loader import TradingDataLoader
# from agents.trading_agent import TradingAgent
#
#
# # Создание кастомных данных для Vectorbt
# class CustomPandasData:
#     def __init__(self, df):
#         self.df = df
#         self.close = df['close']
#         self.volatility = df['volatility']
#         self.close_sma_diff = df['close_sma_diff']
#         self.movement_success = df['movement_success']
#         self.volume_norm = df['volume_norm']
#         self.RSI_14 = df['RSI_14']
#         self.MACD_12_26_9 = df['MACD_12_26_9']
#         self.MACDh_12_26_9 = df['MACDh_12_26_9']
#         self.MACDs_12_26_9 = df['MACDs_12_26_9']
#
#
# def generate_signals(model, data, device, initial_cash=10000):
#     """
#     Используем модель для получения торговых сигналов.
#     """
#     # Предположим, что изначально вся equity равна initial_cash
#     net_worth = np.full(len(data.close), initial_cash, dtype=np.float32)
#     position_size = np.zeros(len(data.close), dtype=np.float32)
#
#     obs = np.array([
#         data.volatility.values,
#         data.close_sma_diff.values,
#         data.movement_success.values,
#         data.volume_norm.values,
#         data.RSI_14.values,
#         data.MACD_12_26_9.values,
#         data.MACDh_12_26_9.values,
#         data.MACDs_12_26_9.values,
#         position_size,
#         net_worth / net_worth[0],  # нормализованный net_worth
#     ]).T
#
#     obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
#     actions = model.select_action(obs_tensor)
#     return actions.cpu().numpy()
#
#
# def main(cfg):
#     # Загрузка конфигурации
#     device = "cpu"
#     loader = TradingDataLoader("data/BTCUSDT_1d.csv", from_date="2024-01-01 01:30:00", to_date="2025-01-01 01:00:00")
#     df = loader.load_data()
#
#     # Подготовка данных для Vectorbt
#     data = CustomPandasData(df)
#
#     # Загрузка модели
#     agent = TradingAgent(cfg, device)
#     agent.actor_module.eval()  # Перевод модели в режим оценки
#     checkpoint_path = cfg.test.checkpoint_path
#     checkpoint = torch.load(checkpoint_path, weights_only=True)
#
#     actor_weights = {key.replace("actor_network_params.module.0.", ""): value
#                      for key, value in checkpoint["loss_module"].items()
#                      if key.startswith("actor_network_params.module.0.")}
#
#     agent.actor_module.load_state_dict(actor_weights, strict=False)
#
#     # Генерация сигналов с использованием модели
#     signals = generate_signals(agent, data, device)
#
#     # Инициализация Vectorbt портфеля
#     portfolio = vbt.Portfolio.from_signals(
#         data.close,  # Цена закрытия
#         entries=signals == 1,  # Покупка
#         exits=signals == 2,  # Продажа
#         init_cash=10000,  # Начальный капитал
#         sl_stop=0.98,  # Стоп-лосс
#         tp_stop=1.1,  # Тейк-профит
#         freq='1D',  # Частота торгов
#     )
#
#     # Вывод результатов
#     portfolio.total_return().vbt.plot(title="Total Return")
#     portfolio.value().vbt.plot(title="Portfolio Value")
#     print(portfolio.stats())
#
# if __name__ == "__main__":
#     # Загрузка конфигурации из YAML
#     cfg = OmegaConf.load("config.yaml")
#     main(cfg)
