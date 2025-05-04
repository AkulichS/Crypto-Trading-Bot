import numpy as np
import pandas as pd
import pandas_ta as ta
import mplfinance as mpf


class TradingDataLoader:
    """
    Загрузчик данных для торговой среды.

    Args:
        filename          Путь к CSV-файлу
        from_date         Дата начала датасета, например 2019-09-09 21:00:00
        to_date           Дата конца датасета, например 2025-01-11 02:30:00
    """

    def __init__(self, filename, from_date='', to_date=''):
        self.filename = filename
        self.from_date = from_date
        self.to_date = to_date

    def load_data(self, return_df: bool = False):  # Загрузка данных (результат в df для графика или numpy для модели)
        df = pd.read_csv(self.filename)
        df['time'] = pd.to_datetime(df['time'])

        if self.from_date and self.to_date:
            df = df[(df['time'] >= self.from_date) & (df['time'] <= self.to_date)]
        elif self.from_date:
            df = df[df['time'] >= self.from_date]
        elif self.to_date:
            df = df[df['time'] <= self.to_date]

        return self._preprocessing_data(df, return_df=return_df)


    @staticmethod
    def _preprocessing_data(df: pd.DataFrame, return_df: bool = False):
        df = df.set_index('time')
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
        df['sma21'] = df['close'].rolling(window=21, min_periods=1).mean()

        # Масштаб для нормализации
        high_max = df['high'].max()
        low_min = df['low'].min()
        scale = high_max - low_min

        # Создание новых признаков
        # Ценовые признаки нормализуем через scale
        df['close_norm'] = (df['close'] - low_min) / scale
        df['volatility'] = (df['high'] - df['low']) / scale
        df['close_sma_diff'] = (df['close'] - df['sma21']) / scale
        df['movement_success'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, 1e-8)

        # Относительные признаки стандартизируем
        df['return'] = df['close'].pct_change().replace(0, 1e-8)
        df['return'] = (df['return'] - df['return'].mean()) / df['return'].std()
        df['acceleration'] = df['return'].diff().replace(0, 1e-8)
        df['acceleration'] = (df['acceleration'] - df['acceleration'].mean()) / df['acceleration'].std()
        df['volume_norm'] = (df['volume'] - df['volume'].mean()) / df['volume'].std()

        # Индикаторы
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)

        # RSI нормализуем от 0 до 1
        df['RSI_14'] = df['RSI_14'] / 100.0

        # MACD + сигнал — стандартизуем
        for col in ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']:
            df[col] = (df[col] - df[col].mean()) / df[col].std()

        df = df.ffill().bfill().astype(np.float32)  # Заполняем NaN ближайшим значением (сначала вперёд, потом назад)

        features = ['close', 'close_norm', 'volatility', 'close_sma_diff', 'movement_success', 'return',
                    'acceleration', 'volume_norm', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']

        return df if return_df else df[features].to_numpy(dtype=np.float32)


    # def plot_candlestick(self, df):
    #     mpf.plot(
    #         df[['open', 'high', 'low', 'close', 'volume']],
    #         type='candle',
    #         style='yahoo',
    #         volume=True
    #     )
