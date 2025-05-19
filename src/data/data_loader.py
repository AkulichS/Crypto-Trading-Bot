import numpy as np
import pandas as pd
import pandas_ta
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

    def load_data(self):  # Загрузка данных
        df = pd.read_csv(self.filename)
        df['time'] = pd.to_datetime(df['time'])

        if self.from_date and self.to_date:
            df = df[(df['time'] >= self.from_date) & (df['time'] <= self.to_date)]
        elif self.from_date:
            df = df[df['time'] >= self.from_date]
        elif self.to_date:
            df = df[df['time'] <= self.to_date]

        return self._preprocessing_data(df)


    @staticmethod
    def _preprocessing_data(df: pd.DataFrame):
        df = df.set_index('time')
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)

        # Масштаб для нормализации
        # high_max = df['high'].max()
        # low_min = df['low'].min()
        # scale = high_max - low_min
        #
        # df['close_norm'] = (df['close'] - low_min) / scale

        window = 50
        mean_close_long = df['close'].rolling(window, min_periods=1).mean()
        std_close_long = df['close'].rolling(window, min_periods=1).std() + 1e-8
        # Рассчитываем z-score
        df['z_close_long'] = (df['close'] - mean_close_long) / std_close_long

        mean_close_short = df['close'].rolling(10, min_periods=1).mean()
        std_close_short = df['close'].rolling(10, min_periods=1).std() + 1e-8
        # Рассчитываем z-score
        df['z_close_short'] = (df['close'] - mean_close_short) / std_close_short

        # df['volume'] = np.log(df['volume'])
        mean_vol = df['volume'].rolling(window, min_periods=1).mean()
        std_vol = df['volume'].rolling(window, min_periods=1).std() + 1e-8
        # Рассчитываем z-score
        df['volume'] = (df['volume'] - mean_vol) / std_vol

        # Вычисляем относительный размер свечей для масштабирования верхнего и нижнего хвоста
        max_high_low = (df['high'] - df['low']).max()
        price_scale = (df['high'] - df['low']) / max_high_low

        high_tail = (
                (df['high'] - np.where(df['close'] > df['open'], df['close'], df['open'])) /
                ((df['high'] - df['low']).rolling(window, min_periods=1).mean() + 1e-8)
        )
        # Рассчитываем z-score для верхнего хвоста
        df['z_high_tail'] = (high_tail - high_tail.rolling(window, min_periods=1).mean()) / \
                            (high_tail.rolling(window, min_periods=1).std() + 1e-8)
        df['z_high_tail'] = df['z_high_tail'] * price_scale

        low_tail = (
                (df['low'] - np.where(df['close'] < df['open'], df['close'], df['open'])) /
                ((df['high'] - df['low']).rolling(window, min_periods=1).mean() + 1e-8)
        )
        # Рассчитываем z-score для нижнего хвоста
        df['z_low_tail'] = (low_tail - low_tail.rolling(window, min_periods=1).mean()) / \
                           (low_tail.rolling(window, min_periods=1).std() + 1e-8)
        df['z_low_tail'] = df['z_low_tail'] * price_scale

        # Индикаторы
        length = 14
        df.ta.adx(length=length, append=True)

        # df['trend_direction'] = 0              # боковик по умолчанию
        # trend_mask = df[f'ADX_{length}'] > 20  # фильтрация тренда (если <= 20, то боковик)
        # df.loc[(df[f'DMP_{length}'] > df[f'DMN_{length}']) & trend_mask, 'trend_direction'] = 1
        # df.loc[(df[f'DMP_{length}'] < df[f'DMN_{length}']) & trend_mask, 'trend_direction'] = -1

        # adx_norm = df[f'ADX_{length}'] / 100
        # df['adx_diff'] = adx_norm.diff().fillna(0)
        # df['adx'] = df['trend'].rolling(window=length, min_periods=1).mean()

        adx_diff = df[f'ADX_{length}'].diff().fillna(0)
        df['z_adx_diff'] = (adx_diff - adx_diff.rolling(window, min_periods=1).mean()) / \
                            adx_diff.rolling(window, min_periods=1).std()

        df = df.drop([f'ADX_{length}', f'DMP_{length}', f'DMN_{length}'], axis=1)

        df['sma_14'] = df['close'].rolling(window=5).mean()  # / df['close'].min()
        df['sma_14'] = ((df['sma_14'] - df['sma_14'].shift(1)) / df['sma_14'].shift(1)).rolling(window=14).mean() * 100

        df['z_close_diff'] = df['z_close_short'] - df['z_close_long']

        df = df.bfill().ffill()

        return df

# loader = TradingDataLoader("BTCUSDT_1d.csv", from_date="2020-01-01 03:00:00", to_date="2020-05-01 03:00:00")
# df1 = loader.load_data()
# print(df1)

