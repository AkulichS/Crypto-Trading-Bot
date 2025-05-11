from omegaconf import OmegaConf
import random
import numpy as np
import torch, torchrl
from datetime import datetime
from torchrl.collectors import SyncDataCollector
from tensordict import TensorDict
from torchrl.record.loggers.tensorboard import TensorboardLogger
# import environment and dataloader
from data.data_loader import TradingDataLoader
from envs.trading_env import TradingEnv
from agents.trading_agent import TradingAgent
import plotly.graph_objects as go


def plot_candles_with_actions(data, actions):
    """
    Построение свечного графика с маркерами действий (buy/sell).
    :param data: DataFrame с индексом времени и колонками ['open', 'high', 'low', 'close']
    :param actions: Список или массив действий (0 — hold, 1 — buy, 2 — sell), длиной len(data)
    """
    import plotly.graph_objects as go

    time_index = data.index

    fig = go.Figure()

    # Свечной график
    fig.add_trace(go.Candlestick(
        x=time_index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name='Candles',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))

    # Списки координат для buy/sell
    buy_x, buy_y = [], []
    sell_x, sell_y = [], []

    for i, action in enumerate(actions):
        ts = time_index[i]
        if action == 1:  # buy
            buy_x.append(ts)
            buy_y.append(data['low'].iloc[i] * 0.985)  # немного ниже свечи
        elif action == 2:  # sell
            sell_x.append(ts)
            sell_y.append(data['high'].iloc[i] * 1.015)  # немного выше свечи

    # BUY: Синий треугольник вверх
    fig.add_trace(go.Scatter(
        x=buy_x, y=buy_y,
        mode='markers',
        marker=dict(symbol='triangle-up', size=10, color='green'),
        name='Buy',
        showlegend=True
    ))

    # SELL: Красный треугольник вниз
    fig.add_trace(go.Scatter(
        x=sell_x, y=sell_y,
        mode='markers',
        marker=dict(symbol='triangle-down', size=10, color='red'),
        name='Sell',
        showlegend=True
    ))

    fig.update_layout(
        title="Свечной график с действиями агента",
        xaxis_title="Время",
        yaxis_title="Цена",
        xaxis_rangeslider_visible=False,
        height=700
    )

    fig.show()


def main(cfg):
    # seed = 42
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    device = "cpu"  #torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    loader = TradingDataLoader("data/BTCUSDT_30m.csv", from_date="2024-01-01 01:30:00", to_date="2025-01-11 01:00:00")
    df = loader.load_data()

    # environment parameters
    params = {
        'features_start_index': 6,
        'initial_balance': cfg.env.initial_balance,
        'position_size': cfg.env.position_size,
        'episode_len': len(df)-1,
        'fee': cfg.env.fee,
        'device': device,
        'pos_log': cfg.log.test_pos_log,
        'episode_log': cfg.log.test_episode_log,
    }

    # init environment
    env = TradingEnv(df=df, **params)

    # Создание модели
    agent = TradingAgent(cfg, device)
    agent.actor_module.eval()  # Перевод сети в режим оценки

    # === Загрузка чекпойнта ===
    checkpoint_path = cfg.test.checkpoint_path
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    actor_weights = {  # Извлечение весов для actor_network
        key.replace("actor_network_params.module.0.", ""): value
        for key, value in checkpoint["loss_module"].items()
        if key.startswith("actor_network_params.module.0.")
    }
    # Загрузка извлеченных весов в actor_net
    agent.actor_module.load_state_dict(actor_weights, strict=False)

    # === TensorBoard логгер ===
    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    tb_logger = TensorboardLogger(f'Test_{timestamp}', './log/tensorboard/')

    rewards = []
    num_epochs = 3

    for epoch in range(num_epochs):
        td = env.reset(to_first_step=True)
        total_reward = 0.0
        actions = []

        for step in range(df.shape[0]-1):
            obs_tensor = td['next']['observation'].unsqueeze(0) if 'next' in td else td['observation'].unsqueeze(0)  # батч 1xD
            with torch.no_grad():
                action = agent.select_action(obs_tensor)

            if (action == 1 and env.position > 0) or (action == 2 and env.position < 0):
                action = 0
            actions.append(action)
            action_td = TensorDict({  # передаём action как TensorDict
                "action": torch.tensor([action], dtype=torch.int8, device=device)
            }, device=device)

            # шаг в среде
            td = env.step(action_td)
            reward = td['next']["reward"].item()
            done = td['next']["done"].item()
            total_reward += reward

            if done:
                td = env.reset()

        rewards.append(total_reward)
        tb_logger.log_scalar("test/episode_reward", total_reward, epoch)

    print("Награды:", rewards)
    print("Средняя награда:", sum(rewards) / len(rewards))

    # Построение свечного графика
    plot_candles_with_actions(df, actions)


    # # === Collector в режиме deterministic ===
    # collector = SyncDataCollector(
    #     env,
    #     policy_module,
    #     frames_per_batch=64,
    #     total_frames=len(data),
    #     device=device,
    #     exploration_type=torchrl.envs.utils.ExplorationType.MODE  #DETERMINISTIC,  # для теста: выбираем argmax
    # )
    #
    # rewards = []
    # for i, tensordict in enumerate(collector):
    #     reward = tensordict["next", "reward"].sum().item()
    #     rewards.append(reward)
    #     tb_logger.log_scalar("test/episode_reward", reward, i)
    #
    # collector.shutdown()





if __name__ == "__main__":

    # === Загрузка конфигурации из YAML ===
    cfg = OmegaConf.load("config.yaml")
    main(cfg)
