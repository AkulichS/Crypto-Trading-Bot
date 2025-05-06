from omegaconf import OmegaConf
import torch, torchrl
from datetime import datetime
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss
from torchrl.trainers import Trainer
from torchrl.trainers import LogReward
from torch.optim.lr_scheduler import StepLR
from torchrl.record.loggers.tensorboard import TensorboardLogger

# import environment and dataloader
from data.data_loader import TradingDataLoader
from envs.trading_env import TradingEnv
from agents.trading_agent import TradingAgent


def main(cfg):
    device = "cpu"  #torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    loader = TradingDataLoader("data/BTCUSDT_30m.csv", from_date="2024-01-01 01:30:00", to_date="2025-01-01 01:00:00")
    data = loader.load_data()

    # environment parameters
    params = {
        'initial_balance': cfg.env.initial_balance,
        'position_size': cfg.env.position_size,
        'episode_len': cfg.env.episode_len,
        'fee': cfg.env.fee,
        'device': device,
        'pos_log': cfg.log.test_pos_log,
        'episode_log': cfg.log.test_episode_log,
    }

    # init environment
    env = TradingEnv(data=data, **params)

    agent = TradingAgent(cfg, env, device)
    policy_module = agent.actor

    # === Загрузка чекпойнта ===
    checkpoint_path = cfg.test.checkpoint_path
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    actor_weights = {  # Извлечение весов для actor_network
        key.replace("actor_network_params.module.0.", ""): value
        for key, value in checkpoint["loss_module"].items()
        if key.startswith("actor_network_params.module.0.")
    }
    # Загрузка извлеченных весов в actor_net
    policy_module.load_state_dict(actor_weights, strict=False)
    policy_module.eval()  # Перевод сети в режим оценки

    # === TensorBoard логгер ===
    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    tb_logger = TensorboardLogger(f'Test_{timestamp}', './log/tensorboard/')

    # === Collector в режиме deterministic ===
    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=device,
        exploration_type=torchrl.envs.utils.ExplorationType.MODE  #DETERMINISTIC,  # для теста: выбираем argmax
    )

    rewards = []
    for i, tensordict in enumerate(collector):
        reward = tensordict["next", "reward"].sum().item()
        rewards.append(reward)
        tb_logger.log_scalar("test/episode_reward", reward, i)

    collector.shutdown()


if __name__ == "__main__":

    # === Загрузка конфигурации из YAML ===
    cfg = OmegaConf.load("config.yaml")
    main(cfg)
