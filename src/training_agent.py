from omegaconf import OmegaConf
import torch
from torch import nn
import numpy as np
from datetime import datetime
from torchrl.collectors import MultiSyncDataCollector, SyncDataCollector
from tensordict.nn import TensorDictModule, InteractionType
from torchrl.envs.transforms import CatTensors
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.trainers import Trainer
from torchrl.modules import ProbabilisticActor, ValueOperator
from torch.distributions import Categorical
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
    loader = TradingDataLoader("data/BTCUSDT_1d.csv", from_date="2020-01-01 03:00:00", to_date="2024-01-01 03:00:00")
    df = loader.load_data()

    # environment parameters
    params = {
        'features_start_index': 4,
        'window_size': cfg.env.window_size,
        'initial_balance': cfg.env.initial_balance,
        'position_size': cfg.env.position_size,
        'episode_len': cfg.env.episode_len,
        'fee': cfg.env.fee,
        'device': device,
        'pos_log': cfg.log.pos_log,
        'episode_log': cfg.log.episode_log,
    }

    # init environment
    env = TradingEnv(df=df, **params)

    agent = TradingAgent(cfg, device)
    policy_module = agent.actor
    value_module = agent.critic

    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=device,
        # max_frames_per_traj=-1,
        # postproc=transform_action,
    )

    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=cfg.loss.clip_epsilon,
        entropy_bonus=bool(cfg.loss.entropy_coef),
        entropy_coef=cfg.loss.entropy_coef,
        # these keys match by default, but we set this for completeness
        critic_coef=cfg.loss.critic_coef,
        loss_critic_type=cfg.loss.loss_critic_type,
        # normalize_advantage=True,
    )

     # Create optimizer
    optim = torch.optim.Adam(
        loss_module.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.eps,
    )

    # Создание планировщика: уменьшаем lr в 10 раз каждые 1000 шагов
    scheduler = StepLR(optim, step_size=1, gamma=0.98)

    # Получаем текущую дату и время для уникального имени файла
    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    filepath = f"checkpoints/model_checkpoint_{timestamp}.pth"

    tb_logger = TensorboardLogger(f'{timestamp}_el{cfg.env.episode_len}_sb{cfg.optim.steps_per_batch}_ec{cfg.loss.entropy_coef}', './log/tensorboard/')

    trainer = Trainer(
        collector=collector,
        optim_steps_per_batch=cfg.optim.steps_per_batch,
        total_frames=cfg.collector.total_frames,
        loss_module=loss_module,
        optimizer=optim,
        frame_skip=1,
        save_trainer_file=filepath,
        logger=tb_logger,
        log_interval=1000,
        # output_transform=output_transform,
    )

    log_reward = LogReward("reward")
    trainer.register_op("post_steps_log", log_reward)

    trainer.train()

if __name__ == "__main__":

    # === Загрузка конфигурации из YAML ===
    cfg = OmegaConf.load("config.yaml")
    main(cfg)