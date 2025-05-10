import torch
from torch import nn
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, ValueOperator
from torch.distributions import Categorical
from torchrl.data import DiscreteTensorSpec
from tensordict import TensorDict


class TradingAgent:
    def __init__(self, cfg, device='cpu'):
        self.cfg = cfg
        self.device = device
        self.action_spec = DiscreteTensorSpec(cfg.model.out_dim)
        self.actor_module = self._build_actor_module()
        self.actor = self._build_actor()
        self.critic = self._build_critic()

    def _build_actor_module(self):
        actor_net = nn.Sequential(
            nn.Linear(self.cfg.model.in_dim, self.cfg.model.actor.h1, device=self.device),
            nn.ReLU(),
            nn.Linear(self.cfg.model.actor.h1, self.cfg.model.actor.h2, device=self.device),
            nn.ReLU(),
            nn.Linear(self.cfg.model.actor.h2, self.cfg.model.out_dim, device=self.device),
        )
        return TensorDictModule(
            module=actor_net,
            in_keys=["observation"],
            out_keys=["logits"]
        )

    def _build_actor(self):
        return ProbabilisticActor(
            module=self.actor_module,
            spec=self.action_spec,
            in_keys={"logits": "logits"},
            distribution_class=Categorical,
            return_log_prob=True
        )

    def _build_critic(self):
        critic_net = nn.Sequential(
            nn.Linear(self.cfg.model.in_dim, self.cfg.model.critic.h1, device=self.device),
            nn.ReLU(),
            nn.Linear(self.cfg.model.critic.h1, self.cfg.model.critic.h2, device=self.device),
            nn.ReLU(),
            nn.Linear(self.cfg.model.critic.h2, 1, device=self.device),
        )
        return ValueOperator(
            module=critic_net,
            in_keys=["observation"]
        )

    def select_action(self, obs_tensor):
        """Выбор действия: mode=True — детерминированный, False — сэмплированный."""
        obs_td = TensorDict({"observation": obs_tensor}, batch_size=[1], device=self.device)
        with torch.no_grad():
            td_out = self.actor_module(obs_td)         # Получаем logits напрямую
            logits = td_out["logits"]
            probs = torch.softmax(logits, dim=-1)      # Превращаем в вероятности
            return torch.argmax(probs, dim=-1).item()  # Действие с максимальной вероятностью


# class TradingAgent:
#     def __init__(self, cfg, device='cpu'):
#         self.cfg = cfg
#         self.device = device
#         self.action_spec = DiscreteTensorSpec(cfg.model.out_dim)
#         self.actor = self._build_actor()
#         self.critic = self._build_critic()
#
#     def _build_actor(self):
#         actor_net = nn.Sequential(
#             nn.Linear(self.cfg.model.in_dim, self.cfg.model.actor.h1, device=self.device),
#             nn.ReLU(),
#             nn.Linear(self.cfg.model.actor.h1, self.cfg.model.actor.h2, device=self.device),
#             nn.ReLU(),
#             nn.Linear(self.cfg.model.actor.h2, self.cfg.model.out_dim, device=self.device),
#         )
#         self.actor_module = TensorDictModule(
#             module=actor_net,
#             in_keys=["observation"],
#             out_keys=["logits"]
#         )
#         return ProbabilisticActor(
#             module=self.actor_module,           # нейросеть actor_net обернутая в TensorDictModule для доступности TensorDict
#             spec=self.action_spec,
#             in_keys={"logits": "logits"},       # ключ logits для распределения
#             distribution_class=Categorical,
#             return_log_prob=True                # возвращать логарифм вероятности для PPO
#         )
#
#     def _build_critic(self):
#         critic_net = nn.Sequential(
#             # nn.Flatten(start_dim=-2),
#             nn.Linear(self.cfg.model.in_dim, self.cfg.model.critic.h1, device=self.device),
#             nn.ReLU(),
#             nn.Linear(self.cfg.model.critic.h1, self.cfg.model.critic.h2, device=self.device),
#             nn.ReLU(),
#             nn.Linear(self.cfg.model.critic.h2, 1, device=self.device),
#         )
#         return ValueOperator(
#             module=critic_net,
#             in_keys=["observation"]
#         )

