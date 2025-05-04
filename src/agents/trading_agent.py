from torch import nn
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, ValueOperator
from torch.distributions import Categorical


class TradingAgent:
    def __init__(self, cfg, env, device):
        self.cfg = cfg
        self.env = env
        self.device = device
        self.actor = self._build_actor()
        self.critic = self._build_critic()

    def _build_actor(self):
        actor_net = nn.Sequential(
            # nn.Flatten(start_dim=-2),
            nn.Linear(self.cfg.model.in_dim, self.cfg.model.actor.h1, device=self.device),
            nn.ReLU(),
            nn.Linear(self.cfg.model.actor.h1, self.cfg.model.actor.h2, device=self.device),
            nn.ReLU(),
            nn.Linear(self.cfg.model.actor.h2, self.cfg.model.out_dim, device=self.device),
        )
        actor_module = TensorDictModule(
            module=actor_net,
            in_keys=["observation"],
            out_keys=["logits"]
        )
        return ProbabilisticActor(
            module=actor_module,             # нейросеть actor_net обернутая в TensorDictModule для доступности TensorDict
            spec=self.env.action_spec,
            in_keys={"logits": "logits"},    # ключ logits для распределения
            distribution_class=Categorical,
            return_log_prob=True             # возвращать логарифм вероятности для PPO
        )

    def _build_critic(self):
        critic_net = nn.Sequential(
            # nn.Flatten(start_dim=-2),
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

