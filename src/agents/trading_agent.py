import torch
from torch import nn
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, ValueOperator
from torch.distributions import Categorical
from torchrl.data import DiscreteTensorSpec
from tensordict import TensorDict


class WrapConv1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, padding='same')
    def forward(self, x):
        if x.dim() < 3:
            x = x.permute(1, 0)
        else:
            x = x.permute(0, 2, 1)
        return self.conv1d(x)

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels=10, embed_dim=64, num_heads=2):
        super().__init__()
        self.conv_proj = WrapConv1d(in_channels, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.conv_proj(x)                 # [B, embed_dim, T]
        if x.dim() < 3:
            x = x.permute(1, 0)
        else:
            x = x.permute(0, 2, 1)            # [B, T, embed_dim] → for attention
        attn_out, _ = self.attn(x, x, x)      # Self-Attention
        x = self.norm(attn_out + x)           # Residual connection
        return x


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
            # WrapConv1d(self.cfg.model.in_dim, self.cfg.model.lstm.hidden),
            # nn.LayerNorm([self.cfg.model.lstm.hidden, self.cfg.env.window_size]),
            SelfAttentionBlock(in_channels=self.cfg.model.in_dim, embed_dim=self.cfg.model.lstm.hidden),
            nn.Flatten(start_dim=-2),
            nn.Linear(self.cfg.model.lstm.hidden * self.cfg.env.window_size, self.cfg.model.actor.h1, device=self.device),
            nn.ReLU(),
            nn.Dropout(p=0.2),
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
            module=self.actor_module,        # нейросеть actor_net обернутая в TensorDictModule для доступности TensorDict
            spec=self.action_spec,
            in_keys={"logits": "logits"},    # ключ logits для распределения
            distribution_class=Categorical,
            return_log_prob=True             # возвращать логарифм вероятности для PPO
        )

    def _build_critic(self):
        critic_net = nn.Sequential(
            # WrapConv1d(self.cfg.model.in_dim, self.cfg.model.lstm.hidden),
            # nn.LayerNorm([self.cfg.model.lstm.hidden, self.cfg.env.window_size]),
            SelfAttentionBlock(in_channels=self.cfg.model.in_dim, embed_dim=self.cfg.model.lstm.hidden),
            nn.Flatten(start_dim=-2),
            nn.Linear(self.cfg.model.lstm.hidden * self.cfg.env.window_size, self.cfg.model.critic.h1, device=self.device),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.cfg.model.critic.h1, self.cfg.model.critic.h2, device=self.device),
            nn.ReLU(),
            nn.Linear(self.cfg.model.critic.h2, 1, device=self.device),
        )

        return ValueOperator(
            module=critic_net,
            in_keys=["observation"]
        )

    def select_action(self, obs_tensor):
        """
        obs_tensor: torch.Tensor of shape [1, 100, feature_dim]
        """
        obs_td = TensorDict({"observation": obs_tensor}, batch_size=[1], device=self.device)
        with torch.no_grad():
            td_out = self.actor_module(obs_td)
            logits = td_out["logits"]
            probs = torch.softmax(logits, dim=-1)
            return torch.argmax(probs, dim=-1)



