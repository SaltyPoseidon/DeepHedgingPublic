# cnn_lstm_policy.py
import torch as th, torch.nn as nn
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from lob_cnn import DeepLOBEncoder                # CNN (уже есть)

class CNNLSTMPolicy(RecurrentActorCriticPolicy):
    """
    features_extractor  → 1-слойный LSTM → actor / critic головы.
    """
    def __init__(self, observation_space, action_space, lr_schedule,
                 lstm_hidden_size=128, **kw):
        self.lstm_hidden_size = lstm_hidden_size
        super().__init__(observation_space, action_space, lr_schedule,
                         features_extractor_class=DeepLOBEncoder,
                         features_extractor_kwargs=dict(
                             lob_K=100, port_dim=16, out_dim=128),
                         **kw)

    # ───────── build networks ─────────────────────────────────
    def _build_mlp_extractor(self) -> None:
        # CNN уже создан -> self.features_dim == 128
        self.lstm = nn.LSTM(input_size=self.features_dim,
                            hidden_size=self.lstm_hidden_size,
                            batch_first=True)
        self.actor_net  = nn.Linear(self.lstm_hidden_size, self.action_dist.latent_dim)
        self.critic_net = nn.Linear(self.lstm_hidden_size, 1)

    # ───────── forward helpers (SB3-contrib API) ─────────────
    def forward_rnn(self, features, states, seq_lengths):
        # features shape: (B*T, feat)  -> (B, T, feat)
        bsz = seq_lengths.shape[0]; time_len = seq_lengths.max()
        feat_seq = features.view(bsz, time_len, -1)
        out, (h, c) = self.lstm(feat_seq, states)
        out = out.reshape(-1, self.lstm_hidden_size)   # back to (B*T, hidden)
        return out, (h, c)

    def get_actor_latent(self, latent):  # π-сеть
        return self.actor_net(latent)

    def get_critic_latent(self, latent): # V̂-сеть
        return self.critic_net(latent)

    def _get_initial_state(self, batch_size: int):
        h = th.zeros((1, batch_size, self.lstm_hidden_size), device=self.device)
        c = th.zeros_like(h)
        return (h, c)
