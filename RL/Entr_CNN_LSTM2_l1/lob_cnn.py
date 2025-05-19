import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class DeepLOBEncoder(BaseFeaturesExtractor):
    """
    Интерпретирует первые 2*K*2 элементов obs как (2, K, 2)-тензор книги,
    пропускает через CNN, добавляет портфельные фичи.
    """
    def __init__(self, observation_space, *, lob_K=100, port_dim=16, out_dim=128):
        super().__init__(observation_space, features_dim=out_dim)
        self.K = lob_K
        self.port_dim = port_dim

        # -------- CNN блок --------
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(3, 2)), nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=(3, 1)), nn.ELU(),
            nn.Flatten()
        )
        with th.no_grad():
            dummy = th.zeros(1, 2, lob_K, 2)
            cnn_dim = self.cnn(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(cnn_dim + port_dim, out_dim),
            nn.ELU()
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        K = self.K
        lob_flat, port = obs[:, :2*K*2], obs[:, 2*K*2:]          # split
        lob = lob_flat.view(-1, 2, K, 2)                         # (B,2,K,2)
        z = self.cnn(lob)
        z = th.cat([z, port], dim=1)
        return self.fc(z)
