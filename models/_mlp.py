import torch.nn as nn


def _build_mlp(in_dim: int, hidden_dims, out_dim: int, activation: str) -> nn.Sequential:
    act_cls = nn.ReLU if activation == "relu" else nn.Tanh
    layers = []
    prev = int(in_dim)
    for h in hidden_dims:
        layers.append(nn.Linear(prev, int(h)))
        layers.append(act_cls())
        prev = int(h)
    layers.append(nn.Linear(prev, int(out_dim)))
    return nn.Sequential(*layers)
