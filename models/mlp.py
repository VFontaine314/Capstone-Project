import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.data_generation import poisson_gene, gen_vec


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(
    nx=4,
    ny=4,
    samples=20000,
    hidden_dim=256,
    epochs=100,
    lr=1e-3,
    batch_size=256,
):
    A_scipy, _ = poisson_gene(nx=nx, ny=ny)
    dim = A_scipy.shape[0]

    v_train, w_train = gen_vec(dim=dim, samples=samples, A=A_scipy)
    v_train = v_train.double()
    w_train = w_train.double()

    v_mean = v_train.mean(dim=0, keepdim=True)
    v_std = v_train.std(dim=0, keepdim=True) + 1e-6
    v_train_norm = (v_train - v_mean) / v_std

    dataset = TensorDataset(v_train_norm, w_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MLP(dim=dim, hidden_dim=hidden_dim)
    model.double()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        total_loss = 0.0
        for v_batch, w_batch in dataloader:
            optimizer.zero_grad()
            w_pred = model(v_batch)
            loss = loss_fn(w_pred, w_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)

    return model, v_mean, v_std, A_scipy


if __name__ == "__main__":
    model, v_mean, v_std, A_scipy = train_mlp()
    _ = (model, v_mean, v_std, A_scipy)
