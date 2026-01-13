import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

# --- Import from data folder ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.data_generation import poisson_gene, gen_vec


# ============================================================
# 1. Sub-Network
# ============================================================
class SubNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim * 2)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        s, t = self.net(x).chunk(2, dim=-1)
        s = 0.1 * torch.tanh(s)
        return s, t


# ============================================================
# 2. Affine Coupling Layer (invertible)
# ============================================================
class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.split = dim // 2
        self.net = SubNet(self.split, dim - self.split, hidden_dim)

    def forward(self, x):
        x1, x2 = x[:, :self.split], x[:, self.split:]
        s, t = self.net(x1)
        y2 = x2 * torch.exp(s) + t
        log_det = s.sum(dim=1)
        return torch.cat([x1, y2], dim=1), log_det

    def inverse(self, y):
        y1, y2 = y[:, :self.split], y[:, self.split:]
        s, t = self.net(y1)
        x2 = (y2 - t) * torch.exp(-s)
        return torch.cat([y1, x2], dim=1)


# ============================================================
# 3. INN Model
# ============================================================
class INN(nn.Module):
    def __init__(self, dim, n_layers=8, hidden_dim=128):
        super().__init__()
        self.layers = nn.ModuleList()
        self.perms = []

        for _ in range(n_layers):
            self.layers.append(AffineCoupling(dim, hidden_dim))
            perm = torch.randperm(dim)
            self.register_buffer(f"perm_{len(self.perms)}", perm)
            self.perms.append(perm)

    def forward(self, x):
        log_det_total = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        for layer, perm in zip(self.layers, self.perms):
            x = x[:, perm]
            x, log_det = layer(x)
            log_det_total += log_det
        return x, log_det_total

    def inverse(self, z):
        for layer, perm in reversed(list(zip(self.layers, self.perms))):
            inv_perm = torch.argsort(perm)
            z = layer.inverse(z)
            z = z[:, inv_perm]
        return z


# ============================================================
# 4. Main
# ============================================================
if __name__ == "__main__":

    # --------------------
    # Config
    # --------------------
    NX, NY = 10, 10
    SAMPLES = 20000
    LAYERS = 8
    HIDDEN = 256
    EPOCHS = 100
    LR = 1e-3
    BATCH_SIZE = 256

    torch.set_default_dtype(torch.float64)

    # --------------------
    # Generate Poisson system
    # --------------------
    print(f"Generating Poisson matrix ({NX}x{NY})...")
    A_scipy, _ = poisson_gene(nx=NX, ny=NY)
    A = torch.tensor(A_scipy.toarray(), dtype=torch.float64)
    DIM = A.shape[0]

    # --------------------
    # Training data
    # --------------------
    print(f"Generating {SAMPLES} samples...")
    v_train, _ = gen_vec(dim=DIM, samples=SAMPLES, A=A_scipy)
    v_train = v_train.double()

    # --------------------
    # Normalization (per-dimension)
    # --------------------
    v_mean = v_train.mean(dim=0, keepdim=True)
    v_std = v_train.std(dim=0, keepdim=True) + 1e-6
    v_train = (v_train - v_mean) / v_std

    # --------------------
    # Model
    # --------------------
    model = INN(dim=DIM, n_layers=LAYERS, hidden_dim=HIDDEN).double()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    dataset = torch.utils.data.TensorDataset(v_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --------------------
    # Training
    # --------------------
    print("Starting INN training...")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for (v_batch,) in dataloader:
            optimizer.zero_grad()

            z, log_det = model(v_batch)

            log_pz = -0.5 * (z ** 2).sum(dim=1)
            loss = -(log_pz + log_det).mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)

        if epoch % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:4d} | NLL = {avg_loss:.6f} | LR = {lr:.2e}")

    # --------------------
    # Sanity checks
    # --------------------
    print("\nRunning sanity checks...")
    model.eval()

    with torch.no_grad():
        # latent check
        z, _ = model(v_train[:1000])
        print(f"Latent mean: {z.mean():.4e}, std: {z.std():.4f}")

        # inverse consistency
        v = v_train[:10]
        z, _ = model(v)
        v_rec = model.inverse(z)
        rel_err = torch.norm(v - v_rec) / torch.norm(v)
        print(f"Inverse relative error: {rel_err:.2e}")

        # learned inverse quality
        w = torch.randn(10, DIM, dtype=torch.float64)
        v = (A @ w.T).T
        v = (v - v_mean) / v_std

        z, _ = model(v)
        w_rec = model.inverse(z)

        mse = torch.nn.functional.mse_loss(w_rec, w)
        print(f"Inverse MSE (A⁻¹ quality): {mse:.4e}")
