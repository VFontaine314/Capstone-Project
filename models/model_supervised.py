import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from torch.utils.data import DataLoader, TensorDataset

# --- Import from data folder ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.data_generation import poisson_gene, gen_vec

# --- 1. Sub-Network (Learns Scale & Translation) ---
class SubNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)

# --- 2. Coupling Layer (Supervised) ---
class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.dim = dim
        self.split = dim // 2
        self.net = SubNet(self.split, dim - self.split, hidden_dim)

    def forward(self, x):
        x1, x2 = x[:, :self.split], x[:, self.split:]
        t = self.net(x1)
        y2 = x2 + t
        y = torch.cat([x1, y2], dim=1)
        return y

# --- 3. Supervised INN Preconditioner ---
class SupervisedINN(nn.Module):
    def __init__(self, dim, n_layers=8, hidden_dim=128):
        super().__init__()
        self.layers = nn.ModuleList()
        self.perms = []

        for _ in range(n_layers):
            self.layers.append(AffineCoupling(dim, hidden_dim))
            perm = torch.randperm(dim)
            self.perms.append(perm)

    def forward(self, x):
        for layer, perm in zip(self.layers, self.perms):
            x = x[:, perm]
            x = layer(x)
        return x

# --- Main Execution ---
if __name__ == "__main__":
    # Config
    NX, NY = 10, 10
    SAMPLES = 20000      
    LAYERS = 8
    HIDDEN = 256
    EPOCHS = 100
    LR = 1e-3
    BATCH_SIZE = 256      

    # 1. Get Dimension from A
    print(f"Generating Poisson Matrix ({NX}x{NY})...")
    A_scipy, _ = poisson_gene(nx=NX, ny=NY)
    DIM = A_scipy.shape[0]

    # 2. Generate Training Data
    print(f"Generating {SAMPLES} training vectors...")
    v_train, w_train = gen_vec(dim=DIM, samples=SAMPLES, A=A_scipy)
    
    # Convert to float64
    v_train = v_train.double()
    w_train = w_train.double()

    # Normalize inputs
    v_mean = v_train.mean(dim=0, keepdim=True)
    v_std = v_train.std(dim=0, keepdim=True) + 1e-6
    v_train_norm = (v_train - v_mean) / v_std

    # Dataset & DataLoader
    dataset = TensorDataset(v_train_norm, w_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Setup model
    model = SupervisedINN(dim=DIM, n_layers=LAYERS, hidden_dim=HIDDEN)
    model.double()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # --- Training ---
    print("Starting Supervised Training...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for v_batch, w_batch in dataloader:
            optimizer.zero_grad()
            w_pred = model(v_batch)
            loss = loss_fn(w_pred, w_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if epoch % 10 == 0:
            lr_current = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}: MSE Loss = {avg_loss:.6f}, LR = {lr_current:.2e}")

    # --- Verification ---
    model.eval()
    with torch.no_grad():
        test_w = torch.randn(1, DIM).double()
        test_v = torch.matmul(test_w, torch.tensor(A_scipy.toarray()).double().t())
        test_v_norm = (test_v - v_mean) / v_std
        pred_w = model(test_v_norm)
        error = torch.nn.functional.mse_loss(pred_w, test_w)
        print(f"\nFinal Test MSE: {error.item():.6f}")

    print("Supervised INN trained. You can now use forward pass as preconditioner.")
