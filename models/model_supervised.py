import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# --- Import from data folder ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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
            nn.Linear(hidden_dim, out_dim * 2) # Output both s and t
        )
        # Identity Init: Initialize weights to 0 so the layer starts as Identity
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(self, x):
        params = self.net(x)
        s, t = params.chunk(2, dim=-1)
        # Widen clamp to allow learning small factors (Crucial for Poisson)
        s = torch.tanh(s) * 10.0  
        return s, t

# --- 2. Coupling Layer (With Flip Support) ---
class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden_dim=128, flip=False):
        super().__init__()
        self.dim = dim
        self.flip = flip
        
        # Handle Odd Dimensions safely
        self.n_split = dim // 2
        self.n_keep = dim - self.n_split
        
        if not self.flip:
            net_in = self.n_split
            net_out = self.n_keep
        else:
            net_in = self.n_keep
            net_out = self.n_split
            
        self.net = SubNet(net_in, net_out, hidden_dim)

    def forward(self, x):
        # Always split the same way: x1 (top), x2 (bottom)
        x1 = x[:, :self.n_split]
        x2 = x[:, self.n_split:]

        if not self.flip:
            # x1 is constant, updates x2
            s, t = self.net(x1)
            y1 = x1
            y2 = x2 * torch.exp(s) + t
        else:
            # x2 is constant, updates x1
            s, t = self.net(x2)
            y1 = x1 * torch.exp(s) + t
            y2 = x2

        return torch.cat([y1, y2], dim=1)

# --- 3. Supervised INN Preconditioner ---
class SupervisedINN(nn.Module):
    def __init__(self, dim, n_layers=8, hidden_dim=128):
        super().__init__()
        
        # Stack layers with alternating flips (False, True, False, True...)
        self.layers = nn.ModuleList([
            AffineCoupling(dim, hidden_dim, flip=(i % 2 == 1))
            for i in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SupervisedMLP(nn.Module):
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


def train_supervised_inn(
    A_scipy,
    samples=100000,
    n_layers=4,
    hidden_dim=121,
    epochs=50,
    lr=1e-3,
    batch_size=1000):
    # 1. Get Dimension from A
    #print(f"Generating Poisson Matrix ({nx}x{ny})...")
    dim = A_scipy.shape[0]

    # 2. Generate Training Data
    #print(f"Generating {samples} training vectors...")
    v_train, w_train = gen_vec(dim=dim, samples=samples, A=A_scipy)
    
    # Convert to float64 (Double Precision is mandatory for Matrix Inversion)
    v_train = v_train.double()
    w_train = w_train.double()

    # 3. Normalize Inputs (v)
    # The network sees v_norm and tries to output w
    v_mean = v_train.mean(dim=0, keepdim=True)
    v_std = v_train.std(dim=0, keepdim=True) + 1e-6
    v_train_norm = (v_train - v_mean) / v_std

    #print(f"Data Normalized. Mean: {v_mean.mean():.4f}, Std: {v_std.mean():.4f}")

    # Dataset & DataLoader
    dataset = TensorDataset(v_train_norm, w_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Setup model
    model = SupervisedINN(dim=dim, n_layers=n_layers, hidden_dim=hidden_dim)
    model.double() # Move model to Float64
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    loss_fn = nn.MSELoss()

    # --- Training ---
    #print("Starting Supervised Training...")
    model.train()
    avg_losses = []
    for _ in range(epochs):
        total_loss = 0.0
        for v_batch, w_batch in dataloader:
            optimizer.zero_grad()
            
            # Forward: Input (Residual v) -> Output (Correction w)
            w_pred = model(v_batch)
            
            loss = loss_fn(w_pred, w_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        avg_losses.append(avg_loss)
        scheduler.step(avg_loss)

        #if epoch % 10 == 0:
            #lr_current = optimizer.param_groups[0]['lr']
            #print(f"Epoch {epoch}: MSE Loss = {avg_loss:.8f}, LR = {lr_current:.2e}")

    return model, v_mean, v_std, avg_losses


def train_supervised_mlp(
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

    model = SupervisedMLP(dim=dim, hidden_dim=hidden_dim)
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

# --- Main Execution ---
if __name__ == "__main__":
    # Train supervised INN model
    A_scipy, _ = poisson_gene(nx=11, ny=11)
    model, v_mean, v_std, avg_losses = train_supervised_inn(A_scipy=A_scipy, epochs=10)

    # Calculate test error
    model.eval()
    with torch.no_grad():
        # Generate a fresh test case
        test_w = torch.randn(1, A_scipy.shape[0]).double()
        test_v = torch.matmul(test_w, torch.tensor(A_scipy.toarray()).double().t())
        
        # IMPORTANT: You must normalize the input v just like training
        test_v_norm = (test_v - v_mean) / v_std
        
        # Predict
        pred_w = model(test_v_norm)
        
        error = torch.nn.functional.mse_loss(pred_w, test_w)
        print(f"\nFinal Test MSE: {error.item():.8f}")
    
    # Plot training loss graphs
    plt.plot(avg_losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Supervised Training Graph")
    plt.legend()
    plt.show()
