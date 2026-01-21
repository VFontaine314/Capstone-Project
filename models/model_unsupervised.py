import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from torch.utils.data import DataLoader, TensorDataset

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
            nn.Linear(hidden_dim, out_dim * 2) 
        )
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(self, x):
        params = self.net(x)
        s, t = params.chunk(2, dim=-1)
        s = torch.tanh(s) * 10.0  
        return s, t

# --- 2. Coupling Layer ---
class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden_dim=128, flip=False):
        super().__init__()
        self.dim = dim
        self.flip = flip
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
        x1 = x[:, :self.n_split]
        x2 = x[:, self.n_split:]

        if not self.flip:
            s, t = self.net(x1)
            y1 = x1
            y2 = x2 * torch.exp(s) + t
        else:
            s, t = self.net(x2)
            y1 = x1 * torch.exp(s) + t
            y2 = x2

        return torch.cat([y1, y2], dim=1)

# --- 3. INN Model ---
class UnsupervisedINN(nn.Module):
    def __init__(self, dim, n_layers=8, hidden_dim=128):
        super().__init__()
        self.layers = nn.ModuleList([
            AffineCoupling(dim, hidden_dim, flip=(i % 2 == 1))
            for i in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def train_unsupervised_inn(
    nx=4,
    ny=4,
    samples=20000,
    n_layers=8,
    hidden_dim=256,
    epochs=100,
    lr=1e-3,
    batch_size=256,
):
    A_scipy, _ = poisson_gene(nx=nx, ny=ny)
    dim = A_scipy.shape[0]

    A_tensor = torch.tensor(A_scipy.toarray()).double()

    v_train, w_train = gen_vec(dim=dim, samples=samples, A=A_scipy)
    v_train = v_train.double()
    w_train = w_train.double()

    v_mean = v_train.mean(dim=0, keepdim=True)
    v_std = v_train.std(dim=0, keepdim=True) + 1e-6
    v_train_norm = (v_train - v_mean) / v_std

    dataset = TensorDataset(v_train_norm, w_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UnsupervisedINN(dim=dim, n_layers=n_layers, hidden_dim=hidden_dim)
    model.double()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    mse_loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for v_batch_norm, _ in dataloader: # Ignore w_batch (ground truth)
            optimizer.zero_grad()
            
            # 1. Forward: Model predicts w based on normalized v
            w_pred = model(v_batch_norm)
            
            # 2. [NEW] Physics Loss Calculation: || A * w_pred - v_physical ||^2
            # We must un-normalize v_batch to compare with physical A*w
            v_batch_phys = v_batch_norm * v_std + v_mean
            
            # Compute A * w (Using transpose for batch multiplication: (w @ A.T))
            v_reconstructed = torch.matmul(w_pred, A_tensor.t())
            
            # 3. Loss measures if the predicted w satisfies the Poisson equation
            loss = mse_loss_fn(v_reconstructed, v_batch_phys)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
    
    return model, v_mean, v_std, A_scipy


# --- Main Execution ---
if __name__ == "__main__":
    # Config
    NX, NY = 11, 11
    SAMPLES = 20000      
    LAYERS = 4
    HIDDEN = 256
    EPOCHS = 100
    LR = 1e-3
    BATCH_SIZE = 256      

    # 1. Get Matrix A and convert to Tensor for Physics Loss
    print(f"Generating Poisson Matrix ({NX}x{NY})...")
    A_scipy, _ = poisson_gene(nx=NX, ny=NY)
    DIM = A_scipy.shape[0]
    
    # [NEW] Convert A to dense tensor for Unsupervised Loss calculation
    A_tensor = torch.tensor(A_scipy.toarray()).double()

    # 2. Generate Training Data
    print(f"Generating {SAMPLES} training vectors...")
    v_train, w_train = gen_vec(dim=DIM, samples=SAMPLES, A=A_scipy)
    
    v_train = v_train.double()
    w_train = w_train.double() # Note: w_train is now ONLY used for verification, not training

    # 3. Normalize Inputs (v)
    v_mean = v_train.mean(dim=0, keepdim=True)
    v_std = v_train.std(dim=0, keepdim=True) + 1e-6
    v_train_norm = (v_train - v_mean) / v_std

    print(f"Data Normalized. Mean: {v_mean.mean():.4f}, Std: {v_std.mean():.4f}")

    dataset = TensorDataset(v_train_norm, w_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Setup model
    model = UnsupervisedINN(dim=DIM, n_layers=LAYERS, hidden_dim=HIDDEN)
    model.double() 
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # [NEW] Loss function is standard MSE, but inputs to it change
    mse_loss_fn = nn.MSELoss()

    # --- Training ---
    print("Starting Unsupervised Training...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for v_batch_norm, _ in dataloader: # Ignore w_batch (ground truth)
            optimizer.zero_grad()
            
            # 1. Forward: Model predicts w based on normalized v
            w_pred = model(v_batch_norm)
            
            # 2. [NEW] Physics Loss Calculation: || A * w_pred - v_physical ||^2
            # We must un-normalize v_batch to compare with physical A*w
            v_batch_phys = v_batch_norm * v_std + v_mean
            
            # Compute A * w (Using transpose for batch multiplication: (w @ A.T))
            v_reconstructed = torch.matmul(w_pred, A_tensor.t())
            
            # 3. Loss measures if the predicted w satisfies the Poisson equation
            loss = mse_loss_fn(v_reconstructed, v_batch_phys)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)

        if epoch % 10 == 0:
            lr_current = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}: Physics Loss = {avg_loss:.8f}, LR = {lr_current:.2e}")

    # --- Verification ---
    model.eval()
    with torch.no_grad():
        test_w = torch.randn(1, DIM).double()
        test_v = torch.matmul(test_w, A_tensor.t())
        
        test_v_norm = (test_v - v_mean) / v_std
        
        pred_w = model(test_v_norm)
        
        # We can still calculate MSE against ground truth here just to see if it worked
        error = torch.nn.functional.mse_loss(pred_w, test_w)
        print(f"\nFinal Test MSE (vs Hidden Ground Truth): {error.item():.8f}")
        
    print("Training complete using Unsupervised Physics Loss.")