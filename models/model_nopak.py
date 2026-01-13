import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# --- Import from data folder ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.data_generation import poisson_gene, gen_vec

# --- 1. Sub-Network (Learns Scale & Translation) ---
class SubNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim * 2) 
        )

    def forward(self, x):
        params = self.net(x)
        s, t = params.chunk(2, dim=-1)
        # Tanh constraint for stability
        s = torch.tanh(s) * 2.0 
        return s, t

# --- 2. Coupling Layer (Forward Only) ---
class AffineCouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.dim = dim
        self.split_dim = dim // 2
        self.sub_net = SubNet(self.split_dim, self.dim - self.split_dim, hidden_dim)

    def forward(self, x):
        # Split input
        x1 = x[:, :self.split_dim]
        x2 = x[:, self.split_dim:]
        
        # Calculate parameters based on x1
        s, t = self.sub_net(x1)

        # Transform x2
        y2 = x2 * torch.exp(s) + t
        
        # x1 stays the same
        y1 = x1
        
        return torch.cat([y1, y2], dim=1)

# --- 3. The Preconditioner Model ---
class PreconditionerINN(nn.Module):
    def __init__(self, dim, num_layers=4, hidden_dim=64):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(AffineCouplingLayer(dim, hidden_dim))
        
        # We still need permutations to mix dimensions between layers
        self.perms = [torch.randperm(dim) for _ in range(num_layers - 1)]

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Shuffle dimensions between layers
            if i < len(self.layers) - 1:
                x = x[:, self.perms[i]]
        return x

# --- Main Execution ---
if __name__ == "__main__":
    # Config
    NX, NY = 25, 25
    SAMPLES = 10000
    LAYERS = 6
    HIDDEN = 128
    EPOCHS = 50
    LR = 1e-3

    # 1. Get Dimension from A
    print(f"Generating Poisson Matrix ({NX}x{NY})...")
    A_scipy, b_scipy = poisson_gene(nx=NX, ny=NY)
    DIM = A_scipy.shape[0]

    # 2. Generate Training Data
    print(f"Generating {SAMPLES} training vectors...")
    v_train, w_train = gen_vec(dim=DIM, samples=SAMPLES, A=A_scipy)

    # 3. Setup Model
    model = PreconditionerINN(dim=DIM, num_layers=LAYERS, hidden_dim=HIDDEN)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(v_train, w_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # 4. Train
    print("Starting Training...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for v_batch, w_batch in dataloader:
            optimizer.zero_grad()
            
            # Forward: v -> predicted_w
            w_pred = model(v_batch)
            
            loss = loss_fn(w_pred, w_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.6f}")

    # 5. Save
    # save_path = f"inn_poisson_{NX}x{NY}.pth"
    # torch.save(model.state_dict(), save_path)
    # print(f"Model saved to {save_path}")