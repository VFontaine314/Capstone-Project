import torch
import normflows as nf
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

dim = 4

# Random linear transformation
A = torch.rand(dim, dim, device=device)

# Base distribution
base = nf.distributions.base.DiagGaussian(dim)

# Build flow
flows = []
for _ in range(10):
    param_map = nf.nets.MLP([dim//2, 100, 100, dim//2*2])
    flows.append(nf.flows.AffineCouplingBlock(param_map))
    flows.append(nf.flows.Permute(dim, mode='swap'))

model = nf.NormalizingFlow(base, flows).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr
=1e-4)
mse = nn.MSELoss()

# Training loop
epoch_size = 32
n_epochs = 2000
losses = []
for epoch in range(n_epochs):
    x = torch.rand(epoch_size, dim, device=device)
    b = x @ A.T

    optimizer.zero_grad()

    # Forward → latent
    b_pred = model.forward(x)
    # Inverse → predicted reconstruction

    # Compute MSE
    loss = mse(b_pred, b)
    losses.append(loss.item())

    
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, MSE Loss: {loss.item():.6f}")



# richardson method

n_iter = 100
x = torch.rand(1, dim, device=device)
b = x @ A.T
x_pred = torch.zeros(1, dim, device=device)
for i in range(n_iter):
    x_pred += model.inverse(b-x_pred @ A.T)
    print(x_pred)
    if torch.allclose(x, x_pred):
        break

print(x)
print(x_pred)

