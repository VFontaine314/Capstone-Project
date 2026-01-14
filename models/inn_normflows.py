import torch
import normflows as nf
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
dim = 4

A = torch.rand(dim, dim, device=device)
# Invertible neural network fθ ≈ A
flows = []
for _ in range(4):
    param_map = nf.nets.MLP(
        [dim // 2, 64, 64, 2 * (dim - dim // 2)]
    )
    flows.append(nf.flows.AffineCouplingBlock(param_map))
    flows.append(nf.flows.Permute(dim, mode="swap"))

base = nf.distributions.base.DiagGaussian(dim)
inn = nf.NormalizingFlow(base, flows).to(device)

optimizer = torch.optim.Adam(inn.parameters(), lr=1e-4)
mse = nn.MSELoss()

for epoch in range(5000):
    x = torch.rand(256, dim, device=device)
    y = x @ A.T   # Ax

    y_pred = inn.forward(x)
    loss = mse(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}, MSE: {loss.item():.3e}")

# Ground truth
x_true = torch.rand(1, dim, device=device)
b = x_true @ A.T

x = torch.zeros_like(x_true)

for k in range(100):
    r = b - x @ A.T              # residual
    dx = inn.inverse(r)       # ≈ A^{-1} r

    x = x + 0.5 * dx             # damping is important

    err = torch.norm(x - x_true)

print(x_true)
print(x)
