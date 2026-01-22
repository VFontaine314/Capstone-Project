import torch
import time
import math
import os
import sys
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.data_generation import poisson_gene, diffusion_gene, heat_gene
from models.model_supervised import train_supervised_inn
from models.model_unsupervised import train_unsupervised_inn
from models.mlp import train_mlp
class Stopwatch:
    def __init__(self):
        self.start_time = None
        self.elapsed_time = 0
        self.running = False

    def start(self):
        """Start de stopwatch."""
        if not self.running:
            self.start_time = time.time() - self.elapsed_time
            self.running = True
        else:
            print("Stopwatch is al gestart.")

    def stop(self):
        """Stop de stopwatch."""
        if self.running:
            self.elapsed_time = time.time() - self.start_time
            self.running = False
        else:
            print("Stopwatch is niet actief.")

    def reset(self):
        """Reset de stopwatch."""
        self.start_time = None
        self.elapsed_time = 0
        self.running = False

    def time(self):
        """Print de huidige tijd van de stopwatch."""
        if self.running:
            current_time = time.time() - self.start_time
        else:
            current_time = self.elapsed_time
        print(f"Tijd: {current_time:.2f} seconden")


def apply(
    A,
    b,
    mode,
    model=None,
    v_mean=None,
    v_std=None,
    epsilon=0.001,
    max_iter=10,
    random_state=0,
    learned_step=0.5,
):
    torch.manual_seed(random_state)
    n, h = A.shape
    x_kplusone = torch.rand(n, dtype=A.dtype)
    x_k = torch.rand(n, dtype=A.dtype)
    i = 0
    resid = A @ x_kplusone - b
    mode = mode.lower()
    norm_resids = []
    while ((torch.dist(x_kplusone, x_k) > epsilon or torch.norm(resid) > epsilon) and i < max_iter):
        x_k = x_kplusone
        resid = A @ x_kplusone - b
        norm_resids.append(torch.norm(resid).item())
        gamma = torch.rand(n)
        if mode == 'inn' or mode == 'mlp':
            resid_norm = (resid.unsqueeze(0) - v_mean) / v_std
            gamma = model.forward(resid_norm).squeeze(0)
            # Dampen learned update to avoid overshooting on out-of-distribution residuals.
            scale = torch.norm(resid) / (torch.norm(gamma) + 1e-12)
            gamma = gamma * min(learned_step, scale.item())
        elif mode == 'nn':
            gamma = model.forward(resid)
        elif mode == 'jacobi':
            diag = torch.diagonal(A)
            gamma = resid / diag
        elif mode == 'gauss':
            diag = torch.diagonal(A)
            lower = torch.diagonal(A, offset=-1)
            gamma = resid / diag
            # Simple forward correction using the lower diagonal.
            gamma[1:] += (lower / diag[1:]) * gamma[:-1]
        elif mode == 'no_p':
            gamma = 0.5 * resid

        x_kplusone = x_k - gamma
        i += 1

    return x_kplusone, i, norm_resids


def comparing(n, num_epochs_INN, num_epochs_NN, model=None, random_state=0):
    nx = n
    ny = nx
    #if nx * ny != n:
    #    nx = n
    #    ny = 1
    A_scipy, b_np = poisson_gene(nx=nx, ny=ny)
    A = torch.tensor(A_scipy.toarray(), dtype=torch.float64)
    b = torch.tensor(b_np, dtype=torch.float64)

    #put A and b in right format - start



    #put A and b in right format - stop

    stopwatch = Stopwatch()

    #note x, number of iterations and time of INN

    model_sup, v_mean_sup, v_std_sup, _, _ = train_supervised_inn(
        nx=nx,
        ny=ny,
        epochs=num_epochs_INN,
    )
    stopwatch.start()

    x_INN_sup, num_iter_INN_sup, norm_resids_sup = apply(A, b, 'INN', model_sup, v_mean=v_mean_sup, v_std=v_std_sup)

    stopwatch.stop()
    time_INN_sup = stopwatch.elapsed_time
    stopwatch.reset()



    model_unsup, v_mean_unsup, v_std_unsup, _, _ = train_unsupervised_inn(
        nx=nx,
        ny=ny,
        epochs=num_epochs_INN,
    )
    stopwatch.start()
    x_INN_unsup, num_iter_INN_unsup, norm_resids_unsup = apply(A, b, 'INN', model_unsup, v_mean=v_mean_unsup, v_std=v_std_unsup)
    stopwatch.stop()
    time_INN_unsup = stopwatch.elapsed_time
    stopwatch.reset()

    model_mlp, v_mean_mlp, v_std_mlp, _ = train_mlp(
        nx=nx,
        ny=ny,
        epochs=num_epochs_NN,
    )
    stopwatch.start()
    x_MLP, num_iter_MLP, norm_resids_MLP = apply(
        A, b, 'mlp', model_mlp, v_mean=v_mean_mlp, v_std=v_std_mlp
    )
    stopwatch.stop()
    time_MLP = stopwatch.elapsed_time
    stopwatch.reset()

    #note x, number of iterations and time of Jacobi preconditioner
    stopwatch.start()
    x_jacobi, num_iter_jacobi, norm_resids_jacobi = apply(A, b, 'jacobi')
    stopwatch.stop()
    time_jacobi = stopwatch.elapsed_time
    stopwatch.reset()

    #note x, number of iterations and time of Gauss preconditioner
    stopwatch.start()
    x_gauss, num_iter_gauss, norm_resids_gauss = apply(A, b, 'gauss')
    stopwatch.stop()
    time_gauss = stopwatch.elapsed_time
    stopwatch.reset()

    #note x, number of iterations and time of Gauss preconditioner
    stopwatch.start()
    x_no_p, num_iter_no_p, norm_resids_no_p = apply(A, b, 'no_p')
    stopwatch.stop()
    time_no_p = stopwatch.elapsed_time
    stopwatch.reset()

    # Print numerical results
    print("comparison complete")
    print("STATS")
    print("*"*20)
    print("model: supervised INN - unsupervised INN - MLP - Jacobi - Gauss - no P")
    print(f"time: {time_INN_sup:.4f} - {time_INN_unsup:.4f} - {time_MLP:.4f} - {time_jacobi:.4f} - {time_gauss:.4f} - {time_no_p:.4f}")
    print(f"num_iter: {num_iter_INN_sup} - {num_iter_INN_unsup} - {num_iter_MLP} - {num_iter_jacobi} - {num_iter_gauss} - {num_iter_no_p}")
    print("*" * 20)

    torch.set_printoptions(precision=4, sci_mode=False)
    print(x_INN_sup.detach())
    print(x_INN_unsup.detach())
    print(x_jacobi.detach())
    print(x_gauss.detach())

    # Plot convergence results
    plt.plot(norm_resids_sup, label="Supervised INN")
    plt.plot(norm_resids_unsup, label="Unsupervised INN")
    plt.plot(norm_resids_MLP, label="MLP")
    plt.plot(norm_resids_jacobi, label="Jacobi")
    plt.plot(norm_resids_gauss, label="Gauss")
    plt.plot(norm_resids_no_p, label="No Preconditioner")
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.title("Convergence Graph")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    comparing(n=11, num_epochs_INN=10, num_epochs_NN=10)
