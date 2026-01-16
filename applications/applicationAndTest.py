import torch
import time
import math
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.data_generation import poisson_gene
from models.model_supervised import train_supervised_inn
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
    epsilon=0.000001,
    max_iter=1000,
    random_state=0,
):
    torch.manual_seed(random_state)
    n, h = A.shape
    x_kplusone = torch.rand(n, dtype=A.dtype)
    x_k = torch.rand(n, dtype=A.dtype)
    i = 0
    mode = mode.lower()
    while (torch.dist(x_kplusone, x_k) > epsilon and i < max_iter):
        x_k = x_kplusone
        resid = A @ x_kplusone - b
        gamma = torch.rand(n)
        if mode == 'inn':
            resid_norm = (resid.unsqueeze(0) - v_mean) / v_std
            gamma = model.forward(resid_norm).squeeze(0)
        elif mode == 'nn':
            gamma = model.forward(resid)
        elif mode == 'jacobi':
            gamma = torch.diagonal(A) * resid
        elif mode == 'gauss':
            diag = torch.diagonal(A)
            lower = torch.diagonal(A, offset=-1)
            gamma = diag * resid
            # Use lower diagonal contribution where it exists.
            gamma[1:] += lower * resid[:-1]

        x_kplusone = x_k + gamma
        i += 1

    return x_kplusone, i


def comparing(n, num_epochs_INN, num_epochs_NN, model=None, random_state=0):
    nx = int(math.sqrt(n))
    ny = nx
    if nx * ny != n:
        nx = n
        ny = 1
    A_scipy, b_np = poisson_gene(nx=nx, ny=ny)
    A = torch.tensor(A_scipy.toarray(), dtype=torch.float64)
    b = torch.tensor(b_np, dtype=torch.float64)

    #put A and b in right format - start



    #put A and b in right format - stop

    stopwatch = Stopwatch()

    #note x, number of iterations and time of INN
    model, v_mean, v_std, _ = train_supervised_inn(
        nx=nx,
        ny=ny,
        epochs=num_epochs_INN,
    )
    stopwatch.start()
    x_INN, num_iter_INN = apply(A, b, 'INN', model, v_mean=v_mean, v_std=v_std)
    stopwatch.stop()
    time_INN = stopwatch.elapsed_time
    stopwatch.reset()

    #note x, number of iterations and time of NN
    #stopwatch.start
    #model = train_NN(A, num_epochs_NN)
    #x_NN, num_iter_NN = apply(A, b, 'NN', model)
    #stopwatch.stop
    #time_NN = stopwatch.elapsed_time
    #stopwatch.reset

    #note x, number of iterations and time of Jacobi preconditioner
    stopwatch.start()
    x_jacobi, num_iter_jacobi = apply(A, b, 'jacobi')
    stopwatch.stop()
    time_jacobi = stopwatch.elapsed_time
    stopwatch.reset()

    #note x, number of iterations and time of Gauss preconditioner
    stopwatch.start()
    x_gauss, num_iter_gauss = apply(A, b, 'gauss')
    stopwatch.stop()
    time_gauss = stopwatch.elapsed_time
    stopwatch.reset()

    print("comparison complete")
    print("STATS")
    print("*"*20)
    print("INN - Jacobi - Gauss")
    print(f"time: {time_INN:.4f} - {time_jacobi:.4f} - {time_gauss:.4f}")
    print(f"num_iter: {num_iter_INN} - {num_iter_jacobi} - {num_iter_gauss}")


if __name__ == "__main__":
    comparing(n=16, num_epochs_INN=10, num_epochs_NN=0)
    print("*" * 20)



