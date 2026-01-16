import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.data_generation import poisson_gene, gen_vec

import time
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


def apply(A, b, mode, model=None, epsilon=0.000001, max_iter=1000, random_state=0):
    torch.manual_seed(random_state)
    n, h = A.shape
    x_kplusone = torch.rand(n)
    x_k = torch.rand(n)
    i = 0
    while (torch.dist(x_kplusone, x_k) > epsilon and i < max_iter):
        x_k = x_kplusone
        resid = A * x_kplusone - b
        gamma = torch.rand(n)
        if mode == 'INN':
            gamma = model.forward(resid)
        elif mode == 'NN':
            gamma = model.forward(resid)
        elif mode == 'Jacobi':
            gamma = torch.diagonal(A) * resid
        elif mode == 'gauss':
            gamma = torch.diagonal(A, offset=-1) * torch.diagonal(A) * resid

        x_kplusone = x_k + gamma
        i += 1

    return x_kplusone, i


def comparing(n, num_epochs_INN, num_epochs_NN, model, random_state=0):

    #put A and b in right format - start
    print(f"Generating Poisson Matrix ({n}x{n})...")
    A, b = poisson_gene(nx=n, ny=n)


    #put A and b in right format - stop

    stopwatch = Stopwatch()

    #note x, number of iterations and time of INN
    stopwatch.start
    x_INN, num_iter_INN = apply(A, b, 'INN', model)
    stopwatch.stop
    time_INN = stopwatch.elapsed_time
    stopwatch.reset

    #note x, number of iterations and time of NN
    #stopwatch.start
    #model = train_NN(A, num_epochs_NN)
    #x_NN, num_iter_NN = apply(A, b, 'NN', model)
    #stopwatch.stop
    #time_NN = stopwatch.elapsed_time
    #stopwatch.reset

    #note x, number of iterations and time of Jacobi preconditioner
    stopwatch.start
    x_jacobi, num_iter_jacobi = apply(A, b, 'jacobi')
    stopwatch.stop
    time_jacobi = stopwatch.elapsed_time
    stopwatch.reset

    #note x, number of iterations and time of Gauss preconditioner
    stopwatch.start
    x_gauss, num_iter_gauss = apply(A, b, model, 'gauss')
    stopwatch.stop
    time_gauss = stopwatch.elapsed_time
    stopwatch.reset

    print("comparison complete")
    print("STATS")
    print("*"*20)
    print("INN - Jacobi - Gauss")
    print("time: " + time_INN + " - " + time_jacobi + " - " + time_gauss)
    print("num_iter: " + num_iter_INN + " - " + num_iter_jacobi + " - " + num_iter_gauss)
    print("*" * 20)


if __name__ == "__main__":
    n = 4
    print(f"Generating Poisson Matrix ({n}x{n})...")
    A, b = poisson_gene(nx=n, ny=n)
    x, i = apply(A, b, mode = "Jacobi", model=None, epsilon=0.000001, max_iter=1000, random_state=0)
    print(x)

