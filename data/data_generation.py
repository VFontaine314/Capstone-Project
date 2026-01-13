import numpy as np
from scipy.sparse import coo_matrix, diags
import ngsolve as ngs
from ngsolve.meshes import MakeQuadMesh

def poisson_gene(nx: int, ny: int):
    """
    Generate data for 2D Poisson problem.
    Ref: https://docu.ngsolve.org/latest/i-tutorials/wta/poisson.html

    Return matrix A and vector b.
    """
    # Create mesh in 2D unit domain
    mesh = MakeQuadMesh(nx=nx, ny=ny)
    # Create finite element space, zero Dirichlet boundary condition
    fes = ngs.H1(mesh, order=1, dirichlet=".*")
    # Define variational form and assemble
    u, v = fes.TrialFunction(), fes.TestFunction()
    a = ngs.BilinearForm(ngs.grad(u) * ngs.grad(v) * ngs.dx)
    f = ngs.LinearForm(v * ngs.dx)
    a.Assemble()
    f.Assemble()

    return _applyBC_export_data(a, f, fes)

def _applyBC_export_data(a, f, fes):
    """
    Apply boundary condition and export data.

    PS:
        - You do NOT have to modify this function.
        - This function can be reused for different problems.

    """
    # Export matrix in sparse format
    rows, cols, vals = a.mat.COO()
    A = coo_matrix((vals, (rows, cols))).tocsc()
    b = f.vec.FV().NumPy()
    A.indptr = A.indptr.astype(np.int64)
    A.indices = A.indices.astype(np.int64)

    # Set boundary conditions.
    boundary_dofs = np.nonzero(~fes.FreeDofs())[0]
    # print(f"Boundary / non-dof is {boundary_dofs}")
    mask = np.ones(A.shape[0])
    mask[boundary_dofs] = 0

    M = diags(mask)
    A = M @ A @ M
    diag_A = A.diagonal()
    diag_A[boundary_dofs] = 1
    A.setdiag(diag_A)
    A.eliminate_zeros()

    b[boundary_dofs] = 0

    return A, b

if __name__ == '__main__':
    A, b = poisson_gene(
        # nx=10,
        # ny=10
        nx=4,
        ny=4
    )

    print(f"{type(A) = }")
    print(f"{type(b) = }")
    print(f"{A.shape = }")
    print(f"{b.shape = }")

    np.set_printoptions(linewidth=200)
    print(f"{A.todense() = }")

    # Then, for example, export to a `.mat` file
    # ...
