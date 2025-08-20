import numpy as np
import matplotlib.pyplot as plt

# Define parameters
c = 1  # Advection speed
len = 2.5
dx = 0.1
t = 1

num_points = int((2 * len) / dx) + 1  # Number of grid points

x = np.linspace(-len, len, num_points)
n = len(x)

# Initial condition: u(x, t=0)
u = np.array([1 - abs(xi) if abs(xi) <= 1 else 0 for xi in x])

# Boundary Conditions
u[0] = 2.5
u[-1] = 2.5


# Construct the tridiagonal system
A = np.zeros((n, n))
C = np.copy(u)  # Right-hand side vector


def thomas_algorithm(b, d, a, c):
    """
    Solves Ax = d where A is a tridiagonal matrix with:
    - a: lower diagonal (length n-1)
    - b: main diagonal (length n)
    - c: upper diagonal (length n-1)
    - d: right-hand side (length n)
    """
    n = len(d)

    for t in range(time_step):


    # Step 1: Forward elimination
    for i in range(1, n):
        w = a[i - 1] / b[i - 1]
        b[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]

    # Step 2: Back substitution
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]  # Last variable

    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x


for nu in np.linspace(0.1, 1, 10):

    dt = dx*nu/c

    for i in range(1, n - 1):
        A[i, i - 1] = b
        A[i, i] = d
        A[i, i + 1] = a

    # Apply boundary conditions in the matrix
    A[0, 0] = A[-1, -1] = 1  # Dirichlet BCs
    C[0] = u[0]
    C[-1] = u[-1]

    # Tridiagonal matrix coefficients
    b = -nu / 2  # Lower diagonal alpha
    d = 1  # Main diagonal beta
    a = nu / 2  # Upper diagonal gamma

    # Extract diagonals for Thomas algorithm
    lower_diag = [b] * (n - 1)  # Below main diagonal
    main_diag = [d] * n  # Main diagonal
    upper_diag = [a] * (n - 1)  # Above main diagonal

    time_step = int((t / dt) + 1)

    # Solve for new u using the Thomas algorithm

    u_new = thomas_algorithm(lower_diag, main_diag, upper_diag, C)

    # Plot results
    plt.plot(x, u, label="Initial Condition")
    plt.plot(x, u_new, label="After One Step", linestyle="dashed")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.legend()
    plt.grid()
    plt.title("Solution Using Thomas Algorithm")
    plt.show()
