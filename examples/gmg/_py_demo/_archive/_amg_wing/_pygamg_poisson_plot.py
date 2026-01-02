import numpy as np
import scipy.sparse.linalg as spla
import pyamg
import time
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import reverse_cuthill_mckee

# Log-spaced grid sizes from 20 to 500
n_values = np.unique(np.logspace(np.log10(20), np.log10(3e3), num=10, dtype=int))

solve_times_ilu = []
solve_times_amg = []
solve_times_direct = []

for n in n_values:
    print(f"\nRunning for n = {n} (A is {n**2} x {n**2})")
    A = pyamg.gallery.poisson((n, n), format='csr')
    b = np.ones(A.shape[0])

    # AMG-preconditioned GMRES
    try:
        ml = pyamg.smoothed_aggregation_solver(A)
        M_amg = ml.aspreconditioner()
        start = time.time()
        x_amg, _ = spla.gmres(A, b, M=M_amg, tol=1e-8)
        amg_time = time.time() - start
    except Exception as e:
        print(f"AMG failed: {e}")
        amg_time = np.nan

    # ILU-preconditioned GMRES, only for n ≤ 300
    if n <= 300:
        try:
            ilu = spla.spilu(A, drop_tol=1e-4, fill_factor=10.0)
            M_ilu = spla.LinearOperator(A.shape, ilu.solve)
            start = time.time()
            x_ilu, _ = spla.gmres(A, b, M=M_ilu, tol=1e-8)
            ilu_time = time.time() - start
        except Exception as e:
            print(f"ILU failed: {e}")
            ilu_time = np.nan
    else:
        ilu_time = np.nan

    # Direct solve
    try:
        start = time.time()
        x_direct = spla.spsolve(A, b)
        direct_time = time.time() - start
    except Exception as e:
        print(f"Direct solve failed: {e}")
        direct_time = np.nan

    solve_times_ilu.append(ilu_time)
    solve_times_amg.append(amg_time)
    solve_times_direct.append(direct_time)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(n_values**2, solve_times_ilu, 'o-', label='ILU-GMRES', linewidth=2)
plt.plot(n_values**2, solve_times_amg, 's-', label='AMG-GMRES', linewidth=2)
plt.plot(n_values**2, solve_times_direct, 'd-', label='Direct Solve (spsolve)', linewidth=2)
plt.xlabel('Num DOF N')
plt.ylabel('Solve time (seconds)')
plt.yscale('log')
plt.title('Solve Time vs Grid Size for Poisson Problem')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xscale('log')
# plt.show()
plt.savefig("out/solve_times_cpu.png", dpi=400)
