import sys
sys.path.append("_src/") # contains nlwinggpu
import nlwinggpu # the so file created by pybind11

# standard python imports
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse

# RUN NOTES
# ==================

# L2 mesh => use CFI4 + predictor + omegaLS_min = 0.5 and omegaMC = 0.7 for best results (about 14 sec runtime)
# L3 mesh => haven't got it working yet, suspect need lower omegaLS_min bounds rn (or higher?) 

# setup GPU solver for nl plate
solver = nlwinggpu.NonlinearWingGPUSolver(
    # level=1,
    level=2,
    # level=3,
    force=4e7,
    SR=10.0,
    use_predictor=True,
    # use_predictor=False, # works well on NL wing now!
    # kmg_print=True,
    kmg_print=False,
    # nl_debug=True, # if debug is on, it will exit on a failure and return the failed state
    nl_debug=False,
    omegaMC=0.7,
    # omegaMC=1.0,
    # omegaMC=1.5,
    # omegaLS_min=0.25,
    omegaLS_min=0.5, 
    debug_gmg=False
)

nvars = solver.get_num_vars()
u0 = np.zeros(nvars)


# NONLINEAR solve
# =====================

start_solve = time.time()
lam0, lamf, inner_atol = 0.2, 1.0, 1e-2
u_nl = solver.continuationSolve(u0, lam0, lamf, inner_atol)
solve_dt = time.time() - start_solve
print(f"continuation solve in {solve_dt=:.4e} sec")
solver.writeSolution("out/wing_nl.vtk", u_nl)


# LINEAR solve
# =======================

lam = 1.0 # full loading
print("doing linear kcycle solve\n")
u_lin = solver.kcycleSolve(u0, lam)
print("done with kcycle solve\n")
solver.writeSolution("out/wing_lin.vtk", u_lin)
print("done with wing linear solve\n")


# NONLINEAR load factor plot vs linear
# ======================================

# RUN_PLOT_NL_LOAD_FACTOR = True
RUN_PLOT_NL_LOAD_FACTOR = False

if RUN_PLOT_NL_LOAD_FACTOR:

    NLOADS = 10
    LOAD_FACTORS = np.linspace(1.0 / NLOADS, 1.0, NLOADS)
    W_LIN = LOAD_FACTORS * np.max(np.abs(u_lin[2::6]))
    W_NL = np.zeros_like(LOAD_FACTORS)
    prev_lam = 0.0

    _soln = u0.copy()

    for i, lam in enumerate(LOAD_FACTORS):
        _lam0 = 0.5
        lam0 = _lam0 * prev_lam + (1.0-_lam0) * lam
        print(f"begin continuation solve in load factor loop")
        _soln = solver.continuationSolve(_soln, lam0, lam, inner_atol)
        prev_lam = lam

        W_NL[i] = np.max(np.abs(_soln[2::6]))

    # now plot the load factors of the plate
    plt.rcParams.update({
        # 'font.family': 'Courier New',  # monospace font
        'font.family' : 'monospace', # since Courier new not showing up?
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'figure.titlesize': 20
    })

    colors = ["#264653", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"]

    fig, ax = plt.subplots(figsize=(9, 7))

    factor = 1e3

    plt.plot(LOAD_FACTORS, W_LIN*factor, '--', color=colors[1], label="linear", linewidth=2.5)
    plt.plot(LOAD_FACTORS, W_NL*factor, 'o-', color=colors[4], label="nl", linewidth=2)
    plt.xlabel(r"$\lambda$" + ", load factor")
    plt.ylabel("Center Disp w" + r"$\ \times \ 1000$")
    plt.savefig("out/wing_lams.svg")


# FREE memory on GPU
# ===================

solver.free()