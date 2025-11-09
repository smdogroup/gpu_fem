import sys
sys.path.append("_src/") # contains nlplategpu
import nlplategpu # the so file created by pybind11

# standard python imports
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse

def get_load_factor_curves(nxe:int, LOAD_FACTORS:np.ndarray):

    # setup GPU solver for nl plate
    solver = nlplategpu.NonlinearPlateGPUSolver(
        pressure=8e6,
        omega=0.75,
        nxe=nxe,
        SR=100.0,
        use_predictor=False, # better for the NL plate case for some reason
        kmg_print=False,
        nl_debug=False, # if on exits on any lin solve failure to see state
    )

    nvars = solver.get_num_vars()
    u0 = np.zeros(nvars)

    lam0, inner_atol = 0.2, 1e-6

    lam = 1.0 # full loading
    u_lin = solver.kcycleSolve(u0, lam)
    # solver.writeSolution("out/plate_lin.vtk", u_lin)

    W_LIN = LOAD_FACTORS * np.max(np.abs(u_lin[2::6]))
    W_NL = np.zeros_like(LOAD_FACTORS)
    prev_lam = 0.0

    _soln = u0.copy()

    for i, lam in enumerate(LOAD_FACTORS):
        _lam0 = 0.5
        lam0 = _lam0 * prev_lam + (1.0-_lam0) * lam
        _soln = solver.continuationSolve(_soln, lam0, lam, inner_atol)
        prev_lam = lam

        W_NL[i] = np.max(np.abs(_soln[2::6]))

    # free up the solver
    print("try to free up the solver")
    solver.free()
    print("done freeing up the solver")

    return W_LIN, W_NL


if __name__ == "__main__":

    # run nonlinear GPU MG analyses for different mesh sizes
    # =============================================================

    NLOADS = 10
    LOAD_FACTORS = np.linspace(1.0 / NLOADS, 1.0, NLOADS)

    # NXE_LIST = [32, 64, 128, 256] #, 512]
    NXE_LIST = [64, 128] #, 512]
    W_LIN_DICT = {_nxe:[] for _nxe in NXE_LIST}
    W_NL_DICT = {_nxe:[] for _nxe in NXE_LIST}

    for NXE in NXE_LIST:
        w_lin, w_nl = get_load_factor_curves(NXE, LOAD_FACTORS)
        W_LIN_DICT[NXE] = w_lin #.copy()
        W_NL_DICT[NXE] = w_nl #.copy()


    # plot the load factors for each mesh size 
    # ==============================================

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


    for i, NXE in enumerate(NXE_LIST):
        plt.plot(LOAD_FACTORS, W_LIN_DICT[NXE], '--', color=colors[i], linewidth=2.5, alpha=0.8)
        plt.plot(LOAD_FACTORS, W_NL_DICT[NXE], 'o-', color=colors[i], label=f"{NXE=}", linewidth=2.5, alpha=0.8)

    plt.legend()

    plt.xlabel(r"$\lambda$" + ", load factor")
    plt.ylabel("Center Disp w" + r"$\ \times \ 1000$")
    plt.savefig("out/2_plate_lam_mesh_conv.svg")