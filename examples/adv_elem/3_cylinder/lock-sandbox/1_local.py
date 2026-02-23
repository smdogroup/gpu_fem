import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.sparse as sp
sys.path.append("../src/")
from std_assembler import StandardCylinderAssembler
from elem import MITCShellElement

# sys.path.append("../../2_plate/src/")
# from smooth import TwodimSupportAddSchwarz

"""
goal of this method is to solve the locking-aware system approximately using local operators / jacobi smoothing with mat-mat products
    * in a way that I can implement on the GPU reasonably well.. and fast
"""


import os, json, hashlib
import scipy.sparse as sp

"""
The assembled kmat and rhs are cached if you run them multiple times to speedup debugging! for same element type!
Also caches some prolong matrices in multigrid
"""

def _wrap_assemble_with_cache(obj, cache_dir=".mg_cache"):
    raw = obj._assemble_system
    def cached():
        elem_key = "mitc" if args.elem.startswith("mitc") else args.elem  # mitc, mitc_lp, mitc_gp, mitc_ep -> mitc
        R = args.curvature

        key = hashlib.md5(json.dumps({
            "elem": elem_key, "nxe": int(obj.nxe),
            "E": 70e9, "nu": 0.3, "thick": float(args.thick),
            "L": float(1.0), "W": float(np.pi/2.0), "R": float(R),
            "clamped": bool(clamped),
        }, sort_keys=True).encode()).hexdigest()[:16]

        os.makedirs(cache_dir, exist_ok=True)
        Kp = f"{cache_dir}/{key}.K.npz"
        Fp = f"{cache_dir}/{key}.F.npy"
        if os.path.exists(Kp) and os.path.exists(Fp):
            print("LOADED from cache!")
            obj.kmat = sp.load_npz(Kp)
            obj.force = np.load(Fp)
        else:
            raw()
            sp.save_npz(Kp, obj.kmat)
            np.save(Fp, obj.force)
    obj._assemble_system = cached

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--elem", type=str, default='mitc_gp', help="--elem, options: mitcp is another good one")
# parser.add_argument("--nxe", type=int, default=32, help="number of elements")
parser.add_argument("--nxe", type=int, default=8, help="number of elements")
parser.add_argument("--thick", type=float, default=1e-3, help="number of elements")
parser.add_argument("--curvature", type=float, default=1.0, help="shell curvature (lower gets flatter)")
# parser.add_argument("--nxemin", type=int, default=16, help="min # elems multigrid")
# parser.add_argument("--coupled", type=int, default=2, help="size of coupling ASW blocks (options are 1 and 2), 1 is still an interesting vertex-edge coupling for DRIG")
# parser.add_argument("--solve", type=str, default='kmg', help="--solve : [direct, vmg, kmg]")
# parser.add_argument("--nsmooth", type=int, default=4, help="number of smoothing steps")
# parser.add_argument("--omega", type=float, default=1.0, help="omega smoother coeff (sometimes needs to be lower)")
# parser.add_argument("--smoother", type=str, default='supp_asw', help="--smooth : [gs, asw, supp_asw]")
# parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False, help="Plot matrices and residual")
# parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False, help="run debug codes")
# parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=False, help="verify defln with simple load")
args = parser.parse_args()

""" verify each beam element and solver type against truth """

elems = ['mitc', 'mitc_gp', 'mitc_lp', 'mitc_ep']
modes = ['standard', 'locking-global', 'locking-local', 'energy-jacobi']
prolong_mode = None
for i in range(4):
    if elems[i] == args.elem:
        prolong_mode = modes[i]

ELEMENT = MITCShellElement(
    prolong_mode=prolong_mode,
    # lam=1e-6, omega=0.5,
    lam=1e-2, omega=0.4,
    # lam=1e0, omega=0.4,
)

# ================================
# make plate assembler
# ================================

clamped = True


ASSEMBLER = StandardCylinderAssembler
f_assembler = ASSEMBLER(
    ELEMENT,
    nxe=args.nxe,
    E=70e9, nu=0.3, thick=args.thick,
    length=1.0,
    hoop_length=np.pi / 2.0,
    radius=1.0/args.curvature,
    load_fcn = lambda x, y, z : 1.0 * np.sin(np.pi * x) * np.sin(np.pi * z) * np.sin(np.pi * -y),
    clamped=clamped,
)
_wrap_assemble_with_cache(f_assembler)
f_assembler._assemble_system()
ELEMENT._xpts_cache[args.nxe] = f_assembler.xpts
ELEMENT._kmat_cache[args.nxe // 2] = f_assembler.kmat 
f_rhs = f_assembler.force.copy()
f_kmat = f_assembler.kmat.copy()


c_assembler = ASSEMBLER(
    ELEMENT,
    nxe=args.nxe // 2,
    E=70e9, nu=0.3, thick=args.thick,
    length=1.0,
    hoop_length=np.pi / 2.0,
    radius=1.0/args.curvature,
    load_fcn = lambda x, y, z : 1.0 * np.sin(np.pi * x) * np.sin(np.pi * z) * np.sin(np.pi * -y),
    clamped=clamped,
)
_wrap_assemble_with_cache(c_assembler)
c_assembler._assemble_system()
ELEMENT._xpts_cache[args.nxe // 2] = c_assembler.xpts
ELEMENT._kmat_cache[args.nxe // 4] = c_assembler.kmat

# # call restrict defect (part of multigrid process)
# c_defect = c_assembler.restrict_defect(f_rhs)

nxe = args.nxe
nxe_c = nxe // 2

P_standard = ELEMENT._build_P2_uncoupled3(nxe_c)
P_standard = ELEMENT._apply_bcs_to_P(P_standard, nxe_c)
ELEMENT._lock_P_cache = {} # reset

P_global = ELEMENT._locking_aware_prolong_global_mitc_v1(nxe_c, length=1.0)
ELEMENT._lock_P_cache = {} # reset

# # if omega too high it makes defects worse.. and soln worse..
# # P_local = ELEMENT._locking_aware_prolong_local_mitc_v2_jacobi(nxe_c, n_sweeps=10, omega=1.5)
# # P_local = ELEMENT._locking_aware_prolong_local_mitc_v2_jacobi(nxe_c, n_sweeps=10, omega=0.7)
# # P_local = ELEMENT._locking_aware_prolong_local_mitc_v2_jacobi(nxe_c, n_sweeps=10, omega=0.5)
# P_local = ELEMENT._locking_aware_prolong_local_mitc_v3_jacobi(nxe_c, n_sweeps=10, omega=0.5)
# ELEMENT._lock_P_cache = {} # reset

P_energy = ELEMENT._energy_smooth_jacobi_v1(nxe_c, n_sweeps=10, omega=0.5)

# ==========================================================================
# that jacobi solver doesn't match the global one like at all.. (esp BCs are wrong near boundary)
# write my own jacobi solver based on data from P_global stored in ELEMENT
# ==========================================================================

G_f = ELEMENT.G_f
G_c = ELEMENT.G_c
P_gam = ELEMENT.P_gam
LHS = ELEMENT.M
RHS = ELEMENT.RHS
free_cols_c = ELEMENT.free_cols_c
fixed_cols_c = ELEMENT.fixed_cols_c
solve_rows_f = ELEMENT.solve_rows_f
# P0_free = ELEMENT.P_0_free



# ===============================================================================
# END OF MY JACOBI SOLVER
# ===============================================================================


# now compare prolongations
c_soln = c_assembler.direct_solve()
nx_f = nxe + 1
dx_f = 1.0 / nxe


names = ['standard', 'global', 'energy']
P_mats = [P_standard, P_global, P_energy]

# names = ['standard', 'global', 'local', 'local2']
# P_mats = [P_standard, P_global, P_local, P_local_v2]

fig, ax = plt.subplots(3, 2, figsize=(12, 10), subplot_kw={'projection' : '3d'})

for i in range(3):
    name = names[i]
    P_mat = P_mats[i]

    f_soln = P_mat @ c_soln
    f_def = f_rhs - f_kmat @ f_soln

    soln_nrm = np.linalg.norm(f_soln)
    def_nrm = np.linalg.norm(f_def)
    method = name
    print(f"{method=} {soln_nrm=:.4e} {def_nrm=:.4e}")

    f_assembler.u = f_soln.copy()
    f_assembler.plot_disp(disp_mag=np.max(np.abs(f_soln)), ax=ax[i,0])

    f_assembler.u = f_def.copy()
    f_assembler.plot_disp(disp_mag=np.max(np.abs(f_def)), ax=ax[i,1])

plt.savefig("p_sandbox.png", dpi=400)
plt.show()



fig, ax = plt.subplots(3, 1, figsize=(12, 10)) #, subplot_kw={'projection' : '3d'})

for i in range(3):
    # print(f"{i=}")
    P_mat = P_mats[i].copy()
    mat = P_mat[:12,:12]

    if sp.isspmatrix_csr(mat):
        np_mat = mat.toarray()
    else:
        np_mat = mat
    
    # print(f"{mat.__dict__}")
    # np_mat = mat.toarray()
    # np_mat = mat.array()
    ax[i].imshow(np_mat)


plt.savefig("p_mat.png", dpi=400)
plt.show()