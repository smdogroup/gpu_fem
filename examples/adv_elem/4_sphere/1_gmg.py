import numpy as np
import sys
sys.path.append("../3_cylinder/src/")
# from elem import DeRhamIsogeometricCylinderElement #, MixedDeRhamIGACylinderElement
from elem import DeRhamMITC_IGACylinderElement


sys.path.append("src/")
from sdrig_assembler import DeRhamIGASphereAssembler
# from drig_assembler2 import MixedDeRhamIGACylinderAssembler

# MITC
from elem import MITCShellElement
from std_assembler import StandardCylinderAssembler

# sys.path.append("../2_plate/src/")
# from asw_derham import TwoDimAddSchwarzDeRhamVertexEdges
# from dasw_cyl import TwoDimAddSchwarzDeRhamCylinderVertexEdges, MixedTwoDimAddSchwarzDeRhamCylinderVertexEdges
from dasw_cyl import TwoDimAddSchwarzDeRhamCylinderVertexEdges

sys.path.append("../1_beam/src/")
from multigrid2 import vcycle_solve, VMG
from smoothers import BlockGaussSeidel, right_pcg2, right_pgmres2

sys.path.append("../2_plate/src/")
from smooth import TwodimSupportAddSchwarz

sys.path.append("../../asw/_py_demo/_src/")
from asw import TwodimAddSchwarz

import os, json, hashlib
import scipy.sparse as sp

"""
The assembled kmat and rhs are cached if you run them multiple times to speedup debugging! for same element type!
Also caches some prolong matrices in multigrid
"""


import argparse
parser = argparse.ArgumentParser()
# parser.add_argument("--elem", type=str, default='mitc_gp', help="--elem, options: tbd")
# parser.add_argument("--solve", type=str, default='kmg', help="--solve : [direct, vmg, kmg]")
# parser.add_argument("--plot", type=str, default=None, help="--plot is str : [w, u, v, thx, thy, thz] or None")
# parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True,
#                     help="Cache assembled kmat/force per MG level to disk") # --no-cache to turn off and reset
# parser.add_argument("--nxe", type=int, default=32, help="number of elements") # 32

# TEMP DEBUG drig element
parser.add_argument("--elem", type=str, default='drig', help="--elem, options: tbd")
parser.add_argument("--solve", type=str, default='kmg', help="--solve : [direct, vmg, kmg]")
parser.add_argument("--plot", type=str, default='w', help="--plot is str : [w, u, v, thx, thy, thz] or None")
parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True,
                    help="Cache assembled kmat/force per MG level to disk") # --no-cache to turn off and reset
parser.add_argument("--nxe", type=int, default=32, help="number of elements") # 32


# usual inputs
parser.add_argument("--nxemin", type=int, default=8, help="min # elems multigrid")
parser.add_argument("--coupled", type=int, default=2, help="size of coupling ASW blocks (options are 1 and 2), 1 is still an interesting vertex-edge coupling for DRIG")
parser.add_argument("--thick", type=float, default=1e-3, help="shell thickness")
# parser.add_argument("--radius", type=float, default=1.0, help="cylinder radius")
parser.add_argument("--curvature", type=float, default=1.0, help="shell curvature (lower gets flatter)")
parser.add_argument("--length", type=float, default=np.pi/2, help="cylinder length")
parser.add_argument("--width", type=float, default=1.0, help="cylinder width (hoop length)")
# parser.add_argument("--solve", type=str, default='direct', help="--solve : [direct, vmg, kmg]")
parser.add_argument("--nsmooth", type=int, default=2, help="number of smoothing steps")
parser.add_argument("--nprolong", type=int, default=10, help="number of smoothing steps for prolongator")
parser.add_argument("--omega", type=float, default=1.0, help="omega smoother coeff (sometimes needs to be lower)")
parser.add_argument("--smoother", type=str, default='supp_asw', help="--smooth : [gs, asw]")
# parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False, help="Plot matrices and residual")
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False, help="run debug codes")
parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=False, help="verify defln with simple load")
args = parser.parse_args()


def _wrap_assemble_with_cache(obj, cache_dir=".mg_cache"):
    raw = obj._assemble_system

    def cached():
        elem_key = "mitc" if args.elem.startswith("mitc") else args.elem  # mitc, mitc_lp, mitc_gp, mitc_ep -> mitc

        key = hashlib.md5(json.dumps({
            "elem": elem_key, "nxe": int(obj.nxe),
            "E": 70e9, "nu": 0.3, "thick": float(args.thick),
            "L": float(args.length), "W": float(args.width), "R": float(R),
            "clamped": bool(clamped),
        }, sort_keys=True).encode()).hexdigest()[:16]

        os.makedirs(cache_dir, exist_ok=True)
        Kp = f"{cache_dir}/{key}.K.npz"
        Fp = f"{cache_dir}/{key}.F.npy"
        if os.path.exists(Kp) and os.path.exists(Fp) and args.cache:
            print("LOADED kmat, F from cache!")
            obj.kmat = sp.load_npz(Kp)
            obj.force = np.load(Fp)
        else:
            raw()
            sp.save_npz(Kp, obj.kmat)
            np.save(Fp, obj.force)
    obj._assemble_system = cached


def _wrap_prolong_with_cache(obj, cache_dir=".mg_cache"):
    raw = obj._assemble_prolongation
    def cached():
        key = hashlib.md5(json.dumps({
            "elem": args.elem, "nxe": int(obj.nxe),
            "E": 70e9, "nu": 0.3, "thick": float(args.thick),
            "L": float(args.length), "W": float(args.width), "R": float(R),
            "clamped": bool(clamped), "nprolong" : int(args.nprolong),
        }, sort_keys=True).encode()).hexdigest()[:16]

        os.makedirs(cache_dir, exist_ok=True)
        Pp = f"{cache_dir}/{key}.P.npz"

        if 'mitc' in args.elem:
            if os.path.exists(Pp) and args.cache:
                print("LOADED P from cache!")
                obj.element._P_cache[obj.nxe] = sp.load_npz(Pp)
            else:
                raw()
                sp.save_npz(Pp, obj.element._P_cache[obj.nxe])
    obj._assemble_prolongation = cached

# t/R leads to potential membrane locking
R = 1.0 / args.curvature
# s = args.width
# theta = s / R # s = R * theta, rotation

# s = args.width
# phi = s / R
# y_max = np.abs(R * np.cos(phi))
# z_max = np.abs(R * np.sin(phi))

print(f"{R=}")

# L = args.length
clamped = True
# clamped = False

axial_factor = 0.0
# axial_factor = 0.3
# larger radius can lead to weird locking behavior


if args.elem == 'drig':
    ELEMENT = DeRhamMITC_IGACylinderElement(
        r=R, 
        curvature_on=True, 
        reduced_integrate_exy=True,
    )
    ASSEMBLER = DeRhamIGASphereAssembler
elif 'mitc' in args.elem:
    elems = ['mitc', 'mitc_gp', 'mitc_lp', 'mitc_ep']
    modes = ['standard', 'locking-global', 'locking-local', 'energy-jacobi']
    prolong_mode = None
    for i in range(4):
        if elems[i] == args.elem:
            prolong_mode = modes[i]

    ELEMENT = MITCShellElement(
        prolong_mode=prolong_mode,
        # lam=1e-6, omega=0.5,
        # lam=1e-2, omega=0.4,
        lam=1e0, omega=0.4,
        n_lock_sweeps=args.nprolong,
    )
    ASSEMBLER = StandardCylinderAssembler
    

import math
def load_fcn(_x, _y):
    _xhat = _x / args.length
    _yhat = _y / args.width
    diag = (args.length**2 + args.width**2)**0.5
    # theta = math.atan2(_yhat, _xhat)
    theta = math.atan2(_y, _x)
    diag_th = math.atan2(args.width, args.length)
    # diag_th = np.pi / 4.0 * 3
    r = np.sqrt(_xhat**2 + _yhat**2)
    r2 = r/diag
    # game of life polar load..
    # return 100.0 * np.sin(5.0  * np.pi * r) * np.cos(4*theta) * r2 * (1 - r2)
    return -100.0 * np.sin(4.0  * np.pi * r2) * np.cos(2.4*(theta-diag_th)) * r2 * (1 - r2)

# def load_fcn(x, y, A=-100.0, m=2.4):
#     xhat = x / args.length
#     yhat = y / args.width

#     # Coordinates along / across the diagonal in hat-space
#     s = 0.5 * (xhat + yhat)      # 0 at (0,0), 1 at (1,1)
#     t = 0.5 * (xhat - yhat)      # flips sign under swap -> use even function for symmetry

#     envelope = np.sin(np.pi * s)            # zero at s=0 and s=1 => (0,0) and (L,W)
#     pattern  = np.cos(m * np.pi * t)        # symmetric about diagonal since cos is even

#     return A * envelope * pattern

# standard assembler construction
assembler = ASSEMBLER(
    ELEMENT,
    nxe=args.nxe,
    E=70e9, nu=0.3, thick=args.thick,
    length=args.length,
    hoop_length=args.width,
    radius=R,
    clamped=clamped,
    load_fcn=load_fcn,
)

_wrap_assemble_with_cache(assembler)


if 'mg' in args.solve:
    # make V-cycle solver
    nxe = args.nxe
    nxe_min = args.nxemin
    if args.debug:
        nxe_min = nxe // 2
    grids = []
    smoothers = []
    # double_smooth = True
    double_smooth = False

    nsmooth = args.nsmooth
    while (nxe >= nxe_min):
        grid = ASSEMBLER(
            ELEMENT,
            nxe=nxe,
            E=70e9, nu=0.3, thick=args.thick,
            length=args.length,
            hoop_length=args.width,
            radius=R,
            clamped=clamped,
            load_fcn=load_fcn,
        )
        # grid._assemble_system()
        _wrap_assemble_with_cache(grid)
        _wrap_prolong_with_cache(grid)
        grid._assemble_system()


        if args.elem == 'mitc_ep':
            # add kmat into kmat cache for the element
            ELEMENT._kmat_cache[nxe // 2] = grid.kmat.copy()

        # register xpts to mitc element for multigrid prolongators
        if 'mitc' in args.elem:
            ELEMENT._xpts_cache[int(nxe)] = grid.xpts

        print(f"{nxe=} with {grid.force.shape=}")
        grids += [grid]
        if args.smoother == 'gs':
            smoother = BlockGaussSeidel.from_assembler(
                grid, omega=args.omega, iters=nsmooth
            )
        elif 'asw' in args.smoother:
            smoother = None

            if args.elem in ['drig', 'drigr']:
                omega = args.omega * 0.5 # need extra mult for them (default best values)
                coupled = args.coupled
                if coupled > 2:
                    print(f"{args.elem=} and {args.coupled=} dropped to 2")
                    coupled = 2
            else:
                omega = args.omega
                coupled = args.coupled

            if args.coupled == 1:
                omega = omega / 2.0 # because some 2x smoothing on thx, thy
                patch_type = "vertex_edges"
            elif args.coupled == 2:
                omega = omega / 4.0 # because 2x smoothing than coupled == 1 schwarz (so ~4x smoothing on thx, thy)
                patch_type = "wblock_vertex_edges"
            elif args.coupled == 3:
                omega = omega / 8.0

            if 'drig' in args.elem:
                print("using Additive schwarz DeRham smoother")
                if args.elem == 'drig':
                    ASW_CLASS = TwoDimAddSchwarzDeRhamCylinderVertexEdges
                # elif args.elem == 'mdrig':
                #     ASW_CLASS = MixedTwoDimAddSchwarzDeRhamCylinderVertexEdges 

                smoother = ASW_CLASS.from_assembler(
                    grid, omega=omega, iters=nsmooth,
                    # patch_type = patch_type,
                    # patch_type="vertex_edges", # one w vertex and nearby 4 edges (2 of thx and 2 of thy)
                    patch_type="wblock_vertex_edges",
                )

            elif args.coupled == 3 and args.smoother == 'supp_asw':
                smoother = TwodimSupportAddSchwarz.from_assembler(
                    grid, omega=omega, iters=nsmooth, #, coupled_size=args.coupled
                )

            else:
                smoother = TwodimAddSchwarz.from_assembler(
                    grid, omega=omega, iters=nsmooth, coupled_size=2
                )
        smoothers += [smoother]
        nxe = nxe // 2
        if double_smooth:
            nsmooth *= 2

    ngrids = len(grids)
    for i in range(ngrids-1):
        grids[i]._assemble_prolongation()
        # cause need all xpts cache to have been registered


# ============================
# linear solve
# ============================

if args.solve == 'direct':
    assembler.direct_solve()
elif args.solve == 'vmg':

    DEVEL_DEBUG = args.debug
    line_search = args.elem in ['drig', 'drigr', 'mitc_lp', 'mitc_gp', 'mitc']

    if DEVEL_DEBUG:
        assembler.u, ncyc = vcycle_solve(grids, pre_smooth=args.nsmooth, post_smooth=args.nsmooth,
                                        # line_search=False, # often need it turned off.. for best conv
                                        # line_search = 'drig' in args.elem,
                                        line_search=line_search,
                                        debug=True,
                                        nvcycles=1000,
                                        rtol=1e-6,
                                        smoothers=smoothers)
    else:
        assembler.u, ncyc = vcycle_solve(grids, pre_smooth=args.nsmooth, post_smooth=args.nsmooth,
                                        # line_search=False, # often need it turned off.. for best conv
                                        # line_search = 'drig' in args.elem,
                                        # line_search=False,
                                        line_search=line_search,
                                        debug=args.debug,
                                        nvcycles=400,
                                        rtol=1e-6,
                                        smoothers=smoothers)

elif args.solve == 'kmg':


    line_search = args.elem in ['drig', 'drigr', 'mitc_lp', 'mitc_gp', 'mitc']
    # line_search = args.elem in ['drig', 'drigr', 'mitc_lp', 'mitc_gp']

    vmg2 = VMG(
        grids, nsmooth=args.nsmooth, 
        ncyc=1, # fewer total v-cycles often..
        # ncyc=2,
        smoothers=smoothers, 
        # line_search = 'drig' in args.elem,
        line_search=line_search,
    )
    pc = vmg2
    assembler._assemble_system()

    # assembler.u, nsteps = right_pgmres2(
    #     A=assembler.kmat, b=assembler.force,
    #     restart=100, M=pc, #M=vmg,
    #     rtol=1e-6,
    # )

    assembler.u, nsteps = right_pcg2(
        A=assembler.kmat, b=assembler.force,
        M=pc, rtol=1e-6, atol=1e-20,
        max_iter=200,
        print_freq=3
    )

    total_vcyc = vmg2.total_vcycles
    print(f"{total_vcyc=}")



# ===============================
# PLOT
# ===============================

# plot is w, u, v, thx, thy

def norm(x):
    return np.max(np.abs(x))

if 'mitc' in args.elem:
    # get max deflection
    disp = assembler.u.copy()
    u = disp[0::6]
    v = disp[1::6]
    w = disp[2::6]

    x = np.arange(assembler.nnx) * assembler.dx
    s = np.arange(assembler.nnx) * assembler.dy
    # s = np.linspace(-self.Ly, 0.0, self.nnx)
    phi = s / assembler.radius

    ix_vec = np.array([inode % assembler.nnx for inode in range(assembler.nnodes)])
    iy_vec = np.array([inode // assembler.nnx for inode in range(assembler.nnodes)])
    x_flat = ix_vec * assembler.dx
    s_flat = iy_vec * assembler.dy
    phi_flat = s_flat / assembler.radius

    # plot normal deflection v * yhat + w * zhat
    # y_flat = -assembler.radius * np.cos(phi_flat)
    # z_flat = assembler.radius * np.sin(phi_flat)
    # yhat = y_flat / assembler.radius; zhat = z_flat / assembler.radius
    xpts = assembler.xpts
    yhat = xpts[1::3]
    zhat = xpts[2::3]

    normal_defln = v * yhat + w * zhat # true normal deflection

    u_nrm = norm(u)
    v_nrm = norm(v)
    w_nrm = norm(w)
    disp_nrm = np.max([v_nrm, w_nrm])
    norm_nrm = norm(normal_defln)

elif args.elem == 'drig':
    u = assembler.u.copy()
    off_w = assembler.off_w; nw = assembler.nw
    normal_defln = u[off_w:off_w + nw]
    disp_nrm = norm(normal_defln)
    norm_nrm = norm(normal_defln)

nxe = args.nxe
print(f"{nxe=} {disp_nrm=:.4e} {norm_nrm=:.4e}")
# print(f"\n\t{u_nrm=:.4e}\n\t{v_nrm=:.4e}\n\t{w_nrm=:.4e}")


# DEBUG to plot forces
# assembler.u = assembler.force * 1.0

if args.plot is not None:

    assembler.plot_disp(
        # disp_mag=0.0,
        disp_mag = 0.05 * R,
        # disp_mag=0.2 * R,
        # disp_mag=0.3 * R,
        # disp_mag=0.4 * R,
        # disp_mag=1.0, # same as radius (as multiple of inf-norm or max value)
        # disp_mag=2.0,
        # disp_mag=5.0,
        mode=args.plot,
    )    
