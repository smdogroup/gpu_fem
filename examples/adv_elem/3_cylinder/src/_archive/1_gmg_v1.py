import numpy as np
import sys
sys.path.append("src/")
from elem import DeRhamIsogeometricCylinderElement, MixedDeRhamIGACylinderElement
from drig_assembler import DeRhamIGACylinderAssembler
from drig_assembler2 import MixedDeRhamIGACylinderAssembler

# MITC
from elem import MITCShellElement
from std_assembler import StandardCylinderAssembler

# sys.path.append("../2_plate/src/")
# from asw_derham import TwoDimAddSchwarzDeRhamVertexEdges
from dasw_cyl import TwoDimAddSchwarzDeRhamCylinderVertexEdges, MixedTwoDimAddSchwarzDeRhamCylinderVertexEdges

sys.path.append("../1_beam/src/")
from multigrid2 import vcycle_solve, VMG
from smoothers import BlockGaussSeidel, right_pcg2, right_pgmres2

sys.path.append("../../asw/_py_demo/_src/")
from asw import TwodimAddSchwarz

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--elem", type=str, default='mitc', help="--elem, options: tbd")
parser.add_argument("--nxe", type=int, default=8, help="number of elements") # 32
parser.add_argument("--nxemin", type=int, default=4, help="min # elems multigrid")
parser.add_argument("--coupled", type=int, default=2, help="size of coupling ASW blocks (options are 1 and 2), 1 is still an interesting vertex-edge coupling for DRIG")
parser.add_argument("--thick", type=float, default=1e-2, help="shell thickness")
# parser.add_argument("--radius", type=float, default=1.0, help="cylinder radius")
parser.add_argument("--curvature", type=float, default=1.0, help="shell curvature (lower gets flatter)")
parser.add_argument("--length", type=float, default=1.0, help="cylinder length")
parser.add_argument("--width", type=float, default=np.pi / 2.0, help="cylinder width (hoop length)")
parser.add_argument("--solve", type=str, default='direct', help="--solve : [direct, vmg, kmg]")
# parser.add_argument("--solve", type=str, default='direct', help="--solve : [direct, vmg, kmg]")
parser.add_argument("--nsmooth", type=int, default=2, help="number of smoothing steps")
parser.add_argument("--omega", type=float, default=0.7, help="omega smoother coeff (sometimes needs to be lower)")
parser.add_argument("--smoother", type=str, default='asw', help="--smooth : [gs, asw]")
parser.add_argument("--plot", type=str, default=None, help="--plot is str : [w, u, v, thx, thy, thz] or None")
# parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False, help="Plot matrices and residual")
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False, help="run debug codes")
parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=False, help="verify defln with simple load")
args = parser.parse_args()

# t/R leads to potential membrane locking
R = 1.0 / args.curvature
# s = args.width
# theta = s / R # s = R * theta, rotation

# s = args.width
# phi = s / R
# y_max = np.abs(R * np.cos(phi))
# z_max = np.abs(R * np.sin(phi))

print(f"{R=}")

L = args.length
clamped = True
# clamped = False

axial_factor = 0.0
# axial_factor = 0.3
# larger radius can lead to weird locking behavior


if args.elem == 'drig':
    ELEMENT = DeRhamIsogeometricCylinderElement(
        r=R, axial_factor=axial_factor,
        curvature_on=True, # curvature terms lead to mem locking
        # curvature_on=False,
    )
    ASSEMBLER = DeRhamIGACylinderAssembler
elif args.elem == 'mdrig':
    ELEMENT = MixedDeRhamIGACylinderElement(
        r=R, axial_factor=axial_factor,
        curvature_on=True, # curvature terms lead to mem locking
        # curvature_on=False,
    )
    ASSEMBLER = MixedDeRhamIGACylinderAssembler
elif args.elem == 'mitc':
    ELEMENT = MITCShellElement()
    ASSEMBLER = StandardCylinderAssembler
    

# standard assembler construction
assembler = ASSEMBLER(
    ELEMENT,
    nxe=args.nxe,
    E=70e9, nu=0.3, thick=args.thick,
    length=L,
    # hoop_length=np.pi, # half-cylinder
    # hoop_length=np.pi*0.5*R, # quarter-cylinder
    hoop_length=args.width,
    radius=R,
    # load_fcn = lambda x,s : 1.0,
    # load_fcn = lambda x, y, z : 1.0, # so outwards deflection
    load_fcn = lambda x, y, z : 1.0 * np.sin(np.pi * x) * np.sin(np.pi * z) * np.sin(np.pi * -y),
    clamped=clamped,
)



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
            length=L,
            hoop_length=args.width,
            radius=R,
            # load_fcn = lambda x,s : 1.0,
            # load_fcn = lambda x, y, z : 1.0, # so outwards deflection
            load_fcn = lambda x, y, z : 1.0 * np.sin(np.pi * x) * np.sin(np.pi * z) * np.sin(np.pi * -y),
            clamped=clamped,
        )
        # grid._assemble_system()

        # register xpts to mitc element for multigrid prolongators
        if 'mitc' in args.elem:
            ELEMENT._xpts_cache[nxe] = grid.xpts

        print(f"{nxe=} with {grid.force.shape=}")
        grids += [grid]
        if args.smoother == 'gs':
            smoother = BlockGaussSeidel.from_assembler(
                grid, omega=args.omega, iters=nsmooth
            )
        elif args.smoother == 'asw':
            smoother = None
            if args.coupled == 1:
                omega = args.omega / 2.0 # because some 2x smoothing on thx, thy
                patch_type = "vertex_edges"
            elif args.coupled == 2:
                omega = args.omega / 4.0 # because 2x smoothing than coupled == 1 schwarz (so ~4x smoothing on thx, thy)
                patch_type = "wblock_vertex_edges"

            if 'drig' in args.elem:
                print("using Additive schwarz DeRham smoother")
                if args.elem == 'drig':
                    ASW_CLASS = TwoDimAddSchwarzDeRhamCylinderVertexEdges
                elif args.elem == 'mdrig':
                    ASW_CLASS = MixedTwoDimAddSchwarzDeRhamCylinderVertexEdges 

                smoother = ASW_CLASS.from_assembler(
                    grid, omega=omega, iters=nsmooth,
                    patch_type = patch_type,
                    # patch_type="vertex_edges", # one w vertex and nearby 4 edges (2 of thx and 2 of thy)
                    # patch_type="wblock_vertex_edges",
                )
            else:
                smoother = TwodimAddSchwarz.from_assembler(
                    grid, omega=omega, iters=nsmooth, coupled_size=2
                )
        smoothers += [smoother]
        nxe = nxe // 2
        if double_smooth:
            nsmooth *= 2


# ============================
# linear solve
# ============================

if args.solve == 'direct':
    assembler.direct_solve()
elif args.solve == 'vmg':

    DEVEL_DEBUG = args.debug

    if DEVEL_DEBUG:
        assembler.u, ncyc = vcycle_solve(grids, pre_smooth=args.nsmooth, post_smooth=args.nsmooth,
                                        # line_search=False, # often need it turned off.. for best conv
                                        line_search = 'drig' in args.elem,
                                        debug=True,
                                        nvcycles=1000,
                                        rtol=1e-6,
                                        smoothers=smoothers)
    else:
        assembler.u, ncyc = vcycle_solve(grids, pre_smooth=args.nsmooth, post_smooth=args.nsmooth,
                                        # line_search=False, # often need it turned off.. for best conv
                                        line_search = 'drig' in args.elem,
                                        # line_search=False,
                                        debug=args.debug,
                                        nvcycles=400,
                                        rtol=1e-6,
                                        smoothers=smoothers)

elif args.solve == 'kmg':

    vmg2 = VMG(
        grids, nsmooth=args.nsmooth, 
        ncyc=1, # fewer total v-cycles often..
        # ncyc=2,
        smoothers=smoothers, line_search = 'drig' in args.elem,
    )
    pc = vmg2
    assembler._assemble_system()

    assembler.u, nsteps = right_pgmres2(
        A=assembler.kmat, b=assembler.force,
        restart=100, M=pc, #M=vmg,
        rtol=1e-6,
    )

    total_vcyc = vmg2.total_vcycles
    print(f"{total_vcyc=}")



# ===============================
# PLOT
# ===============================

# plot is w, u, v, thx, thy

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

def norm(x):
    return np.max(np.abs(x))

u_nrm = norm(u)
v_nrm = norm(v)
w_nrm = norm(w)
disp_nrm = np.max([v_nrm, w_nrm])
norm_nrm = norm(normal_defln)
nxe = args.nxe
print(f"{nxe=} {disp_nrm=:.4e} {norm_nrm=:.4e}")
print(f"\n\t{u_nrm=:.4e}\n\t{v_nrm=:.4e}\n\t{w_nrm=:.4e}")


# DEBUG to plot forces
# assembler.u = assembler.force * 1.0

if args.plot is not None:

    assembler.plot_disp(
        # disp_mag=0.2,
        # disp_mag=0.3 * R,
        disp_mag=0.4 * R,
        # disp_mag=1.0, # same as radius (as multiple of inf-norm or max value)
        # disp_mag=2.0,
        # disp_mag=5.0,
        mode=args.plot,
    )    
