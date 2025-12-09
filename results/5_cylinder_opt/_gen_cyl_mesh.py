import numpy
import argparse


def write_80(line):
    newline = "{:80s}\n".format(line.strip("\n"))
    fout.write(newline)


parser = argparse.ArgumentParser()
parser.add_argument("--lx", type=float, default=1.0, help="Cylinder length")
parser.add_argument("--ly", type=float, default=0.5, help="Cylinder radius")
parser.add_argument("--ncx", type=int, default=32, help="num comps along X")
parser.add_argument("--nxe", type=int, default=128, help="# elements along X")
parser.add_argument("--name", type=str, default="cylinder")
args = parser.parse_args()

# Cylinder dimensions
lx = args.lx      # along X
R = args.ly       # cylinder radius

# Number of components
ncx = args.ncx
ncy = ncx
ncomp = ncx * ncy

# Number of elements along each direction
nex = args.nxe
ney = nex   # for simplicity, same number of elements along hoop

# Nodes
nx = nex + 1       # axial nodes
ny = ney           # hoop nodes (last node wraps around)
xtmp = numpy.linspace(0, lx, nx)
theta_tmp = numpy.linspace(0, 2 * numpy.pi, ny, endpoint=False)
X, Theta = numpy.meshgrid(xtmp, theta_tmp)
Y = R * numpy.cos(Theta)
Z = R * numpy.sin(Theta)
nodes = numpy.stack((X, Y, Z), axis=2)
nmat = nodes.reshape((nx * ny, 3))

# Node numbering
nid = numpy.zeros((nx, ny), dtype=int)
x_bcnodes = []
count = 1
for i in range(ny):
    for j in range(nx):
        nid[j, i] = count
        if j == 0 or j == nx-1:
            x_bcnodes.append(count)
        count += 1

# Connectivity with wrap-around in hoop direction
conn = {i + 1: [] for i in range(ncomp)}
ie = 1
for i in range(ny):
    for j in range(nex):
        icx = i // (ny // ncx)
        icy = j // (nex // ncy)
        compID = 1 + ncx * icy + icx

        # Wrap-around last element in hoop direction
        i_next = (i + 1) % ny

        conn[compID].append(
            [ie, nid[j, i], nid[j + 1, i], nid[j + 1, i_next], nid[j, i_next]]
        )
        ie += 1

# Write BDF
output_file = args.name + ".bdf"

print("open BDF file")
with open(output_file, "w") as fout:
    write_80("SOL 103")
    write_80("CEND")
    write_80("BEGIN BULK")

    # Make component names
    print("make component names")
    compNames = {}
    compID = 1
    for i in range(ncy):
        for j in range(ncx):
            compNames[compID] = "PLATE.{:03d}/SEG.{:02d}".format(i, j)
            compID += 1

    def write_bulk_line(key, items, format="small"):
        if format == "small":
            width = 8
            writekey = key
        elif format == "large":
            width = 16
            writekey = key + "*"
        line = "{:8s}".format(writekey)
        for item in items:
            if type(item) in [int, numpy.int64, numpy.int32]:
                line += "{:{width}d}".format(item, width=width)[:width]
            elif type(item) in [float, numpy.float64]:
                line += "{: {width}f}".format(item, width=width)[:width]
            elif type(item) is str:
                line += "{:{width}s}".format(item, width=width)[:width]
            else:
                print(type(item), item)
            if len(line) == 72:
                write_80(line)
                line = " " * 8
        if len(line) > 8:
            write_80(line)

    print("write nodes")
    for i in range(nx * ny):
        write_bulk_line("GRID", [i + 1, 0, nmat[i, 0], nmat[i, 1], nmat[i, 2], 0, 0, 0])

    # Precompute element lines
    print("pre-compute element lines")
    element_lines = []
    for compID, elements in conn.items():
        famPrefix = "$       Shell element data for family    "
        famString = "{}{:39s}".format(famPrefix, compNames[compID])
        element_lines.append(famString)
        
        key_col = [compID] * len(elements)
        for k, element in zip(key_col, elements):
            elem_data = [element[0], k] + element[1:]
            element_lines.append(elem_data)

    print("write element lines")
    for line in element_lines:
        if isinstance(line, str):
            write_80(line)
        else:
            write_bulk_line("CQUAD4", line)

    # xpts: flattened array [x0,y0,z0, x1,y1,z1, ...]
    X = nmat[:,0]
    Y = nmat[:,1]
    Z = nmat[:,2]

    import numpy as np
    N = X.shape[0]
    node_ids = np.arange(1, N+1)

    # --- Cylinder geometry ---
    Lx = np.max(X) - np.min(X)
    nx = int(np.sqrt(N))     # crude axial count estimate
    dx = Lx / (nx - 1)

    R = 0.5
    dth = 2*np.pi / nx
    ds  = R * dth

    # Circumferential angle
    TH = np.arctan2(Z, Y)     # matches your version

    # --- Base load magnitude (like plate) ---
    load_mag = 5e7 * dx * ds
    load_mag *= (2.14 / 3.56)

    # --- Normalized coordinates ---
    x_hat = X / Lx
    th    = TH
    th_hat = th / (2*np.pi)

    # --- OUT-OF-PLANE magnitude (your rose function) ---
    Fmag_out = load_mag * (
        0.3*np.cos(5.0*th + 2*np.pi*x_hat) +
        0.7*np.cos(10.0*th + np.pi/6.0 + 5.3*np.pi*x_hat)
    ) * np.sin(5*np.pi*x_hat + x_hat**2)

    # =====================================================
    #              IN-PLANE LOAD (NEW)
    # =====================================================

    # fraction like plate version
    inplane_frac = 0.6
    Fmag_in = -inplane_frac * load_mag * np.sin(np.pi * x_hat)

    # ***Cylinder tangent directions***
    # Hoop tangent:   eθ = [-sinθ  0   cosθ]
    # Axial tangent:  ex = [1 0 0]
    e_theta_y = -np.sin(TH)   # Y direction component
    e_theta_z =  np.cos(TH)   # Z direction component

    # In-plane is split between axial + hoop
    # You can change weights if needed
    axial_frac = 0.5
    hoop_frac  = 0.5

    Fx_in = axial_frac * Fmag_in           # axial component
    Fy_in = hoop_frac  * Fmag_in * e_theta_y
    Fz_in = hoop_frac  * Fmag_in * e_theta_z

    # =====================================================
    #     OUT-OF-PLANE direction (local normal)
    # =====================================================
    # Radial normal:
    # er = [0, cosθ, sinθ]
    Fy_out = Fmag_out * np.cos(TH)
    Fz_out = Fmag_out * np.sin(TH)

    # Total forces per node
    Fx = Fx_in                # only axial contributes to x
    Fy = Fy_in + Fy_out
    Fz = Fz_in + Fz_out

    # =====================================================
    #            WRITE NASTRAN FORCE CARDS
    # =====================================================
    load_sid = 1
    cid = 0

    for i, nid in enumerate(node_ids):
        fx, fy, fz = Fx[i], Fy[i], Fz[i]

        if abs(fx) > 0:
            write_bulk_line("FORCE", [load_sid, nid, cid, fx, 1.0, 0.0, 0.0])
        if abs(fy) > 0:
            write_bulk_line("FORCE", [load_sid, nid, cid, fy, 0.0, 1.0, 0.0])
        if abs(fz) > 0:
            write_bulk_line("FORCE", [load_sid, nid, cid, fz, 0.0, 0.0, 1.0])

    


    print("write BCs")
    for node in x_bcnodes:
        write_bulk_line("SPC", [1, node, "1234", 0.0])

    write_80("ENDDATA")
