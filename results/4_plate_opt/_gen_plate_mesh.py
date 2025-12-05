import numpy
import argparse


def write_80(line):
    newline = "{:80s}\n".format(line.strip("\n"))
    fout.write(newline)


parser = argparse.ArgumentParser()
parser.add_argument("--lx", type=float, default=1.0, help="Beam length")
parser.add_argument("--ly", type=float, default=1.0, help="Beam width")
parser.add_argument("--ncx", type=int, default=32, help="num comps in x")
parser.add_argument("--nxe", type=int, default=128, help="# elements along length (and each direc)")
parser.add_argument("--name", type=str, default="plate")
args = parser.parse_args()

# Overall plate dimensions
lx = args.lx
ly = args.ly

# Number of components in each direction
ncx = args.ncx
ncy = ncx
ncomp = ncx * ncy

# Number of elements along each edge of a single panel
nex = args.nxe
ney = nex

# Nodes
nx = nex + 1
ny = ney + 1
xtmp = numpy.linspace(0, lx, nx)
ytmp = numpy.linspace(0, ly, ny)
X, Y = numpy.meshgrid(xtmp, ytmp)
Z = numpy.zeros_like(X)
nodes = numpy.stack((X, Y, Z), axis=2)
nmat = nodes.reshape((nx * ny, 3))

# Node numbering
nid = numpy.zeros((nx, ny), dtype="intc")
xL_bcnodes = [] # x = +-const
yB_bcnodes = [] # y = +-const
xR_bcnodes = [] # x = +-const
yT_bcnodes = [] # y = +-const
count = 1
for i in range(ny):
    for j in range(nx):
        nid[j, i] = count
        if j == 0:
            xL_bcnodes.append(count)
        if j == nx - 1:
            xR_bcnodes.append(count)
        if i == 0:
            yB_bcnodes.append(count)
        if i == ny - 1:
            yT_bcnodes.append(count)
        count += 1

# Connectivity
nex = nx - 1
ney = ny - 1
ne = nex * ney
conn = {i + 1: [] for i in range(ncomp)}
ie = 1
print("make connectivity")
for i in range(ney):
    for j in range(nex):
        icx = i // (nex // ncx)
        icy = j // (ney // ncy)
        compID = 1 + ncx * icy + icx
        conn[compID].append(
            [ie, nid[j, i], nid[j + 1, i], nid[j + 1, i + 1], nid[j, i + 1]]
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
    # Write nodes
    for i in range(nx * ny):
        write_bulk_line("GRID", [i + 1, 0, nmat[i, 0], nmat[i, 1], nmat[i, 2], 0, 0, 0])

    # # Write elements
    # compID = 1
    # for key in conn:
    #     famPrefix = "$       Shell element data for family    "
    #     famString = "{}{:39s}".format(famPrefix, compNames[compID])
    #     write_80(famString)
    #     compID += 1
    #     for element in conn[key]:
    #         element.insert(1, key)
    #         write_bulk_line("CQUAD4", element)

    # Precompute all element lines for BDF
    print("pre-compute element lines")
    element_lines = []
    for compID, elements in conn.items():
        famPrefix = "$       Shell element data for family    "
        famString = "{}{:39s}".format(famPrefix, compNames[compID])
        element_lines.append(famString)
        
        key_col = [compID] * len(elements)
        for k, element in zip(key_col, elements):
            # Prepend compID without modifying the original list
            elem_data = [element[0], k] + element[1:]
            element_lines.append(elem_data)

    # Write all element lines
    print("write element lines")
    for line in element_lines:
        if isinstance(line, str):
            write_80(line)
        else:
            write_bulk_line("CQUAD4", line)

    # ----------------------------------------------------------
    # WRITE NODAL LOADS (FORCE CARDS) including in-plane components
    # ----------------------------------------------------------
    print("write nodal loads (with in-plane)")

    # Compute load magnitudes per your given formula
    Xn = nmat[:, 0]
    Yn = nmat[:, 1]

    Lx = numpy.max(Xn) - numpy.min(Xn)
    nx_calc = int(Xn.shape[0] ** 0.5)
    dx = Lx / (nx_calc - 1)
    dy = dx

    R = numpy.sqrt(Xn**2 + Yn**2)
    # TH preserved if you want it — not used in radial distribution below
    TH = numpy.arctan2(Xn, Yn)

    P = 1.6e7
    base_load_mag = P * dx * dy
    # load_corr = 1.1
    # load_corr = 1.08
    load_corr = 1.3
    # slightly diff initial loads due to quadpt weighting in GPU version of code.. (hack fix here)
    base_load_mag *= load_corr  # this is the same per-node out-of-plane magnitude

    # Fraction of load that is in-plane (change as needed)
    inplane_frac = 0.6  # 10% of base load goes into in-plane components by default

    # compute per-node components
    Fz_vals = base_load_mag * numpy.sin(5.0 * numpy.pi * R / Lx) * numpy.cos(4.0 * TH)

    # In-plane: distribute radially as Fx = inplane_mag * (X/R), Fy = inplane_mag * (Y/R)
    inplane_mag = -inplane_frac * base_load_mag * numpy.sin(numpy.pi * R / Lx / 1.414)
    Fx_vals = inplane_mag * numpy.cos(TH)
    Fy_vals = inplane_mag * numpy.sin(TH)

    # Nastran FORCE card approach:
    # write one FORCE per direction per node (Fx, Fy, Fz) using direction cosines (1,0,0) etc.
    # Format: FORCE, SID, G, CID, F, N1, N2, N3
    load_sid = 1
    cid = 0

    # Node ids are 1..(nx*ny)
    for idx in range(nx * ny):
        node_id = idx + 1

        fx = float(Fx_vals[idx])
        fy = float(Fy_vals[idx])
        fz = float(Fz_vals[idx])

        # Only write non-zero forces to keep BDF smaller (optional)
        # Write Fx
        if abs(fx) > 0.0:
            write_bulk_line("FORCE", [load_sid, node_id, cid, fx, 1.0, 0.0, 0.0])
        # Write Fy
        if abs(fy) > 0.0:
            write_bulk_line("FORCE", [load_sid, node_id, cid, fy, 0.0, 1.0, 0.0])
        # Write Fz
        if abs(fz) > 0.0:
            write_bulk_line("FORCE", [load_sid, node_id, cid, fz, 0.0, 0.0, 1.0])



    # Write boundary conditions
    print("write BCs")
    for node in xL_bcnodes:
        write_bulk_line("SPC", [1, node, "134", 0.0])
    for node in yB_bcnodes:
        write_bulk_line("SPC", [1, node, "235", 0.0])
    for node in xR_bcnodes:
        write_bulk_line("SPC", [1, node, "34", 0.0])
    for node in yT_bcnodes:
        write_bulk_line("SPC", [1, node, "35", 0.0])

    write_80("ENDDATA")
