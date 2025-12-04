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

    print("write BCs")
    for node in x_bcnodes:
        write_bulk_line("SPC", [1, node, "1234", 0.0])

    write_80("ENDDATA")
