import numpy
import argparse


def write_80(line):
    newline = "{:80s}\n".format(line.strip("\n"))
    fout.write(newline)

# clamped plate BDF file

parser = argparse.ArgumentParser()
parser.add_argument("--lx", type=float, default=1.0, help="Plate length")
parser.add_argument("--pressure", type=float, default=2.0e2, help="Plate length")
# parser.add_argument("--clamped", type=int, default=1, help="is clamped (0/1)")
parser.add_argument("--ncx", type=int, default=1, help="# components along length")
parser.add_argument("--nex", type=int, default=100, help="# elements along length")
parser.add_argument("--name", type=str, default="Beam")
args = parser.parse_args()

# Overall plate dimensions
lx = args.lx

# Number of components in each direction
ncx = args.ncx

# Number of elements along each edge of a single panel
nex = args.nex

# Nodes
nx = ncx * nex + 1
xtmp = numpy.linspace(0, lx, nx)

dx = lx / args.nex

total_force = args.pressure * 1.0
nodal_load = total_force / (nx - 2)


# Write BDF
output_file = args.name + ".bdf"

with open(output_file, "w") as fout:
    write_80("""INIT MASTER(S)
NASTRAN SYSTEM(442)=-1,SYSTEM(319)=1
ID FEMAP,FEMAP
SOL SESTATIC
CEND
  TITLE = Beam Test
  ECHO = NONE
  DISPLACEMENT(PLOT) = ALL
  SPCFORCE(PLOT) = ALL
  OLOAD(PLOT) = ALL
  FORCE(PLOT,CORNER) = ALL
  STRESS(PLOT,CORNER) = ALL
SUBCASE 1
  SUBTITLE = z-shear
  SPC = 1
  LOAD = 1""")
    write_80("BEGIN BULK")

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

    # Write nodes
    for i in range(nx):
        write_bulk_line("GRID", [i + 1, 0, dx * i, 0.0, 0.0, 0, 0, 0])

    for i in range(1, nx-1):
        # xi = i * 1.0 / args.nex
        write_bulk_line("FORCE", [1, i+1, 0, 1., 0., 0., nodal_load])

    # Write elements
    compID = 1
    for ielem in range(1, args.nex + 1):
        write_bulk_line("CBAR", [ielem, 1, ielem, ielem+1, 0., 1., 0.])
    
    # left pin
    write_bulk_line("SPC1", [1, "123456", 1])
    write_bulk_line("SPC1", [1, "3", nx])

    write_80("ENDDATA")
