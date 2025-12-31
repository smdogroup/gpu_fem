"""
fro funtofem/examples/framework/1_*py naca wing example
"""

from funtofem import *
from tacs import caps2tacs
import openmdao.api as om
from mpi4py import MPI
import shutil, os
from pyNastran.bdf.bdf import BDF
import argparse


"""
argparse:
python gen_wing_mesh.py --level 0     => generates L0 mesh (coarsest)
python gen_wing_mesh.py --level 1     => generates L1 mesh
etc..
"""

parser = argparse.ArgumentParser(
    description="Generate wing mesh at different refinement levels."
)
parser.add_argument(
    "--level",
    type=int,
    choices=[0, 1, 2, 3, 4],
    required=True,
    help="Mesh refinement level: 0 = coarsest, 1 = finer, 2 etc..",
)
args = parser.parse_args()

# approx # elements per level..
# level 0 : 7k elements
# level 1 : 24k elements
# level 2 : 90k elements

# --------------------------------------------------------------#
# Setup CAPS Problem and FUNtoFEM model
# --------------------------------------------------------------#
comm = MPI.COMM_WORLD
f2f_model = FUNtoFEMmodel("tacs_wing")
wing = Body.aeroelastic(
    "wing"
)  # says aeroelastic but not coupled, may want to make new classmethods later...


# define the Tacs model
tacs_model = caps2tacs.TacsModel.build(csm_file="large_naca_wing.csm", comm=comm)

# global mesh size settings for each level..
if args.level == 0:
    tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
        edge_pt_min=2,
        edge_pt_max=5,
        global_mesh_size=0.1,
        max_surf_offset=0.08,
        max_dihedral_angle=20,
    ).register_to(
        tacs_model
    )

if args.level == 1:
    tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
        edge_pt_min=5,
        edge_pt_max=8,
        global_mesh_size=0.05,
        max_surf_offset=0.03,
        max_dihedral_angle=10,
    ).register_to(
        tacs_model
    )

elif args.level == 2:
    tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
        edge_pt_min=12,
        edge_pt_max=15,
        global_mesh_size=0.01,
        max_surf_offset=0.01,
        max_dihedral_angle=5,
    ).register_to(
        tacs_model
    )

elif args.level == 3:
    tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
        edge_pt_min=26,
        edge_pt_max=30,
        global_mesh_size=0.005,
        max_surf_offset=0.005,
        max_dihedral_angle=3,
    ).register_to(
        tacs_model
    )


elif args.level == 4:
    tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
        edge_pt_min=54,
        edge_pt_max=60,
        global_mesh_size=0.0025,
        max_surf_offset=0.0025,
        max_dihedral_angle=1,
    ).register_to(
        tacs_model
    )

elif args.level == 5:
    tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
        edge_pt_min=108,
        edge_pt_max=120,
        global_mesh_size=0.0015,
        max_surf_offset=0.0015,
        max_dihedral_angle=0.8,
    ).register_to(
        tacs_model
    )

tacs_aim = tacs_model.tacs_aim

aluminum = caps2tacs.Isotropic.aluminum().register_to(tacs_model)

# setup the thickness design variables + automatic shell properties
# using Composite functions, this part has to go after all funtofem variables are defined...
nribs = int(tacs_model.get_config_parameter("nribs"))
nspars = int(tacs_model.get_config_parameter("nspars"))
nOML = int(tacs_aim.get_output_parameter("nOML"))

init_thickness = 0.08
for irib in range(1, nribs + 1):
    name = f"rib{irib}"
    caps2tacs.ShellProperty(
        caps_group=name, material=aluminum, membrane_thickness=init_thickness
    ).register_to(tacs_model)
    Variable.structural(name, value=init_thickness).set_bounds(
        lower=0.01, upper=0.2, scale=100.0
    ).register_to(wing)

for ispar in range(1, nspars + 1):
    name = f"spar{ispar}"
    caps2tacs.ShellProperty(
        caps_group=name, material=aluminum, membrane_thickness=init_thickness
    ).register_to(tacs_model)
    Variable.structural(name, value=init_thickness).set_bounds(
        lower=0.01, upper=0.2, scale=100.0
    ).register_to(wing)

for name in ["LEspar", "TEspar"]:
    caps2tacs.ShellProperty(
        caps_group=name, material=aluminum, membrane_thickness=init_thickness
    ).register_to(tacs_model)
    Variable.structural(name, value=init_thickness).set_bounds(
        lower=0.01, upper=0.2, scale=100.0
    ).register_to(wing)

for prefix in ["uOML", "lOML"]:
    for iOML in range(1, nOML + 1):
        name = prefix + str(iOML)
        caps2tacs.ShellProperty(
            caps_group=name, material=aluminum, membrane_thickness=init_thickness
        ).register_to(tacs_model)
        Variable.structural(name, value=init_thickness).set_bounds(
            lower=0.01, upper=0.2, scale=100.0
        ).register_to(wing)

# add constraints and loads
# clamped vs SS root
# caps2tacs.PinConstraint("root").register_to(tacs_model)
caps2tacs.PinConstraint("root", dof_constraint=123456).register_to(tacs_model)

# caps2tacs.GridForce("OML", direction=[0, 0, 1.0], magnitude=10).register_to(tacs_model) # don't need that here

# run the tacs model setup and register to the funtofem model
f2f_model.structural = tacs_model

# register the funtofem Body to the model
wing.register_to(f2f_model)

# make the scenario(s)
tacs_scenario = Scenario.steady("tacs", steps=100)
Function.ksfailure(ks_weight=10.0).optimize(
    scale=30.0, upper=0.267, objective=False, plot=True
).register_to(tacs_scenario)
Function.mass().optimize(scale=1.0e-2, objective=True, plot=True).register_to(
    tacs_scenario
)
tacs_scenario.register_to(f2f_model)

# make the BDF and DAT file for TACS structural analysis
tacs_aim.setup_aim()
tacs_aim.pre_analysis()

# -------------------------------------------------
# here we have to use pynastran to combine BDF and DAT file since the TACS C++ MeshLoader (from BDF)
# can't read BDF + DAT file pair like pynastran..

orig_bdf = os.path.join("capsStruct_0", "Scratch", "tacs", "tacs.dat")
final_bdf = "naca_wing_L" + str(args.level) + ".bdf"

model = BDF()
model.read_bdf(orig_bdf, xref=True)

# Write out as a single combined BDF
model.write_bdf(final_bdf, size=16)


# TODO : this still doesn't quite work yet for running case..
# # --------------------------------------------------
# # try writing out BDF file in more benign order of elems (ESP/CAPS has strange order of actual nodes, elems despite writing out components in right order)
# # the CQUAD4 1,2,3, etc are spTE3  then next block is spTE4 and rib3? so here I try to fix that by rebuilding BDF more structured way

# # --- Prepare new BDF ---
# import copy
# new_model = copy.deepcopy(model) # so that way we keep BEGIN BULK labels, etc.
# # --- Keep track of nodes already added ---
# added_nodes = set()
# # --- Loop through components ---
# nnodes = len(model.nodes.items())
# node_map = { inode : inode for inode in range(1, nnodes + 1)}

# # first loop through and add nodes to the list
# new_nid = 1
# for icomp, comp in enumerate(component_groups):
#     for eid, elem in sorted(model.elements.items()):
#         pid = elem.Pid()
#         prop = model.properties[pid]
#         if hasattr(prop, 'comment') and comp in prop.comment:
#             # print(f"{elem.__dict__=}")
#             # print(f"\n{elem.node_ids=}")

#             # Add element nodes first
#             for nid in elem.node_ids:
#                 if nid not in added_nodes:
#                     node = model.nodes[nid]
#                     node2 = copy.deepcopy(node)
#                     node2.nid = new_nid
#                     # print(f"{node2.__dict__=}")
#                     new_model.nodes[new_nid] = node2
#                     added_nodes.add(new_nid)

#                     # update map, so we can update nodes in elem later..
#                     node_map[nid] = new_nid
#                     new_nid += 1

# # now add new elements again deepcopying
# new_eid = 1
# for icomp, comp in enumerate(component_groups):
#     for eid, elem in sorted(model.elements.items()):
#         pid = elem.Pid()
#         prop = model.properties[pid]
#         if hasattr(prop, 'comment') and comp in prop.comment:

#             # just need to update the elem pid and nodes and elem id
#             elem2 = copy.deepcopy(elem)
#             elem2.pid = icomp + 1
#             elem2.eid = new_eid
#             old_nodes = elem.nodes
#             new_nodes = [node_map[_node] for _node in old_nodes]
#             elem2.nodes = new_nodes

#             elem2.nodes_ref = [new_model.nodes[nid] for nid in elem2.nodes]
#             prop = model.properties[elem.pid]
#             prop2 = copy.deepcopy(prop)
#             prop2.pid = icomp + 1
#             elem2.pid_ref = prop2
#             # print(f"{elem.__dict__=}")
#             # print(f"{elem2.__dict__=}")
#             new_model.elements[new_eid] = elem2
#             new_eid += 1

# for spc_group in new_model.spcs:
#     for spc in new_model.spcs[spc_group]:
#         # print(f"old : {spc.__dict__=}")
#         old_nodes = spc.nodes
#         new_nids = [node_map[_node] for _node in old_nodes]
#         new_nodes = [new_nid for new_nid in new_nids]
#         spc.nodes = new_nodes
#         spc.nodes_ref = [new_model.nodes[new_nid] for new_nid in new_nodes]
#         # print(f"new : {spc.__dict__=}")

# # --- Write out sorted BDF ---
# new_model.write_bdf(final_bdf, size=16) #, interspersed=True)
# print(f"Written sorted BDF with elements and nodes reordered by component: {final_bdf}")

