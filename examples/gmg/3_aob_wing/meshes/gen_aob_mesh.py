"""
Sean P. Engelstad, Georgia Tech 2023

Local machine optimization for the panel thicknesses using nribs-1 OML panels and nribs-1 LE panels
"""

from funtofem import *
import numpy as np
import time
start_time = time.time()
import argparse
import shutil, os
from pyNastran.bdf.bdf import BDF

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--spanMult", type=float, default=1.0) # recommend 1.0, 1.1, 1.2, 1.3
parent_parser.add_argument(
    "--level",
    type=int,
    choices=[0, 1, 2, 3, 4],
    required=True,
    help="Mesh refinement level: 0 = coarsest, 1 = finer, 2 etc..",
)
args = parent_parser.parse_args()

# import openmdao.api as om
from mpi4py import MPI
from tacs import caps2tacs, pytacs
import os


comm = MPI.COMM_WORLD

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "aob_geom", "gbm.csm")

# F2F MODEL and SHAPE MODELS
# ----------------------------------------

f2f_model = FUNtoFEMmodel('model')
tacs_model = caps2tacs.TacsModel.build(
    csm_file=csm_path,
    comm=comm,
    problem_name="capsStruct",
    active_procs=[0],
    verbosity=1,
)

# global mesh size settings for each level..
# if args.level == 0:
#     tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
#         edge_pt_min=2,
#         edge_pt_max=5,
#         global_mesh_size=0.1,
#         max_surf_offset=0.08,
#         max_dihedral_angle=20,
#     ).register_to(
#         tacs_model
#     )

# finer level 0
if args.level == 0:
    tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
        edge_pt_min=2,
        edge_pt_max=4,
        global_mesh_size=0.2,
        max_surf_offset=0.16,
        max_dihedral_angle=40,
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


f2f_model.structural = tacs_model

tacs_aim = tacs_model.tacs_aim
tacs_aim.set_config_parameter("view:flow", 0)
tacs_aim.set_config_parameter("view:struct", 1)
# tacs_aim.set_design_parameter("spanMult", args.spanMult)
#tacs_aim.set_config_parameter("fullRootRib", 0)

egads_aim = tacs_model.mesh_aim

# # 17000 if elem_mult = 1
# # 70000 if elem_mult = 2
# elem_mult = 2

# if comm.rank == 0:
#     aim = egads_aim.aim
#     aim.input.Mesh_Sizing = {
#         "chord": {"numEdgePoints": 20*elem_mult},
#         "span": {"numEdgePoints": 10*elem_mult},
#         "vert": {"numEdgePoints": 10*elem_mult},
#     }

# BODIES AND STRUCT DVs
# -------------------------------------------------

wing = Body.aeroelastic("wing", boundary=2)
# aerothermoelastic

# setup the material and shell properties
nribs = int(tacs_model.get_config_parameter("nribs"))
nOML = nribs - 1
null_material = caps2tacs.Orthotropic.null().register_to(tacs_model)

def num_to_padstr(mynum):
    # if mynum < 10:
    #     return "0" + str(mynum)
    # else:
    #     return str(mynum)
    return str(mynum) # not really needed tbh.. as long as it does lOML then uOML, then ribs and spars, etc.

# create the design variables by components now
# since this mirrors the way TACS creates design variables
component_groups = ["rib" + num_to_padstr(irib) for irib in range(1, nribs + 1)]
for prefix in ["spLE", "spTE", "uOML", "lOML"]:
    component_groups += [prefix + num_to_padstr(iOML) for iOML in range(1, nOML + 1)]
component_groups = sorted(component_groups)

# print(f"{component_groups=}")
# exit()

# delim = "-"
delim = "_" # had to change to _ so pynastran doesn't complain about prop to DV definitions (just for this case)

for icomp, comp in enumerate(component_groups):
    caps2tacs.CompositeProperty.null(comp, null_material).register_to(tacs_model)

    # NOTE : need to make the struct DVs in TACS in the same order as the blade callback
    # which is done by components and then a local order

    # panel length variable
    if "rib" in comp:
        panel_length = 0.38
    elif "sp" in comp:
        panel_length = 0.36
    elif "OML" in comp:
        panel_length = 0.65
    Variable.structural(
        f"{comp}{delim}" + TacsSteadyInterface.LENGTH_VAR, value=panel_length
    ).set_bounds(
        lower=0.0,
        scale=1.0,
        state=True,  # need the length & width to be state variables
    ).register_to(
        wing
    )

    # stiffener pitch variable
    Variable.structural(f"{comp}{delim}spitch", value=0.20).set_bounds(
        lower=0.05, upper=0.5, scale=1.0
    ).register_to(wing)

    # panel thickness variable, shortened DV name for ESP/CAPS, nastran requirement here
    Variable.structural(f"{comp}{delim}T", value=0.02).set_bounds(
        lower=0.002, upper=0.1, scale=100.0
    ).register_to(wing)

    # stiffener height
    Variable.structural(f"{comp}{delim}sheight", value=0.05).set_bounds(
        lower=0.002, upper=0.1, scale=10.0
    ).register_to(wing)

    # stiffener thickness
    Variable.structural(f"{comp}{delim}sthick", value=0.02).set_bounds(
        lower=0.002, upper=0.1, scale=100.0
    ).register_to(wing)

    Variable.structural(
        f"{comp}{delim}" + TacsSteadyInterface.WIDTH_VAR, value=panel_length
    ).set_bounds(
        lower=0.0,
        scale=1.0,
        state=True,  # need the length & width to be state variables
    ).register_to(
        wing
    )

# caps2tacs.PinConstraint("root", dof_constraint=12346).register_to(tacs_model)
# caps2tacs.PinConstraint("root", dof_constraint=2346).register_to(tacs_model)
caps2tacs.PinConstraint("root", dof_constraint=246).register_to(tacs_model)
caps2tacs.PinConstraint("sob", dof_constraint=13).register_to(tacs_model)

# register the wing body to the model
wing.register_to(f2f_model)

# INITIAL STRUCTURE MESH, SINCE NO STRUCT SHAPE VARS
# --------------------------------------------------

tacs_aim.setup_aim()
tacs_aim.pre_analysis()

# -------------------------------------------------
# here we have to use pynastran to combine BDF and DAT file since the TACS C++ MeshLoader (from BDF)
# can't read BDF + DAT file pair like pynastran..

orig_bdf = os.path.join("capsStruct_0", "Scratch", "tacs", "tacs.dat")
final_bdf = "aob_wing_L" + str(args.level) + ".bdf"

model = BDF()
model.read_bdf(orig_bdf, xref=True)

# Write out as a single combined BDF (orig)
model.write_bdf(final_bdf, size=16) #, sort_ids=False)

# TODO : still doesn't quite work yet..

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

