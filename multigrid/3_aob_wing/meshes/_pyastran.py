
from funtofem import *
import numpy as np
import time
start_time = time.time()
import argparse
import shutil, os
from pyNastran.bdf.bdf import BDF
import copy

def num_to_padstr(mynum):
    return str(mynum) # not really needed tbh.. as long as it does lOML then uOML, then ribs and spars, etc.

nribs = 23
nOML = nribs - 1
component_groups = ["rib" + num_to_padstr(irib) for irib in range(1, nribs + 1)]
for prefix in ["spLE", "spTE", "uOML", "lOML"]:
    component_groups += [prefix + num_to_padstr(iOML) for iOML in range(1, nOML + 1)]
component_groups = sorted(component_groups)

orig_bdf = os.path.join("capsStruct_0", "Scratch", "tacs", "tacs.dat")
final_bdf = "aob_wing_L0.bdf"

model = BDF()
model.read_bdf(orig_bdf, xref=True)

# # Write out as a single combined BDF (orig)
# model.write_bdf(final_bdf, size=16, sort_ids=False)

# --------------------------------------------------
# try writing out BDF file in more benign order of elems (ESP/CAPS has strange order of actual nodes, elems despite writing out components in right order)
# the CQUAD4 1,2,3, etc are spTE3  then next block is spTE4 and rib3? so here I try to fix that by rebuilding BDF more structured way

# --- Prepare new BDF ---
new_model = copy.deepcopy(model) # so that way we keep BEGIN BULK labels, etc.
# --- Keep track of nodes already added ---
added_nodes = set()
# --- Loop through components ---
nnodes = len(model.nodes.items())
node_map = { inode : inode for inode in range(1, nnodes + 1)}

# first loop through and add nodes to the list
new_nid = 1
for icomp, comp in enumerate(component_groups):
    for eid, elem in sorted(model.elements.items()):
        pid = elem.Pid()
        prop = model.properties[pid]
        if hasattr(prop, 'comment') and comp in prop.comment:
            # print(f"{elem.__dict__=}")
            # print(f"\n{elem.node_ids=}")

            # Add element nodes first
            for nid in elem.node_ids:
                if nid not in added_nodes:
                    node = model.nodes[nid]
                    node2 = copy.deepcopy(node)
                    node2.nid = new_nid
                    # print(f"{node2.__dict__=}")
                    new_model.nodes[new_nid] = node2
                    added_nodes.add(new_nid)

                    # update map, so we can update nodes in elem later..
                    node_map[nid] = new_nid
                    new_nid += 1

# now add new elements again deepcopying
new_eid = 1
for icomp, comp in enumerate(component_groups):
    for eid, elem in sorted(model.elements.items()):
        pid = elem.Pid()
        prop = model.properties[pid]
        if hasattr(prop, 'comment') and comp in prop.comment:

            # just need to update the elem pid and nodes and elem id
            elem2 = copy.deepcopy(elem)
            elem2.pid = icomp + 1
            elem2.eid = new_eid
            old_nodes = elem.nodes
            new_nodes = [node_map[_node] for _node in old_nodes]
            elem2.nodes = new_nodes

            elem2.nodes_ref = [new_model.nodes[nid] for nid in elem2.nodes]
            prop = model.properties[elem.pid]
            prop2 = copy.deepcopy(prop)
            prop2.pid = icomp + 1
            elem2.pid_ref = prop2
            # print(f"{elem.__dict__=}")
            # print(f"{elem2.__dict__=}")
            new_model.elements[new_eid] = elem2
            new_eid += 1

new_model.materials = []
new_model.properties = []

# print(f"{node_map[519]=}")

for spc_group in new_model.spcs:
    for spc in new_model.spcs[spc_group]:
        # print(f"old : {spc.__dict__=}")
        old_nodes = spc.nodes
        new_nids = [node_map[_node] for _node in old_nodes]
        new_nodes = [new_nid for new_nid in new_nids]
        spc.nodes = new_nodes
        spc.nodes_ref = [new_model.nodes[new_nid] for new_nid in new_nodes]
        # print(f"new : {spc.__dict__=}")

# --- Write out sorted BDF ---
new_model.write_bdf(final_bdf, size=16) #, interspersed=True)
print(f"Written sorted BDF with elements and nodes reordered by component: {final_bdf}")
