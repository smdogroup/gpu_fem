""" develop algorithm for parallel graph coloring (demonstrate on CPU serial here first) """

import numpy as np
import matplotlib.pyplot as plt

""" inputs """
# plate problem,
# nxe = 4
nxe = 8
# nxe = 16
# nxe = 32
# nxe = 64
# nxe = 128

nx = nxe + 1
N = nx**2

# batch fraction for randomized parallel coloring
# randomized smaller batches should have less parallel update conflicts => resulting in more parallel and lower # of colors
# batch_frac = 0.2
batch_frac = 0.1

# maybe look at probabilities of nodes in batch being adjacent or connected based on batch size?
# maybe it can still settle and decrease colors later?

""" create element connectivity and init colors """
parallel_colors = np.zeros(N)

elem_conn = []
for ielem in range(nxe**2):
    ixe, iye = ielem % nxe, ielem // nxe
    _node = iye * nx + ixe
    nodes = [_node, _node + 1, _node + nx + 1, _node + nx]
    elem_conn += [nodes]

# also need to convert this to list of adjacent nodes.. in dict?
adj_nodes_dict = {inode:[] for inode in range(N)}
for loc_nodes in elem_conn:
    for inode in loc_nodes:
        for jnode in loc_nodes:
            if jnode == inode: continue # don't include node in itself the adjacency list
            adj_nodes_dict[inode] += [jnode]

# now make unique
for inode in range(N):
    adj_nodes_dict[inode] = list(np.unique(np.array(adj_nodes_dict[inode])))

# print(f"{adj_nodes_dict=}")
# exit()

""" demo the coloring algorithm as if in parallel """

def get_min_color(_adj_colors):
    for _color in range(1, 20):
        if not(_color in _adj_colors): return _color
    return None

batch_size = int(N * batch_frac)
# NOTE : red-black GS would not work for 4-node finite elements (because up to 8 nodal connections)

for i_outer in range(300):

    # get current num unique colors
    n_unique_colors = np.unique(parallel_colors[parallel_colors > 0]).shape[0]

    print(f"{i_outer=} {n_unique_colors=}")

    # choose random batch (no repeats)
    rand_batch = np.random.permutation(N)[:batch_size]
    # print(F"{rand_batch=}")

    # copy out current colors since in parallel you can't see changes of other procs of threads, etc.
    current_colors = parallel_colors.copy()

    # now plot each iteration..
    # if i_outer % 10 == 0:
    #     COLOR_MAT = np.reshape(current_colors.copy(), (nx, nx))
    #     plt.imshow(COLOR_MAT, cmap='jet')
    #     plt.show()

    for inode in rand_batch:
        adj_nodes = adj_nodes_dict[inode]
        adj_colors = current_colors[adj_nodes]
        parallel_colors[inode] = get_min_color(adj_colors)

    # can speedu

    # check how many nodes could be decreased (conv)
    can_decrease = 0
    for inode in range(N):
        adj_nodes = adj_nodes_dict[inode]
        adj_colors = parallel_colors[adj_nodes]
        can_dec = parallel_colors[inode] != get_min_color(adj_colors)

        can_decrease += int(can_dec)
    print(f"\t{can_decrease=}")
    if can_decrease == 0: break

# versus greedy one
greedy_colors = np.zeros(N)
for inode in range(N):
    adj_nodes = adj_nodes_dict[inode]
    adj_colors = greedy_colors[adj_nodes]
    greedy_colors[inode] = get_min_color(adj_colors)
n_greedy_colors = np.unique(greedy_colors[greedy_colors > 0]).shape[0]
print(f'{n_greedy_colors=}')


# compare greedy vs parallel coloring
fig, ax = plt.subplots(1, 2)

SERIAL_COLOR_MAT = np.reshape(greedy_colors.copy(), (nx, nx))
ax[0].imshow(SERIAL_COLOR_MAT, cmap='jet')
ax[0].set_title(f"Serial : {n_greedy_colors} colors")

GPU_COLOR_MAT = np.reshape(current_colors.copy(), (nx, nx))
ax[1].imshow(GPU_COLOR_MAT, cmap='jet')
ax[1].set_title(f"Parallel : {n_unique_colors} colors")
plt.show()


# show matrix sparsities..
# construct init 

def get_matrix_sparsity(colors):
    # get permutation vec from the colors
    n_colors = np.unique(colors).shape[0]
    i_perm = np.zeros(N, dtype=np.int32)
    perm = np.zeros(N, dtype=np.int32)
    ct = 0
    for i_color in range(1, n_colors + 1):
        for i in range(N):
            if colors[i] == i_color: 
                i_perm[ct] = i
                perm[i] = ct
                ct += 1

    # print(f'{colors=}')
    # print(f'{perm=}')

    # get matrix sparsity (no reorder first)
    A = np.zeros((N, N))
    for elem_nodes in elem_conn:
        elem_colors = colors[elem_nodes]
        for j in elem_nodes:
            A[elem_nodes, j] = elem_colors

    # plt.imshow(A)
    # plt.show()
    
    # now reorder
    A_perm = A[i_perm,:][:,i_perm]

    # plt.imshow(A_perm)
    # plt.show()

    # now also nan the zeros..
    # A_perm[A_perm == 0.0] = np.nan

    # plt.imshow(A_perm)
    # plt.show()
    return A_perm

A_greedy = get_matrix_sparsity(greedy_colors)
A_parallel = get_matrix_sparsity(parallel_colors)

plt.close('all')
fig, ax = plt.subplots(1, 2)

ax[0].imshow(A_greedy, cmap='jet')
ax[0].set_title(f"A_serial : {n_greedy_colors} colors")

ax[1].imshow(A_parallel, cmap='jet')
ax[1].set_title(f"A_parallel : {n_unique_colors} colors")
plt.show()
