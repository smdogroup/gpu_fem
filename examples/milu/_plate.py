
from __src import get_tacs_matrix, sort_vis_maps
from __src import random_ordering, reorder_bsr6_nofill, gen_plate_mesh
from __ilu import q_ordering
from _fillin import ilu_k_pattern_bsr
import numpy as np

def make_plate_case(args, qorder_p:float=1.0, complex_load:bool=True, apply_bcs:bool=True):
    """make plate case helper function (to be used in this script for single-level ILU and the next script for multilevel ILU)"""

    gen_plate_mesh(nxe=args.nxe, lx=1.0, ly=1.0, apply_bcs=apply_bcs)

    # ====================================================
    # 1) create and assemble FEA problem
    # ====================================================

    thickness = args.thick

    if complex_load:
        load_fcn = None # None gives more complicated loading function..
    else: # simple load
        load_fcn = lambda _x, _y : 1.0

    if apply_bcs:
        bdf_file = "plate.bdf"
    else:
        bdf_file = "plate_nobc.bdf"

    A00, rhs00, xpts00 = get_tacs_matrix(bdf_file=bdf_file, thickness=thickness, load_fcn=load_fcn)

    # doesn't quite work because the matrix values are not computed to higher precision first?

    # ===================================================
    # 2) random reordering (instead of Qordering for now)
    # ===================================================

    np.random.rand(12345678)

    N = A00.shape[0]
    nnodes = N // 6
    nnzb = A00.data.shape[0]
    print(f"{nnodes=}")

    # permute to lexigraphic ordering
    # since TACS reads in a weird order
    # ====================================
    free_dof = [_ for _ in range(N)]
    sort_fw, sort_bk = sort_vis_maps(args.nxe, xpts00, free_dof)
    perm = np.zeros(nnodes, dtype=np.int32)
    iperm = np.zeros_like(perm)
    # print(f"{sort_fw=}")
    for i in range(perm.shape[0]):
        j = sort_fw[6 * i] // 6
        perm[i] = j
        iperm[j] = i
    # print(f"{perm=}")
    A0 = reorder_bsr6_nofill(A00.copy(), perm, iperm)
    rhs0 = rhs00.reshape(nnodes, 6)[iperm].reshape(-1) 

    xpts00_arr = xpts00.reshape((nnodes, 3))
    xpts0_arr = xpts00_arr[iperm]
    # xpts0 = xpts0_arr.reshape(-1)

    if args.random:
        print("doing random..")
        # perm, iperm = random_ordering(nnodes)

        # qorder_p = 0.5
        # qorder_p = 1.0
        # qorder_p = 2.0
        # qorder_p = 0.5
        # qorder_p = 0.1

        perm, iperm = q_ordering(A0, prune_factor=qorder_p)
        A = reorder_bsr6_nofill(A0.copy(), perm, iperm)
        rhs = rhs0.reshape(nnodes, 6)[iperm].reshape(-1)
        # print(f"{A0.shape=} {A.shape=} {rhs.shape=}")
    else:
        A = A0.copy()
        rhs = rhs0.copy()
        perm, iperm = np.arange(0, nnodes), np.arange(0, nnodes)

    xpts_arr = xpts0_arr[iperm]
    xpts = xpts_arr.reshape(-1)


    # add ILU(k) fillin to the matrix
    A = ilu_k_pattern_bsr(A, k_fill=args.fill)

    return A0, rhs0, A, rhs, perm, xpts