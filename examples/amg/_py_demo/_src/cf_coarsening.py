import numpy as np

"""
Three main types of coarsening stencils in AMG, each leading to different operator complexity (and thus extra memory)
    1) C1 coarsening, original Ruge-Stuben chapter https://epubs.siam.org/doi/10.1137/1.9781611971057.ch4, and discussed by hypre  https://epubs.siam.org/doi/epdf/10.1137/040615729
    2) H1 coarsening, see hypre  https://epubs.siam.org/doi/epdf/10.1137/040615729
    3) CR coarsening, https://www.sciencedirect.com/science/article/pii/S0096300320307487
# see also this paper from RN-AMG with some extra ideas, https://epubs.siam.org/doi/epdf/10.1137/16M1082706
"""

def strength_of_connections_csr(A_csr, threshold:float=0.25):
    """
    Compute strength of connections S_{ij} and S_{ij}^T for sparse CSR matrix
    """

    A = A_csr
        
    # strength of connection graph
    nnodes = A.shape[0]
    STRENGTH = [[] for i in range(nnodes)]
    for i in range(nnodes):
        row_inds = A.indices[A.indptr[i] : A.indptr[i+1]]
        row_vals = A.data[row_inds]
        row_max = np.max(np.abs(row_vals))

        for jp in range(A.indptr[i], A.indptr[i+1]):
            j = A.indices[jp]
            conn = np.abs(A.data[jp])
            if conn >= threshold * row_max:
                STRENGTH[i] += [j]

    # then construct also the transpose strength of connections
    STRENGTH_TR = [[] for i in range(nnodes)]
    for i in range(nnodes):
        jlist = STRENGTH[i]
        for j in jlist:
            STRENGTH_TR[j] += [i]
    
    return STRENGTH, STRENGTH_TR

def RS_csr_coarsening(A_csr, threshold:float=0.25):
    """
    # basic Ruge Stuben coarsening 

    # see Ruge Stuben paper, https://epubs.siam.org/doi/10.1137/1.9781611971057.ch4
    # original coarsening method from 1987
    # see also hypre paper, https://epubs.siam.org/doi/epdf/10.1137/040615729
    # higher operator complexity, better residual reduction per Krylov iteration than H1
    """

    # get strength of connections
    nnodes = A_csr.shape[0]
    STRENGTH, STRENGTH_TR = strength_of_connections_csr(A_csr, threshold)
    LAM = np.array([len(STRENGTH_TR[i]) for i in range(nnodes)])

    # ==========================================================================
    """algorithm A2 for preliminary C-point choice from Ruge Stuben page 29"""
    # ==========================================================================

    # do sets using bools instead of add and delete to lists
    C = np.full(nnodes, False) # coarse
    F = np.full(nnodes, False) # fine
    U = np.full(nnodes, True) # universe (to be moved to C and F sets)
    node_list = np.arange(0, nnodes)

    # for _ in range(30):
    while np.sum(U) > 0: # while universe set not empty
        # step 2 : pick entry in universe with maximal LAM (number of strong connections)
        # add to coarse set, then remove from universe
        U_nodes = node_list[U]
        ind = np.argmax(LAM[U])
        i = U_nodes[ind]
        if LAM[i] == 0:
            print(f"ERROR : LAM[{i}] = 0 but it was selected as coarse node..")
            break

        # print(f"{i=} {U[i]=} {LAM[i]}")

        C[i] = True
        U[i] = False

        # U_nodes = node_list[U]
        # print(f"{U_nodes=}")

        # step 3 : for all j in S_i^T and universe still, (candidate fine nodes) do:
        #  new F points from all strong neighbors unlabeled as C/F yet
        new_F_points = np.array([jj for jj in STRENGTH_TR[i] if U[jj]])
        # print(f"{new_F_points=}")
        # print(f"{new_F_points=}")
        for j in new_F_points:
            # step 4 add new F point to F set
            F[j] = True
            U[j] = False

            # step 5 update lam connections of F-node neighbors (increase lam)
            # not sure why
            F_neighbors = np.array([kk for kk in STRENGTH[j] if U[kk]])
            LAM[F_neighbors] += 1
        
        # step 6: decrement LAM (num strong neighbors) for the added C point
        C_neighbors = np.array([jj for jj in STRENGTH[i] if U[jj]])
        # print(f"{C_neighbors=}")
        if C_neighbors.shape[0] > 0:
            LAM[C_neighbors] -= 1

        # print(f"\tremoved C-node {i} with now {LAM[i]=} and {U[i]=}")

        # break

    # Algorithm A3 - final C-point choice,checking strong connections
    T = np.full(nnodes, False) # test points for F
    F_rem = F.copy() # remaining F nodes that are untested
    while np.sum(F_rem) > 0:
        i = np.argmax(F_rem)
        T[i] = True; F_rem[i] = False
        Si = STRENGTH[i]
        Ci = [jj for jj in Si if C[jj]]
        Dis = [jj for jj in Si if not(jj in Ci)]
        # Ni = A_csr.indices[A_csr.indptr[i] : A_csr.indptr[i+1]]
        # Diw = np.array([jj for jj in Ni if not(jj in Si)])
        Ci_bar = []
        # TODO : could also define tentative prolongator here also
        for j in Dis:
            Sj_and_Ci = [jj for jj in STRENGTH[j] if jj in Ci]
            if len(Sj_and_Ci) == 0:
                if len(Ci_bar) != 0:
                    # change this F point to C point
                    C[i] = True
                    F[i] = False
                    # then continue with next loop
                    continue
                else:
                    Ci_bar += [j]
                    Ci += [j]
                    Dis.remove(j)
        # step 9
        # move additional nodes to C nodes
        for jj in Ci_bar:
            C[jj] = True
            F[jj] = False

    # or just return bool masks..
    return C, F



def CLJP_coarsening(A_csr):
    # termed this in paper, https://epubs.siam.org/doi/epdf/10.1137/040615729
    # but algorithm of CFJP found in https://www.researchgate.net/publication/223921531_BoomerAMG_A_parallel_algebraic_multigrid_solver_and_preconditioner
    pass

def PMIS_csr_coarsening(A_csr):
    # see hypre paper, https://epubs.siam.org/doi/epdf/10.1137/040615729
    # lower operator complexity, worse residual reduction per Krylov step than CLJP C1 method
    pass

def HMIS_csr_coarsening(A_csr):
    # see hypre paper, https://epubs.siam.org/doi/epdf/10.1137/040615729
    # lower operator complexity, worse residual reduction per Krylov step than CLJP C1 method
    pass

def Falgout_csr_coarsening(A_csr):
    # not sure which paper defined in, but is discussed / used in this paper of hypre, 
    pass

def CR_csr_coarsening(A_csr):
    # see paper, https://www.sciencedirect.com/science/article/pii/S0096300320307487
    # best of both worlds of C1 and H1 methods, medium operator complexity, better conv.. is goal
    pass