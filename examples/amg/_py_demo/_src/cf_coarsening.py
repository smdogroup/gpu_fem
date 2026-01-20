import numpy as np
import scipy.sparse as sp

"""
Three main types of coarsening stencils in AMG, each leading to different operator complexity (and thus extra memory)
    1) C1 coarsening, original Ruge-Stuben chapter https://epubs.siam.org/doi/10.1137/1.9781611971057.ch4, and discussed by hypre  https://epubs.siam.org/doi/epdf/10.1137/040615729
    2) H1 coarsening, see hypre  https://epubs.siam.org/doi/epdf/10.1137/040615729
    3) CR coarsening, https://www.sciencedirect.com/science/article/pii/S0096300320307487
# see also this paper from RN-AMG with some extra ideas, https://epubs.siam.org/doi/epdf/10.1137/16M1082706
"""

def strength_of_connections_csr(A_csr:sp.csr_matrix, threshold:float=0.25):
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

def RS_csr_coarsening(A_csr:sp.csr_matrix, threshold:float=0.25):
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

def standard_csr_coarsening(A_csr:sp.csr_matrix, threshold:float=0.25):
    """from multigrid book"""

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
        C[i] = True
        U[i] = False

        # step 3 : for all j in S_i^T and universe still, (candidate fine nodes) do:
        #  new F points from all strong neighbors unlabeled as C/F yet
        new_F_points = np.array([jj for jj in STRENGTH_TR[i] if U[jj]])
        for j in new_F_points:
            # step 4 add new F point to F set
            F[j] = True
            U[j] = False
            F_neighbors = np.array([kk for kk in STRENGTH[j] if U[kk]])
            LAM[F_neighbors] += 2 # increases by 2 here to promote standard coarsening (instead of 1 for Ruge-Stuben)
        
        # step 6: decrement LAM (num strong neighbors) for the added C point
        C_neighbors = np.array([jj for jj in STRENGTH[i] if U[jj]])
        if C_neighbors.shape[0] > 0:
            LAM[C_neighbors] -= 1

    # or just return bool masks..
    return C, F

def aggressive_A2_csr_coarsening(A_csr: sp.csr_matrix, threshold: float = 0.25):
    """
    Aggressive A2 coarsening with the requirement that for any second-level
    connection (C => F => C) there must be at least two distinct paths.
    
    The procedure is:
      1. Perform standard coarsening on A_csr to obtain coarse (C) and fine (F) nodes.
      2. Build second-level strength-of-connection maps (STRENGTH2 and STRENGTH_TR2)
         by considering only chains C -> F -> C that occur at least twice.
      3. Re-index the second-level maps to obtain restricted maps (RSTRENGTH2 and RSTRENGTH_TR2)
         for coarse nodes only.
      4. Apply a standard coarsening procedure on the restricted (coarse) nodes using
         the level-2 information.
      5. Map decisions back to the original global ordering.
    
    Parameters:
      A_csr : sp.csr_matrix
          Input matrix in CSR format.
      threshold : float
          Parameter for defining strength-of-connection.
    
    Returns:
      C, F : numpy.ndarray (bool)
          Updated Boolean masks for coarse and fine nodes in the original ordering.
    """
    # FIRST STAGE: Standard coarsening.
    C, F = standard_csr_coarsening(A_csr, threshold)
    nnodes = A_csr.shape[0]
    
    # Get first-level strength-of-connection info.
    STRENGTH, STRENGTH_TR = strength_of_connections_csr(A_csr, threshold)
    
    # Build second-level strength-of-connection maps for coarse nodes only.
    # Here we count the number of distinct (C=>F=>C) paths.
    STRENGTH2 = [[] for _ in range(nnodes)]
    STRENGTH_TR2 = [[] for _ in range(nnodes)]
    
    for i in range(nnodes):
        if not C[i]:
            continue  # only process coarse nodes
        
        # First hop (from i to fine nodes).
        fine_neighbors = [f for f in STRENGTH[i] if not C[f]]
        # Count second-level coarse nodes reached via these fine nodes.
        counter = {}
        for f in fine_neighbors:
            for j in STRENGTH[f]:
                if C[j]:
                    counter[j] = counter.get(j, 0) + 1
        # Keep only those coarse nodes reached by at least two distinct paths.
        STRENGTH2[i] = [j for j, count in counter.items() if count >= 2]
        
        # Do the same for the transpose.
        fine_neighbors_tr = [f for f in STRENGTH_TR[i] if not C[f]]
        counter_tr = {}
        for f in fine_neighbors_tr:
            for j in STRENGTH_TR[f]:
                if C[j]:
                    counter_tr[j] = counter_tr.get(j, 0) + 1
        STRENGTH_TR2[i] = [j for j, count in counter_tr.items() if count >= 2]
    
    # Restrict the level-2 arrays to the coarse nodes and re-index them.
    coarse_nodes = np.nonzero(C)[0]  # global indices of coarse nodes
    n_coarse = len(coarse_nodes)
    
    # Create a mapping from global coarse node index to new (local) ordering.
    coarse_map = {global_idx: local_idx for local_idx, global_idx in enumerate(coarse_nodes)}
    
    RSTRENGTH2 = []
    RSTRENGTH_TR2 = []
    
    for global_idx in coarse_nodes:
        # Convert each neighbor to its local index if it is coarse.
        local_neighbors = [coarse_map[j] for j in STRENGTH2[global_idx] if j in coarse_map]
        RSTRENGTH2.append(local_neighbors)
        
        local_neighbors_tr = [coarse_map[j] for j in STRENGTH_TR2[global_idx] if j in coarse_map]
        RSTRENGTH_TR2.append(local_neighbors_tr)
    
    # SECOND STAGE: Apply standard coarsening on the restricted coarse nodes.
    C2 = np.full(n_coarse, False)  # secondary coarse flags
    F2 = np.full(n_coarse, False)  # secondary fine flags
    U2 = np.full(n_coarse, True)   # universe (nodes not yet assigned)
    node_list = np.arange(n_coarse)
    LAM2 = np.array([len(neighbors) for neighbors in RSTRENGTH2])
    
    while np.any(U2):
        # Select node in U2 with maximum LAM2.
        U2_nodes = node_list[U2]
        i_local = U2_nodes[np.argmax(LAM2[U2])]
        if LAM2[i_local] == 0:
            print(f"ERROR: LAM2[{i_local}] is 0, but it was selected as coarse.")
            break
        C2[i_local] = True
        U2[i_local] = False
        
        # Mark level-2 transpose neighbors (still in U2) as fine.
        new_F_points = [j for j in RSTRENGTH_TR2[i_local] if U2[j]]
        for j in new_F_points:
            F2[j] = True
            U2[j] = False
            # Increase LAM2 for neighbors of the new fine node to promote coarse selection.
            for k in RSTRENGTH2[j]:
                if U2[k]:
                    LAM2[k] += 2
        # Decrement LAM2 for strong neighbors of the newly selected coarse node.
        for j in RSTRENGTH2[i_local]:
            if U2[j]:
                LAM2[j] -= 1
    
    # Map the secondary stage decisions back to the original (global) ordering.
    for local_idx, global_idx in enumerate(coarse_nodes):
        if F2[local_idx]:
            C[global_idx] = False
            F[global_idx] = True
    
    return C, F

def aggressive_A1_csr_coarsening(A_csr: sp.csr_matrix, threshold: float = 0.25):
    """
    Aggressive A2 coarsening by doing standard coarsening twice.
    
    First, a standard coarsening on A_csr yields coarse (C) and fine (F) sets.
    Then, we define second-level (length-2) strength-of-connection for coarse nodes,
    but only via a path from a coarse node to a fine neighbor, then to a coarse node.
    Finally, standard coarsening is performed on this reduced, re-indexed set.
    
    Parameters:
      A_csr : sp.csr_matrix
          Input matrix in CSR format.
      threshold : float
          Parameter for defining strength-of-connection.
    
    Returns:
      C, F : numpy.ndarray (bool)
          Updated Boolean masks for coarse and fine nodes in the original ordering.
    """
    # First stage: standard coarsening on the full matrix.
    C, F = standard_csr_coarsening(A_csr, threshold)
    nnodes = A_csr.shape[0]
    
    # Get first-level strength-of-connection information.
    # STRENGTH[i] holds indices j that are strongly connected to i.
    # STRENGTH_TR is the transpose version.
    STRENGTH, STRENGTH_TR = strength_of_connections_csr(A_csr, threshold)
    
    # Build the second-level strength-of-connection arrays only for coarse nodes.
    # Instead of taking coarse neighbors of coarse nodes directly,
    # we only allow chains going from a coarse node to a fine neighbor,
    # and then from that fine neighbor to a coarse node.
    STRENGTH2 = [[] for _ in range(nnodes)]
    STRENGTH_TR2 = [[] for _ in range(nnodes)]
    
    for i in range(nnodes):
        if not C[i]:
            continue  # only process coarse nodes
        
        # First hop: from i (coarse) to its fine neighbors (first-level).
        fine_neighbors = [j for j in STRENGTH[i] if not C[j]]
        sec_neighbors = set()
        # Second hop: from each fine neighbor f to its coarse neighbors.
        for f in fine_neighbors:
            sec_neighbors.update([j for j in STRENGTH[f] if C[j]])
        STRENGTH2[i] = list(sec_neighbors)
        
        # For the transpose version, use the same idea:
        fine_neighbors_tr = [j for j in STRENGTH_TR[i] if not C[j]]
        sec_neighbors_tr = set()
        for f in fine_neighbors_tr:
            sec_neighbors_tr.update([j for j in STRENGTH_TR[f] if C[j]])
        STRENGTH_TR2[i] = list(sec_neighbors_tr)
    
    # Restrict the level-2 arrays to the coarse nodes and re-index them.
    coarse_nodes = np.nonzero(C)[0]  # Global indices of coarse nodes.
    n_coarse = len(coarse_nodes)
    
    # Create a mapping from global coarse node index to a new (local) ordering.
    coarse_map = {global_idx: local_idx for local_idx, global_idx in enumerate(coarse_nodes)}
    
    RSTRENGTH2 = []
    RSTRENGTH_TR2 = []
    
    for global_idx in coarse_nodes:
        # Convert each neighbor to its local index if it is a coarse node.
        local_neighbors = [coarse_map[j] for j in STRENGTH2[global_idx] if j in coarse_map]
        RSTRENGTH2.append(local_neighbors)
        
        local_neighbors_tr = [coarse_map[j] for j in STRENGTH_TR2[global_idx] if j in coarse_map]
        RSTRENGTH_TR2.append(local_neighbors_tr)
    
    # SECOND STAGE: Apply standard coarsening on the restricted (coarse) nodes.
    C2 = np.full(n_coarse, False)  # secondary coarse flags
    F2 = np.full(n_coarse, False)  # secondary fine flags
    U2 = np.full(n_coarse, True)   # universe (nodes not yet assigned)
    node_list = np.arange(n_coarse)
    LAM2 = np.array([len(neighbors) for neighbors in RSTRENGTH2])
    
    while np.any(U2):
        # Select the node in U2 with the maximum LAM2.
        U2_nodes = node_list[U2]
        i_local = U2_nodes[np.argmax(LAM2[U2])]
        if LAM2[i_local] == 0:
            print(f"ERROR: LAM2[{i_local}] is 0, but it was selected as coarse.")
            break
        C2[i_local] = True
        U2[i_local] = False
        
        # For all nodes in the level-2 transpose neighbor set and still in U2, mark as fine.
        new_F_points = [j for j in RSTRENGTH_TR2[i_local] if U2[j]]
        for j in new_F_points:
            F2[j] = True
            U2[j] = False
            
            # Increase LAM2 in the neighborhood of the new fine node to promote coarse selection.
            for k in RSTRENGTH2[j]:
                if U2[k]:
                    LAM2[k] += 2
                    
        # Decrement LAM2 for the strong neighbors of the newly selected coarse node.
        for j in RSTRENGTH2[i_local]:
            if U2[j]:
                LAM2[j] -= 1
    
    # Map the secondary stage decisions back to the original ordering:
    # For each coarse node (global index) in the initial splitting,
    # if its secondary designation is fine, update the global masks.
    for local_idx, global_idx in enumerate(coarse_nodes):
        if F2[local_idx]:
            C[global_idx] = False
            F[global_idx] = True

    return C, F

def CLJP_coarsening(A_csr:sp.csr_matrix):
    # termed this in paper, https://epubs.siam.org/doi/epdf/10.1137/040615729
    # but algorithm of CFJP found in https://www.researchgate.net/publication/223921531_BoomerAMG_A_parallel_algebraic_multigrid_solver_and_preconditioner
    pass

def PMIS_csr_coarsening(A_csr:sp.csr_matrix):
    # see hypre paper, https://epubs.siam.org/doi/epdf/10.1137/040615729
    # lower operator complexity, worse residual reduction per Krylov step than CLJP C1 method
    pass

def HMIS_csr_coarsening(A_csr:sp.csr_matrix):
    # see hypre paper, https://epubs.siam.org/doi/epdf/10.1137/040615729
    # lower operator complexity, worse residual reduction per Krylov step than CLJP C1 method
    pass

def Falgout_csr_coarsening(A_csr:sp.csr_matrix):
    # not sure which paper defined in, but is discussed / used in this paper of hypre, 
    pass

def CR_csr_coarsening(A_csr:sp.csr_matrix):
    # see paper, https://www.sciencedirect.com/science/article/pii/S0096300320307487
    # best of both worlds of C1 and H1 methods, medium operator complexity, better conv.. is goal
    pass