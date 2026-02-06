import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

from mpi4py import MPI
from tacs import TACS, elements, constitutive


def get_tacs_matrix(bdf_file, E:float, nu:float, load_fcn, thickness:float=0.02):

    # Load structural mesh from BDF file
    tacs_comm = MPI.COMM_WORLD
    struct_mesh = TACS.MeshLoader(tacs_comm)
    struct_mesh.scanBDFFile(bdf_file)

    # Set constitutive properties
    rho = 2500.0  # density, kg/m^3
    # E = 70e9  # elastic modulus, Pa
    # nu = 0.3  # poisson's ratio
    kcorr = 5.0 / 6.0  # shear correction factor
    ys = 350e6  # yield stress, Pa
    min_thickness = 0.002
    max_thickness = 10.0 # 0.2
    # thickness = 0.02

    # Loop over components, creating stiffness and element object for each
    num_components = struct_mesh.getNumComponents()
    for i in range(num_components):
        descriptor = struct_mesh.getElementDescript(i)
        # Setup (isotropic) property and constitutive objects
        prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)
        # Set one thickness dv for every component
        stiff = constitutive.IsoShellConstitutive(
            prop, t=thickness, tMin=min_thickness, tMax=max_thickness, tNum=i
        )

        element = None
        transform = None
        if descriptor in ["CQUAD", "CQUADR", "CQUAD4"]:
            element = elements.Quad4Shell(transform, stiff)
        struct_mesh.setElement(i, element)

    # Create tacs assembler object from mesh loader
    tacs = struct_mesh.createTACS(6)

    # Set up and solve the analysis problem
    res = tacs.createVec()
    mat = tacs.createSchurMat(TACS.NATURAL_ORDER)

    xpts_vec = tacs.createNodeVec()
    tacs.getNodes(xpts_vec)
    xpts_arr = xpts_vec.getArray()
    nnodes = xpts_arr.shape[0] // 3

    # Create the forces
    forces = tacs.createVec()
    force_array = forces.getArray()
    # force_array[2::6] += 1.0  # uniform load in z direction
    x = xpts_arr[0::3]
    y = xpts_arr[1::3]
    # r = np.sqrt(x**2 + y**2)
    # force_array[2::6] += 100.0 # simple loading first
    
    # def load_fcn(_x,_y):
    #     import math
    #     theta = math.atan2(_y, _x)
    #     r = np.sqrt(_x**2 + _y**2)
    #     r2 = r/np.sqrt(2)
    #     # game of life polar load..
    #     # return 100.0 * np.sin(5.0  * np.pi * r) * np.cos(4*theta) * r2 * (1 - r2)
    #     return -100.0 * np.sin(4.0  * np.pi * r) * np.cos(2.4*theta-np.pi/4.0*2.4) * r2 * (1 - r2)
    #     # return 100.0 * np.sin(5.0  * np.pi * r) * np.cos(2*theta-np.pi/2.0)
    #     # return 100.0 * np.sin(2.0 * np.pi * _x) * np.sin(np.pi * _y)

    nxe = int(np.sqrt(nnodes)) - 1
    int_nnodes = (nxe-1)**2
    force_array[2::6] = np.array([load_fcn(x[i], y[i]) / int_nnodes for i in range(nnodes)])
    # force_array[2::6] += np.sin(2.0 * np.pi * x) * np.sin(np.pi * y)

    # force_array[2::6] += 100.0 * np.sin(3 * np.pi * x)
    # force_array[2::6] += 100.0 * np.sin(3 * np.pi * r)
    tacs.applyBCs(forces)

    # Assemble the Jacobian and factor
    alpha = 1.0
    beta = 0.0
    gamma = 0.0
    tacs.zeroVariables()
    tacs.assembleJacobian(alpha, beta, gamma, res, mat)
    # tacs.applyBCs(res, mat)

    data = mat.getMat()
    # print(f"{data=}")
    A_bsr = data[0]
    # print(f"{A_bsr=} {type(A_bsr)=}")

    # TODO : just get serial A part in a minute
    return A_bsr, force_array, xpts_arr


class TACSAssembler:
    """
    Full-system TACS assembler:
      - Builds full K and full f from TACS (no DOF elimination here)
      - Reorders nodes to structured order: x primary, y secondary
      - Provides MG prolongate/restrict_defect using P2=kron(P1,P1) applied per dof-slice
      - dof_per_node = 6
    Constructor signature matches StandardPlateAssembler + requires bdf_file.
    """

    def __init__(
        self,
        ELEMENT,                 # unused (kept for compatibility)
        nxe: int,
        E: float = 70e9,         # unused (TACS sets internally in your get_tacs_matrix)
        nu: float = 0.3,         # unused
        thick: float = 1.0e-2,
        length: float = 1.0,     # unused (BDF defines geometry)
        width: float = 1.0,      # unused
        load_fcn=lambda x, y: 1.0,   # unused unless you edit get_tacs_matrix
        clamped: bool = False,       # unused here; TACS BCs come from BDF/applyBCs
        split_disp_bc: bool = False, # unused
        bdf_file: str = None,
    ):
        if bdf_file is None:
            raise ValueError("TACSAssembler requires bdf_file=...")

        self.element = ELEMENT  # unused
        self.E = E
        self.nu = nu
        self.nxe = int(nxe)
        self.thick = float(thick)
        self.bdf_file = bdf_file
        self.load_fcn = load_fcn

        self.dof_per_node = 6
        self.nnx = self.nxe + 1
        self.nnodes = self.nnx * self.nnx
        self.N = self.dof_per_node * self.nnodes

        self.kmat = None
        self.force = None
        self.u = None

        # node reordering maps (unsorted <-> sorted)
        self.node_sort_map = None      # sorted_index -> unsorted_index
        self.inv_node_sort_map = None  # unsorted_index -> sorted_index

        # MG caches
        self._P1_cache = {}
        self._P2_cache = {}

        self.xpts_unsorted = None
        self.xpts_sorted = None

        self._assemble_system()

    # ----------------------------
    # Sorting utilities
    # ----------------------------
    def _build_node_sort(self, xpts_unsorted: np.ndarray):
        """
        Build node permutation such that nodes are ordered by x primary, y secondary.
        Returns:
          node_sort_map: array of length Nnodes, where sorted_node_id -> unsorted_node_id
          inv_node_sort_map: unsorted_node_id -> sorted_node_id
        """
        x = xpts_unsorted[0::3]
        y = xpts_unsorted[1::3]

        # x primary, y secondary
        node_sort_map = np.lexsort((y, x)).astype(np.int32)

        inv = np.empty_like(node_sort_map)
        inv[node_sort_map] = np.arange(node_sort_map.size, dtype=np.int32)

        return node_sort_map, inv

    def _permute_full_vec_nodes(self, v_full_unsorted: np.ndarray, dpn: int) -> np.ndarray:
        """
        Permute a full vector of size dpn*Nnodes from UNSORTED node order to SORTED node order.
        """
        Nnodes = self.nnodes
        assert v_full_unsorted.size == dpn * Nnodes
        V = v_full_unsorted.reshape((Nnodes, dpn))
        Vs = V[self.node_sort_map, :]
        return Vs.reshape(-1)

    def _unpermute_full_vec_nodes(self, v_full_sorted: np.ndarray, dpn: int) -> np.ndarray:
        """
        Permute a full vector of size dpn*Nnodes from SORTED node order to UNSORTED node order.
        """
        Nnodes = self.nnodes
        assert v_full_sorted.size == dpn * Nnodes
        V = v_full_sorted.reshape((Nnodes, dpn))
        Vu = V[self.inv_node_sort_map, :]
        return Vu.reshape(-1)

    def _permute_full_mat_nodes(self, A_unsorted: sp.spmatrix, dpn: int) -> sp.csr_matrix:
        """
        Permute a full matrix from unsorted node ordering to sorted node ordering.
        The permutation is at *node* level, but applied to scalar DOFs.

        If Pnode is the permutation on nodes, then scalar permutation is:
          P = kron(Pnode, I_dpn)
          A_sorted = P * A_unsorted * P^T
        """
        Nnodes = self.nnodes
        assert A_unsorted.shape == (dpn * Nnodes, dpn * Nnodes)

        # Build scalar permutation indices directly (no big kron matrix)
        # sorted scalar dof index = dpn*sorted_node + a
        # unsorted scalar dof index = dpn*unsorted_node + a
        perm = np.empty(dpn * Nnodes, dtype=np.int32)
        for s in range(Nnodes):
            u = int(self.node_sort_map[s])
            base_s = dpn * s
            base_u = dpn * u
            perm[base_s:base_s + dpn] = np.arange(base_u, base_u + dpn, dtype=np.int32)

        # A_sorted = A_unsorted[perm, perm]
        A_sorted = A_unsorted.tocsr()[perm, :][:, perm].tocsr()
        return A_sorted

    # ----------------------------
    # TACS assembly (full)
    # ----------------------------
    def _assemble_system(self):
        # You provide this function; it returns full matrix + full force + xpts
        A_bsr, force_full_unsorted, xpts_unsorted = get_tacs_matrix(
            self.bdf_file, thickness=self.thick, E=self.E, nu=self.nu, load_fcn=self.load_fcn
        )

        if not sp.issparse(A_bsr):
            raise TypeError(f"Expected SciPy sparse for A_bsr, got {type(A_bsr)}")

        self.xpts_unsorted = xpts_unsorted

        # sanity on size
        nnodes = (xpts_unsorted.size // 3)
        if nnodes != self.nnodes:
            raise ValueError(
                f"BDF nnodes={nnodes}, but nxe={self.nxe} implies nnodes={(self.nxe+1)**2}"
            )
        if force_full_unsorted.size != self.N:
            raise ValueError(f"force size {force_full_unsorted.size} != 6*nnodes {self.N}")
        if A_bsr.shape != (self.N, self.N):
            raise ValueError(f"A shape {A_bsr.shape} != ({self.N},{self.N})")

        # build node sorting maps
        self.node_sort_map, self.inv_node_sort_map = self._build_node_sort(xpts_unsorted)

        # permute xpts to sorted order (for plotting etc.)
        X = xpts_unsorted.reshape((self.nnodes, 3))
        self.xpts_sorted = X[self.node_sort_map, :].reshape(-1)

        # permute full force and full matrix to sorted ordering
        f_sorted = self._permute_full_vec_nodes(force_full_unsorted, dpn=6)
        A_sorted = self._permute_full_mat_nodes(A_bsr, dpn=6)

        # store
        self.force = f_sorted
        self.kmat = A_sorted.tobsr(blocksize=(6, 6))  # keep block structure

    def direct_solve(self):
        self.u = spla.spsolve(self.kmat.tocsc(), self.force)
        return self.u

    def plot_disp(self):
        """
        Plot uz (index 2) in sorted node order on structured grid.
        """
        if self.u is None:
            raise RuntimeError("Run direct_solve() first.")

        uz = self.u[2::6].reshape((self.nnx, self.nnx))

        xs = self.xpts_sorted[0::3].reshape((self.nnx, self.nnx))
        ys = self.xpts_sorted[1::3].reshape((self.nnx, self.nnx))

        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(xs, ys, uz, cmap="viridis", linewidth=0, antialiased=True, shade=True)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("uz")
        ax.view_init(elev=25, azim=-135)
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08, label="uz")
        plt.tight_layout()
        plt.show()

    # ----------------------------
    # MG prolongation/restriction (dpn=6), no element class
    # ----------------------------
    def _build_P1_scalar(self, nxe_coarse: int) -> sp.csr_matrix:
        if nxe_coarse in self._P1_cache:
            return self._P1_cache[nxe_coarse]

        nc = nxe_coarse + 1
        nf = 2 * nxe_coarse + 1  # = 2*nc - 1

        rows = []
        cols = []
        vals = []

        # even fine nodes copy coarse
        for i in range(nc):
            rows.append(2 * i)
            cols.append(i)
            vals.append(1.0)

        # odd fine nodes average
        for i in range(nc - 1):
            r = 2 * i + 1
            rows += [r, r]
            cols += [i, i + 1]
            vals += [0.5, 0.5]

        P1 = sp.coo_matrix((vals, (rows, cols)), shape=(nf, nc)).tocsr()
        self._P1_cache[nxe_coarse] = P1
        return P1

    def _build_P2_scalar(self, nxe_coarse: int) -> sp.csr_matrix:
        if nxe_coarse in self._P2_cache:
            return self._P2_cache[nxe_coarse]
        P1 = self._build_P1_scalar(nxe_coarse)
        P2 = sp.kron(P1, P1, format="csr")
        self._P2_cache[nxe_coarse] = P2
        return P2
    
    def apply_bcs_2d(self, u: np.ndarray, nxe: int):
        nx = nxe + 1
        U = u.reshape((nx * nx, 6))

        for j in range(nx):
            for i in range(nx):
                on_edge = (i == 0) or (i == nx - 1) or (j == 0) or (j == nx - 1)
                if not on_edge:
                    continue
                k = i + nx * j
                U[k, 2] = 0.0
                # if self.clamped:
                #     U[k, 1] = 0.0
                #     U[k, 2] = 0.0

    def prolongate(self, coarse_soln: np.ndarray):
        """
        Prolongate from coarse grid (nxe/2) to this grid (nxe).
        Assumes coarse_soln is a FULL vector in sorted ordering with dpn=6.
        """
        dpn = 6
        nxe_f = self.nxe
        nxe_c = nxe_f // 2

        nxc = nxe_c + 1
        Nc = nxc * nxc
        assert coarse_soln.size == dpn * Nc, f"expected {dpn*Nc}, got {coarse_soln.size}"

        nxf = nxe_f + 1
        Nf = nxf * nxf
        P2 = self._build_P2_scalar(nxe_c)

        fine = np.zeros(dpn * Nf, dtype=float)
        for a in range(dpn):
            fine[a::dpn] = P2 @ coarse_soln[a::dpn]

        self.apply_bcs_2d(fine, nxe_f)
        return fine

    def restrict_defect(self, fine_defect: np.ndarray):
        """
        Restrict from this grid (nxe) to coarse grid (nxe/2) using R = P^T.
        Assumes fine_defect is a FULL vector in sorted ordering with dpn=6.
        """
        # print(f"{fine_defect.shape=} {}")

        dpn = 6
        nxe_f = self.nxe * 2
        nxf = nxe_f + 1
        Nf = nxf * nxf
        assert fine_defect.size == dpn * Nf, f"expected {dpn*Nf}, got {fine_defect.size}"

        nxe_c = nxe_f // 2
        nxc = nxe_c + 1
        Nc = nxc * nxc

        P2 = self._build_P2_scalar(nxe_c)
        R2 = P2.T

        coarse = np.zeros(dpn * Nc, dtype=float)
        for a in range(dpn):
            coarse[a::dpn] = R2 @ fine_defect[a::dpn]
        
        self.apply_bcs_2d(coarse, nxe_c)
        return coarse
