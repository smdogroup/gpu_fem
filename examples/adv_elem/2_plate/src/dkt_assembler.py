import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

class DKTPlateAssembler:
    def __init__(
        self,
        ELEMENT,
        nxe:int,
        E:float=70e9,
        nu:float=0.3,
        thick:float=1.0e-2,
        length:float=1.0,
        width:float=1.0,
        load_fcn=lambda x, y: 1.0,
        clamped:bool=False,
        split_disp_bc:bool=False,
        bdf_file:str=""
    ):
        
        # FOR NOW..
        assert length == width
        assert not clamped

        self.element = ELEMENT
        self.nxe = int(nxe)
        self.E = E
        self.nu = nu
        self.thick = thick
        self.length = length
        self.width = width
        self.load_fcn = load_fcn
        self.split_disp_bc = split_disp_bc

        self.kmat = None
        self.force = None
        self.u = None

        self.dx = self.length / self.nxe

        self.dof_per_node = self.element.dof_per_node
        self.nnx = nxe + 1
        self.nnodes = self.nnx**2
        self.ndof = 3 * self.nnodes
        self.nelems = nxe**2 * 2

        # uniform grid spacing
        self.dx = self.length / self.nxe
        self.dy = self.width / self.nxe


    def _assemble_stiffness_matrix_csr(self, nxe, E, thick, nu):
        """Assemble global stiffness matrix (CSR) for unit-square regular grid with DKT triangles."""
        nx = nxe + 1
        N = nx**2
        ndof = 3 * N
        nelems = nxe**2 * 2
        h = self.length / nxe

        # Each triangle contributes a 9x9 matrix => 81 entries
        nnz_per_elem = 81
        I = np.empty(nelems * nnz_per_elem, dtype=np.int32)
        J = np.empty(nelems * nnz_per_elem, dtype=np.int32)
        V = np.empty(nelems * nnz_per_elem, dtype=np.float64)

        p = 0  # write pointer into I,J,V

        for ielem in range(nelems):
            iquad = ielem // 2
            itri = ielem % 2
            ixe = iquad % nxe
            iye = iquad // nxe

            x1 = h * ixe
            x2 = x1 + h
            y1 = h * iye
            y2 = y1 + h
            n1 = nx * iye + ixe
            n3 = n1 + nx
            n2, n4 = n1 + 1, n3 + 1

            if itri == 0:
                x_elem = np.array([x1, x2, x1], dtype=float)
                y_elem = np.array([y1, y1, y2], dtype=float)
                local_nodes = (n1, n2, n3)
            else:
                x_elem = np.array([x2, x1, x2], dtype=float)
                y_elem = np.array([y2, y2, y1], dtype=float)
                local_nodes = (n4, n3, n2)

            Kelem = self.element.get_kelem(E, thick, nu, x_elem, y_elem)  # (9,9)

            # global dof indices for this tri (9,)
            dofs = np.array([3*inode + idof for inode in local_nodes for idof in range(3)],
                            dtype=np.int32)

            # fill COO triplets (vectorized)
            # rows: repeat dofs for each column, cols: tile dofs for each row
            rr = np.repeat(dofs, 9)
            cc = np.tile(dofs, 9)

            I[p:p+81] = rr
            J[p:p+81] = cc
            V[p:p+81] = Kelem.reshape(-1)
            p += 81

        # Build sparse matrix and sum duplicates automatically
        K = sp.coo_matrix((V, (I, J)), shape=(ndof, ndof)).tocsr()

        return K
    
    def _assemble_stiffness_matrix_bsr(self, nxe, E, thick, nu):
        """Assemble global stiffness matrix as BSR with 3x3 blocks per node (DKT triangles)."""
        K_csr = self._assemble_stiffness_matrix_csr(nxe, E, thick, nu)
        K_bsr = K_csr.tobsr(blocksize=(3,3))
        return K_bsr

    def _apply_bcs_helper(self, bcs, K, F):
        K[bcs,:] = 0.0
        K[:,bcs] = 0.0
        for bc in bcs:
            K[bc,bc] = 1.0
        F[bcs] = 0.0
        return K, F

    def apply_bcs(self, nxe, K, F):
        """apply bcs to the global stiffness matrix and forces"""
        bcs = self.element._get_bcs(nxe)
        return self._apply_bcs_helper(bcs, K, F)
    
    def _assemble_system(self, bcs:bool=True):
        
        self.kmat = self._assemble_stiffness_matrix_csr(self.nxe, self.E, self.thick, self.nu)
        # self.kmat = self._assemble_stiffness_matrix_bsr(self.nxe, self.E, self.thick, self.nu)
        self.force = np.zeros(self.ndof)
        
        # distr load
        self.int_nnodes = (self.nxe-1)**2
        self.force[0::3] += self.load_fcn(0.5, 0.5) / self.int_nnodes * 1.9 # assumes const load here

        if bcs:  self.kmat, self.force = self.apply_bcs(self.nxe, self.kmat, self.force)
        self.kmat = self.kmat.tobsr(blocksize=(3,3))
        return
    
    def get_xpts(self) -> np.ndarray:
        """
        Return global nodal coordinates as a flat (3*nnodes,) array:

            [x1, y1, z1,  x2, y2, z2,  ...]

        Structured Q1 grid on [0,length] x [0,width], z = 0.
        Node ordering matches assembler connectivity:
            inode = ix + nnx * iy
        """
        xyz = np.zeros(3 * self.nnodes, dtype=np.double)

        for inode in range(self.nnodes):
            ix = inode % self.nnx
            iy = inode // self.nnx

            x = ix * self.dx
            y = iy * self.dy
            z = 0.0

            xyz[3*inode + 0] = x
            xyz[3*inode + 1] = y
            xyz[3*inode + 2] = z

        return xyz
    
    def direct_solve(self):
        self._assemble_system()
        self.u = sp.linalg.spsolve(self.kmat.tocsc(), self.force)
        # self.u = np.linalg.solve(self.kmat.toarray(), self.force)

        # kmat_inv = np.linalg.inv(self.kmat.toarray())
        # plt.imshow(kmat_inv)
        # plt.show()

        # system is not being solved correctly..
        # print(f"{self.u=}")
        # resid = self.kmat.dot(self.u) - self.force
        # print(f"{resid=}")
        return self.u

    def plot_disp(self):
        """3D surface plot of w(x,y) on the CONTROL grid (debug)."""
        if self.u is None:
            raise RuntimeError("Run direct_solve() first.")

        dpn = self.dof_per_node
        U = self.u.reshape((self.nnodes, dpn))
        w = U[:,0]

        # IMPORTANT: inode = ix + nnx*iy  -> W[iy, ix]
        W = w.reshape((self.nnx, self.nnx))  # row=iy, col=ix

        wmin = float(W.min())
        wmax = float(W.max())
        # print(f"w range: [{wmin:.6e}, {wmax:.6e}], ptp={wmax-wmin:.6e}")

        # physical coordinates consistent with how you build elem_xpts
        x = np.arange(self.nnx) * self.dx
        y = np.arange(self.nnx) * self.dx
        X, Y = np.meshgrid(x, y, indexing="xy")   # shapes (ny, nx)

        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(
            X, Y, W,   # W must be (ny, nx) to match X,Y
            cmap="viridis",
            linewidth=0,
            antialiased=True,
            shade=True
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("w")
        ax.view_init(elev=25, azim=-135)

        # Don’t let aspect ratio blow up if deflection is tiny
        zrange = max(1e-14, wmax - wmin)
        # ax.set_box_aspect((self.length, self.width, zrange))

        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08, label="w")
        plt.tight_layout()
        plt.show()



    def prolongate(self, coarse_soln):
        return self.element.prolongate(coarse_soln, self.nxe // 2)
    
    def restrict_defect(self, fine_defect):
        return self.element.restrict_defect(fine_defect, self.nxe)