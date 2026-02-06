import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
from _sparse_utils import build_csr_from_conn

class StandardPlateAssembler:
    """
    Structured-grid assembler for standard C0 plate elements on a unit square mesh.
    Intended for Q1 Reissner–Mindlin (4-node quad), dof per node = 3: [w, thx, thy].

    Same constructor signature as IGAPlateAssembler.
    """

    def __init__(
        self,
        ELEMENT,
        nxe: int,
        E: float = 70e9,
        nu: float = 0.3,
        thick: float = 1.0e-2,
        length: float = 1.0,
        width: float = 1.0,
        load_fcn=lambda x, y: 1.0,
        clamped: bool = False,
        split_disp_bc: bool = False,
        bdf_file:str=""
    ):
        self.element = ELEMENT
        self.nxe = int(nxe)
        self.E = E
        self.nu = nu
        self.thick = thick
        self.length = length
        self.width = width
        self.load_fcn = load_fcn
        self.split_disp_bc = split_disp_bc

        self.element.clamped = clamped

        self.kmat = None
        self.force = None
        self.u = None

        # ---- element expectations for RM/Q1 ----
        # if you keep element.ORDER, use it; otherwise just assume Q1
        if hasattr(self.element, "ORDER"):
            assert self.element.ORDER == 1, "StandardPlateAssembler is for ORDER=1 (Q1) elements."
        assert self.element.dof_per_node == 3, "RM plate here expects dof_per_node=3 (w,thx,thy)."
        assert self.element.nodes_per_elem == 4, "RM/Q1 expects 4-node quad."

        self.nnx = self.nxe + 1
        self.nnodes = self.nnx * self.nnx
        self.dof_per_node = self.element.dof_per_node
        self.N = self.dof_per_node * self.nnodes
        self.num_elements = self.nxe * self.nxe

        # uniform grid spacing
        self.dx = self.length / self.nxe
        self.dy = self.width / self.nxe

        self.clamped = bool(clamped)
        self.element.clamped = self.clamped  # if element uses it

        # boundary node list
        self.bcs = []
        for inode in range(self.nnodes):
            ix = inode % self.nnx
            iy = inode // self.nnx
            if ix == 0 or iy == 0 or ix == self.nnx - 1 or iy == self.nnx - 1:
                self.bcs.append(inode)

        # ---- ELEMENT connectivity: 4 nodes per quad ----
        # node layout per element (Q1):
        # 0:(ex,ey), 1:(ex+1,ey), 2:(ex+1,ey+1), 3:(ex,ey+1)
        self.conn = []
        for ey in range(self.nxe):
            for ex in range(self.nxe):
                n0 = ex + self.nnx * ey
                n1 = (ex + 1) + self.nnx * ey
                n2 = (ex + 1) + self.nnx * (ey + 1)
                n3 = ex + self.nnx * (ey + 1)
                self.conn.append(np.array([n0, n1, n2, n3], dtype=int))

        # DOF connectivity (4*dpn = 12 dof per elem)
        dpn = self.dof_per_node
        self.dof_conn = []
        for loc in self.conn:
            dofs = np.empty((4 * dpn,), dtype=int)
            k = 0
            for n in loc:
                base = dpn * n
                for a in range(dpn):
                    dofs[k] = base + a
                    k += 1
            self.dof_conn.append(dofs)

        # CSR pattern at NODE level, then we store BSR blocks (dpn x dpn) per node-node
        self.rowp, self.cols, self.nnzb = build_csr_from_conn(self.conn, self.nnodes)

    def _elem_xpts_from_loc(self, loc_conn: np.ndarray):
        """Return elem_xpts of length 12: [x,y,0] per local node, consistent with element node order."""
        elem_xpts = np.zeros(12, dtype=np.double)
        for lnode, gnode in enumerate(loc_conn):
            ix = gnode % self.nnx
            iy = gnode // self.nnx
            elem_xpts[3*lnode + 0] = ix * self.dx
            elem_xpts[3*lnode + 1] = iy * self.dy
            elem_xpts[3*lnode + 2] = 0.0
        return elem_xpts

    def _apply_bcs(self, dpn: int):
        """
        RM-friendly BCs.
        - split_disp_bc is not supported here.
        - simply-supported: w=0 on boundary
        - clamped: w=thx=thy=0 on boundary
        """
        if self.split_disp_bc:
            raise NotImplementedError("split_disp_bc not for this plate; not used for RM/Q1 StandardPlateAssembler.")

        if self.clamped:
            idofs = [0, 1, 2]
        else:
            idofs = [0]  # w only

        for node in self.bcs:
            for idof in idofs:
                # zero the selected row (block-row = node, row within block = idof)
                for colp in range(self.rowp[node], self.rowp[node + 1]):
                    bc = self.cols[colp]
                    self.data[colp, idof, :] = 0.0
                    if bc == node:
                        self.data[colp, idof, idof] = 1.0
                self.force[dpn * node + idof] = 0.0

    def _assemble_system(self, bcs:bool=True):
        dpn = self.dof_per_node
        self.data = np.zeros((self.nnzb, dpn, dpn), dtype=np.double)
        self.force = np.zeros(self.N, dtype=np.double)

        for ielem in range(self.num_elements):
            loc_conn = self.conn[ielem]
            loc_dofs = self.dof_conn[ielem]
            elem_xpts = self._elem_xpts_from_loc(loc_conn)

            kelem = self.element.get_kelem(self.E, self.nu, self.thick, elem_xpts)   # (12,12)
            felem = self.element.get_felem(mag=self.load_fcn, elem_xpts=elem_xpts)   # (12,)

            # LHS scatter into block-CSR (node connectivity -> BSR data)
            for lbr, br in enumerate(loc_conn):
                for colp in range(self.rowp[br], self.rowp[br + 1]):
                    bc = self.cols[colp]
                    hit = np.where(loc_conn == bc)[0]
                    if hit.size == 0:
                        continue
                    lbc = int(hit[0])
                    self.data[colp, :, :] += kelem[
                        dpn*lbr:dpn*(lbr+1),
                        dpn*lbc:dpn*(lbc+1)
                    ]

            # RHS scatter (dof-level)
            np.add.at(self.force, loc_dofs, felem)

        # BCs
        if bcs: self._apply_bcs(dpn)

        # Build global BSR
        self.kmat = sp.bsr_matrix((self.data, self.cols, self.rowp), shape=(self.N, self.N))

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
        return self.u

    def plot_disp(self):
        if self.u is None:
            raise RuntimeError("Run direct_solve() first.")

        dpn = self.dof_per_node
        U = self.u.reshape((self.nnodes, dpn))
        w = U[:, 0]

        W = w.reshape((self.nnx, self.nnx))
        x = np.arange(self.nnx) * self.dx
        y = np.arange(self.nnx) * self.dy
        X, Y = np.meshgrid(x, y, indexing="xy")

        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, W, cmap="viridis", linewidth=0, antialiased=True, shade=True)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("w")
        ax.view_init(elev=25, azim=-135)
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08, label="w")
        plt.tight_layout()
        plt.show()

    # ------------------------
    # Multigrid hooks
    # ------------------------
    def prolongate(self, coarse_soln):
        return self.element.prolongate(coarse_soln, self.nxe // 2)
    
    def restrict_defect(self, fine_defect):
        return self.element.restrict_defect(fine_defect, self.nxe * 2)
    
