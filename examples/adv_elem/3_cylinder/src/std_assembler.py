import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../2_plate/src/")
from _sparse_utils import build_csr_from_conn

class StandardCylinderAssembler:
    """
    Structured-grid assembler for standard C0 shell elements on a unit square mesh.
    Intended for Q1 Reissner–Mindlin (4-node quad), dof per node = 6: [u, v, w, thx, thy, thz].

    It's a partial cylindrical shell (not )
    """

    def __init__(
        self,
        ELEMENT,
        nxe: int,
        E: float = 70e9,
        nu: float = 0.3,
        thick: float = 1.0e-2,
        length: float = 1.0,
        radius: float = 1.0,
        hoop_length: float = np.pi,
        load_fcn=lambda x, y, z: 1.0,
        clamped: bool = False,
        split_disp_bc: bool = False,
    ):
        self.element = ELEMENT
        self.nxe = int(nxe)
        self.E = E
        self.nu = nu
        self.thick = thick
        self.length = length
        self.radius = float(radius)
        self.Ly = float(hoop_length)
        self.load_fcn = load_fcn
        self.split_disp_bc = split_disp_bc

        self.element.clamped = clamped
        self.dof_per_node = 6

        self.kmat = None
        self.force = None
        self.u = None

        # ---- element expectations for RM/Q1 ----
        # if you keep element.ORDER, use it; otherwise just assume Q1
        if hasattr(self.element, "ORDER"):
            assert self.element.ORDER == 1, "StandardCylinderAssembler is for ORDER=1 (Q1) elements."
        assert self.element.dof_per_node == 6, "RM cylinder here expects dof_per_node=6 (u,v,w,thx,thy,thz)."
        assert self.element.nodes_per_elem == 4, "RM/Q1 expects 4-node quad."

        self.nnx = self.nxe + 1
        self.nnodes = self.nnx * self.nnx
        self.dof_per_node = self.element.dof_per_node
        self.N = self.dof_per_node * self.nnodes
        self.num_elements = self.nxe * self.nxe

        # uniform grid spacing
        self.dx = self.length / self.nxe
        self.dy = self.Ly / self.nxe
        self.dth = self.dy / self.radius

        self.clamped = bool(clamped)
        self.element.clamped = self.clamped  # if element uses it

        # boundary node list
        # -----------------------------
        # BCs as DOF indices (not nodes)
        # -----------------------------
        dpn = self.dof_per_node
        bc_dofs = set()

        for inode in range(self.nnodes):
            ix = inode % self.nnx
            iy = inode // self.nnx

            on_xmin = (ix == 0)                 # x = 0 edge
            on_thmin = (iy == 0)                # min-theta edge
            on_bnd = (ix == 0 or iy == 0 or ix == self.nnx - 1 or iy == self.nnx - 1)

            if not on_bnd:
                continue

            base = dpn * inode

            if self.clamped:
                # clamp all 6 dofs on all boundary edges
                for a in range(dpn):
                    bc_dofs.add(base + a)
            else:
                # SS rules:
                # 1) w = 0 on all boundary edges
                bc_dofs.add(base + 2)   # w

                # 2) u = 0 on x = 0 edge
                if on_xmin:
                    bc_dofs.add(base + 0)  # u

                # 3) v = 0 on min-theta edge
                if on_thmin:
                    bc_dofs.add(base + 1)  # v

        # store as sorted list for reproducibility
        self.bcs = sorted(bc_dofs)

        # ---- ELEMENT connectivity: 4 nodes per quad ----
        # node layout per element (Q1):
        # 0:(ex,ey), 1:(ex+1,ey), 2:(ex,ey+1), 3:(ex+1,ey+1)
        self.conn = []
        for ey in range(self.nxe):
            for ex in range(self.nxe):
                n0 = ex + self.nnx * ey
                n1 = (ex + 1) + self.nnx * ey
                n2 = ex + self.nnx * (ey + 1)
                n3 = (ex + 1) + self.nnx * (ey + 1)
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
            phi = iy * self.dth
            elem_xpts[3*lnode + 0] = ix * self.dx
            elem_xpts[3*lnode + 1] = -self.radius * np.cos(phi)
            elem_xpts[3*lnode + 2] = self.radius * np.sin(phi)
        return elem_xpts

    def _apply_bcs(self, dpn: int):
        """
        Apply BCs given as global DOF indices in self.bcs.

        SS:
          - u=0 on x=0
          - v=0 on min-theta edge
          - w=0 on all boundary edges
        Clamped:
          - u=v=w=thx=thy=thz=0 on all boundary edges
        """
        if self.split_disp_bc:
            raise NotImplementedError("split_disp_bc not supported for this cylinder assembler.")

        for gdof in self.bcs:
            node = gdof // dpn
            idof = gdof - dpn * node

            # zero the selected row (block-row = node, row within block = idof)
            for colp in range(self.rowp[node], self.rowp[node + 1]):
                bc = self.cols[colp]
                self.data[colp, idof, :] = 0.0
                if bc == node:
                    self.data[colp, idof, idof] = 1.0

            self.force[gdof] = 0.0

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

    def direct_solve(self):
        self._assemble_system()
        self.u = sp.linalg.spsolve(self.kmat.tocsc(), self.force)
        return self.u

    def plot_disp(self, disp_mag:float, mode:str):
        import matplotlib.colors as mcolors
        import matplotlib.cm as cm
            
        if self.u is None:
            raise RuntimeError("Run direct_solve() first.")

        dpn = self.dof_per_node
        U = self.u.reshape((self.nnodes, dpn))
        w = U[:, 2]

        W = w.reshape((self.nnx, self.nnx))
        x = np.arange(self.nnx) * self.dx
        s = np.arange(self.nnx) * self.dy
        # s = np.linspace(-self.Ly, 0.0, self.nnx)
        phi = s / self.radius
        X, Phi = np.meshgrid(x, phi, indexing="xy")
        Y = (self.radius + W) * -np.cos(Phi)
        Z = (self.radius + W) * np.sin(Phi)

        # fig = plt.figure(figsize=(9, 6))
        # ax = fig.add_subplot(111, projection="3d")
        # surf = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True, shade=True)

        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # ax.set_zlabel("w")
        # ax.view_init(elev=25, azim=-135)
        # fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08, label="w")
        # plt.tight_layout()

        # ---- color by selected field ----
        C = np.abs(W)
        C_face = 0.25 * (C[:-1, :-1] + C[1:, :-1] + C[:-1, 1:] + C[1:, 1:])

        norm = mcolors.Normalize(vmin=float(C_face.min()), vmax=float(C_face.max()))
        cmap = cm.get_cmap("viridis")
        facecolors = cmap(norm(C_face))

        # created_fig = False
        # if ax is None:
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection="3d")
        # created_fig = True
        # else:
        #     fig = ax.figure

        ax.plot_surface(
            X, Y, Z,
            facecolors=facecolors,
            linewidth=0,
            antialiased=True,
            shade=False,
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("radial")
        # ax.set_title(f"color={'w'}")
        ax.view_init(elev=25, azim=-135)

        # if created_fig:
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.08, label=f"|w|")
        plt.tight_layout()
        plt.show()
        plt.show()

    # ------------------------
    # Multigrid hooks
    # ------------------------
    def prolongate(self, coarse_soln):
        return self.element.prolongate(coarse_soln, self.nxe // 2)
    
    def restrict_defect(self, fine_defect):
        return self.element.restrict_defect(fine_defect, self.nxe * 2)
    
