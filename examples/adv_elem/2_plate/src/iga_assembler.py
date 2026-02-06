import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
from _sparse_utils import build_csr_from_conn

class IGAPlateAssembler:
    # for IGA bases or elements

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

        assert self.element.ORDER == 2
        self.nnx = self.nxe + 2
        self.nnodes = self.nnx**2
        self.dof_per_node = self.element.dof_per_node
        self.N = self.dof_per_node * self.nnodes
        self.num_elements = self.nxe**2

        # TODO : fix based on IGABeamAssemblerV2 to get proper mesh conv rate
        self.dx_ctrl = self.length / (self.nnx - 1) # control point spacing
        self.h_span = self.length / self.nxe # knot span length (integration interval)
        self.dy_ctrl = self.width / (self.nnx - 1)
        # self.dx = self.length / (self.nnx - 1)
        # self.dy = self.width  / (self.nnx - 1)

        assert not clamped  # as you had
        self.clamped = clamped
        self.element.clamped = False

        self.bcs = []
        for inode in range(self.nnodes):
            ix = inode % self.nnx
            iy = inode // self.nnx
            if ix == 0 or iy == 0 or ix == self.nnx - 1 or iy == self.nnx - 1:
                self.bcs.append(inode)

        # ELEMENT CONN (9 nodes per elem)
        self.conn = []
        for ielem in range(self.num_elements):
            ixe = ielem % self.nxe
            iye = ielem // self.nxe
            loc = []
            for lnode in range(9):
                lx = lnode % 3
                ly = lnode // 3
                ix = ixe + lx
                iy = iye + ly
                loc.append(self.nnx * iy + ix)
            self.conn.append(np.array(loc, dtype=int))

        # DOF CONN (9*dpn dofs per elem)
        dpn = self.dof_per_node
        self.dof_conn = []
        for loc in self.conn:
            dofs = np.empty((9*dpn,), dtype=int)
            k = 0
            for n in loc:
                base = dpn*n
                for a in range(dpn):
                    dofs[k] = base + a
                    k += 1
            self.dof_conn.append(dofs)

        self.rowp, self.cols, self.nnzb = build_csr_from_conn(self.conn, self.nnodes)

    def _assemble_system(self, bcs:bool=True):
        dpn = self.dof_per_node
        self.data = np.zeros((self.nnzb, dpn, dpn), dtype=np.double)
        self.force = np.zeros(self.N)

        for ielem in range(self.num_elements):
            ixe = ielem % self.nxe
            iye = ielem // self.nxe
            loc_conn = self.conn[ielem]
            loc_dofs = self.dof_conn[ielem]

            # elem_xpts: [x,y,0] for 9 nodes -> length 27
            elem_xpts = np.zeros(27, dtype=np.double)
            for lnode, gnode in enumerate(loc_conn):
                ix = gnode % self.nnx
                iy = gnode // self.nnx
                elem_xpts[3*lnode + 0] = ix * self.dx_ctrl
                elem_xpts[3*lnode + 1] = iy * self.dy_ctrl
                elem_xpts[3*lnode + 2] = 0.0

            kelem = self.element.get_kelem(
                self.E, self.nu, self.thick, elem_xpts,
                left_bndry  = (ixe == 0),
                right_bndry = (ixe == self.nxe - 1),
                bot_bndry   = (iye == 0),
                top_bndry   = (iye == self.nxe - 1),
            )

            felem = self.element.get_felem(
                mag=self.load_fcn,
                elem_xpts=elem_xpts,
                left_bndry  = (ixe == 0),
                right_bndry = (ixe == self.nxe - 1),
                bot_bndry   = (iye == 0),
                top_bndry   = (iye == self.nxe - 1),
            )

            # LHS
            for lbr, br in enumerate(loc_conn):
                for colp in range(self.rowp[br], self.rowp[br+1]):
                    bc = self.cols[colp]
                    hit = np.where(loc_conn == bc)[0]
                    if hit.size == 0:
                        continue
                    lbc = int(hit[0])
                    self.data[colp,:,:] += kelem[dpn*lbr:dpn*(lbr+1), dpn*lbc:dpn*(lbc+1)]

            # RHS (DOF-level)
            np.add.at(self.force, loc_dofs, felem)

        # ---- BCs ----
        if bcs:
            if self.split_disp_bc:
                # expects dpn=3: [w_b, w_s1, w_s2]
                assert dpn == 3

                for node in self.bcs:
                    ix = node % self.nnx
                    iy = node // self.nnx
                    on_left  = (ix == 0)
                    on_right = (ix == self.nnx - 1)
                    on_bot   = (iy == 0)
                    on_top   = (iy == self.nnx - 1)

                    # bottom-left corner: pin all 3 (clean gauge fix)
                    if on_left and on_bot:
                        for colp in range(self.rowp[node], self.rowp[node+1]):
                            bc = self.cols[colp]
                            self.data[colp,:,:] = np.eye(3) if (bc == node) else 0.0
                        self.force[3*node:3*node+3] = 0.0
                        continue

                    # row 0: w_b + w_s1 + w_s2 = 0
                    for colp in range(self.rowp[node], self.rowp[node+1]):
                        bc = self.cols[colp]
                        self.data[colp,0,:] = 0.0
                        if bc == node:
                            self.data[colp,0,0] = 1.0
                            self.data[colp,0,1] = 1.0
                            self.data[colp,0,2] = 1.0
                    self.force[3*node + 0] = 0.0

                    # row 1: w_s1 = 0 on left
                    if on_left: # or on_top:
                        for colp in range(self.rowp[node], self.rowp[node+1]):
                            bc = self.cols[colp]
                            self.data[colp,1,:] = 0.0
                            if bc == node:
                                self.data[colp,1,1] = 1.0
                        self.force[3*node + 1] = 0.0

                    # row 2: w_s2 = 0 on bottom
                    if on_bot: # or on_right:
                        for colp in range(self.rowp[node], self.rowp[node+1]):
                            bc = self.cols[colp]
                            self.data[colp,2,:] = 0.0
                            if bc == node:
                                self.data[colp,2,2] = 1.0
                        self.force[3*node + 2] = 0.0

            else:
                # simple SS: w=0 on boundary (idof 0), leave rotations free
                # (if you later want clamped, change idofs below to range(dpn))
                for node in self.bcs:
                    idofs = [0]
                    for idof in idofs:
                        for colp in range(self.rowp[node], self.rowp[node+1]):
                            bc = self.cols[colp]
                            for jdof in range(dpn):
                                self.data[colp,idof,jdof] = 0.0
                            if bc == node:
                                self.data[colp,idof,idof] = 1.0
                        self.force[dpn*node + idof] = 0.0

        self.kmat = sp.bsr_matrix((self.data, self.cols, self.rowp), shape=(self.N, self.N))

        # kmat_dense = self.kmat.toarray()
        # plt.imshow(np.log(1 + np.abs(kmat_dense) + 1e-14))
        # plt.show()
        # print(f'{self.force=}')

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

    def plot_disp(self, combine_split=True):
        """3D surface plot of w(x,y) on the CONTROL grid (debug)."""
        if self.u is None:
            raise RuntimeError("Run direct_solve() first.")

        dpn = self.dof_per_node
        U = self.u.reshape((self.nnodes, dpn))

        # print(f"{U=}")

        if self.split_disp_bc and combine_split and dpn == 3:
            w = U[:, 0] + U[:, 1] + U[:, 2]
        else:
            w = U[:, 0]

        # IMPORTANT: inode = ix + nnx*iy  -> W[iy, ix]
        W = w.reshape((self.nnx, self.nnx))  # row=iy, col=ix

        wmin = float(W.min())
        wmax = float(W.max())
        # print(f"w range: [{wmin:.6e}, {wmax:.6e}], ptp={wmax-wmin:.6e}")

        # physical coordinates consistent with how you build elem_xpts
        x = np.arange(self.nnx) * self.dx_ctrl
        y = np.arange(self.nnx) * self.dy_ctrl
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
        return self.element.prolongate(coarse_soln)
    
    def restrict_defect(self, fine_defect):
        return self.element.restrict_defect(fine_defect)