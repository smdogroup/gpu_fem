import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# =============================================================================
# 2D Assembler: structured grid, block-CSR (3x3 blocks), deRham plate
# =============================================================================
class DeRhamIGAPlateAssembler:
    """
    Structured 2D Reissner–Mindlin plate discretization (de Rham compatible):
      unknown ordering (global):
        u = [ w(0..nw-1), thx(0..ntx-1), thy(0..nty-1) ]
    """

    def __init__(
        self,
        ELEMENT,
        nxe: int,
        E=70e9,
        nu=0.3,
        thick=1e-2,
        length:float=1.0,
        width:float=1.0,
        load_fcn=lambda x, y: 1.0,
        clamped: bool = False,
        split_disp_bc:bool=False,
        bdf_file:str=""
    ):
        self.element = ELEMENT(clamped=clamped) if callable(ELEMENT) else ELEMENT
        self.nxe = int(nxe)
        # self.nye = int(nye)
        self.nye = int(nxe)
        self.E = float(E)
        self.nu = float(nu)
        self.thick = float(thick)
        self.Lx = float(length)
        self.Ly = float(width)
        self.load_fcn = load_fcn
        self.clamped = bool(clamped)

        self.element.clamped = clamped

        # DOF per node
        self.dof_per_node = 1

        # grid sizes
        self.nxw, self.nyw = self.nxe + 2, self.nye + 2
        self.nxtx, self.nytx = self.nxe + 1, self.nye + 2
        self.nxty, self.nyty = self.nxe + 2, self.nye + 1

        self.nw = self.nxw * self.nyw
        self.ntx = self.nxtx * self.nytx
        self.nty = self.nxty * self.nyty
        self.N = self.nw + self.ntx + self.nty

        self.dx = self.Lx / self.nxe
        self.dy = self.Ly / self.nye

        self.kmat = None
        self.force = np.zeros(self.N)
        self.u = None

        # build BC list (strongly enforced u[dof]=0)
        self.bcs = self._build_bcs()

        # connectivity (structured)
        self.conn_w, self.conn_tx, self.conn_ty = self._build_connectivity()

        # build block-CSR patterns (safe stencils)
        self._build_block_patterns()
        self._alloc_blocks()

    # -----------------------------
    # Connectivity
    # -----------------------------
    def _node(self, i, j, nx):
        return i + nx * j

    def _build_connectivity(self):
        conn_w = []
        conn_tx = []
        conn_ty = []

        # element (ex,ey) corresponds to the "lower-left" knot span in w grid at (ex,ey)
        for ey in range(self.nye):
            for ex in range(self.nxe):
                # w: 3x3 starting at (ex,ey) in (nxw,nyw)
                wloc = []
                for ly in range(3):
                    for lx in range(3):
                        wloc.append(self._node(ex + lx, ey + ly, self.nxw))
                conn_w.append(wloc)

                # thx: 2x3 starting at (ex,ey) in (nxtx,nytx)
                txloc = []
                for ly in range(3):
                    for lx in range(2):
                        txloc.append(self._node(ex + lx, ey + ly, self.nxtx))
                conn_tx.append(txloc)

                # thy: 3x2 starting at (ex,ey) in (nxty,nyty)
                tyloc = []
                for ly in range(2):
                    for lx in range(3):
                        tyloc.append(self._node(ex + lx, ey + ly, self.nxty))
                conn_ty.append(tyloc)

        return conn_w, conn_tx, conn_ty

    def _build_bcs(self):
        bcs = []

        def on_bndry(i, j, nx, ny):
            return (i == 0) or (i == nx-1) or (j == 0) or (j == ny-1)

        # w boundary
        for j in range(self.nyw):
            for i in range(self.nxw):
                if on_bndry(i, j, self.nxw, self.nyw):
                    bcs.append(self._node(i, j, self.nxw))

        # theta boundary if clamped
        if self.clamped:
            off_tx = self.nw
            off_ty = self.nw + self.ntx

            for j in range(self.nytx):
                for i in range(self.nxtx):
                    if on_bndry(i, j, self.nxtx, self.nytx):
                        bcs.append(off_tx + self._node(i, j, self.nxtx))

            for j in range(self.nyty):
                for i in range(self.nxty):
                    if on_bndry(i, j, self.nxty, self.nyty):
                        bcs.append(off_ty + self._node(i, j, self.nxty))

        # unique / sorted (helps reproducibility)
        bcs = sorted(set(bcs))
        return bcs

    # -----------------------------
    # CSR pattern helpers
    # -----------------------------
    @staticmethod
    def _row_find_col(col_ind: np.ndarray, start: int, end: int, target_col: int) -> int:
        for p in range(start, end):
            if col_ind[p] == target_col:
                return p
        return -1

    def _add_to_block(self, A: sp.csr_matrix, rowp, cols, i: int, j: int, val: float):
        start, end = rowp[i], rowp[i + 1]
        p = self._row_find_col(cols, start, end, j)
        if p < 0:
            raise RuntimeError(f"CSR pattern missing entry ({i},{j}). Increase stencil.")
        A.data[p] += val

    def _build_pattern_from_stencil(self, nrow: int, row_grid, col_grid, stencil_ix, stencil_iy):
        """
        row_grid: (nx_r, ny_r)
        col_grid: (nx_c, ny_c)
        stencil_ix: (imin, imax) inclusive offsets in i
        stencil_iy: (jmin, jmax) inclusive offsets in j
        """
        nx_r, ny_r = row_grid
        nx_c, ny_c = col_grid

        rowp = [0]
        cols = []
        nnz = 0

        for jr in range(ny_r):
            for ir in range(nx_r):
                # row id
                # gather candidate cols
                cand = []
                for dj in range(stencil_iy[0], stencil_iy[1] + 1):
                    jc = jr + dj
                    if jc < 0 or jc >= ny_c:
                        continue
                    for di in range(stencil_ix[0], stencil_ix[1] + 1):
                        ic = ir + di
                        if ic < 0 or ic >= nx_c:
                            continue
                        cand.append(ic + nx_c * jc)
                cand = sorted(set(cand))
                cols.extend(cand)
                nnz += len(cand)
                rowp.append(nnz)

        return rowp, np.array(cols, dtype=np.int32), nnz

    def _build_block_patterns(self):
        # Safe stencils (a bit generous, prevents pattern misses)
        # ww: (5x5)
        self.ww_rowp, self.ww_cols, self.ww_nnz = self._build_pattern_from_stencil(
            self.nw, (self.nxw, self.nyw), (self.nxw, self.nyw), (-2, 2), (-2, 2)
        )
        # w-tx: (4x5)
        self.wtx_rowp, self.wtx_cols, self.wtx_nnz = self._build_pattern_from_stencil(
            self.nw, (self.nxw, self.nyw), (self.nxtx, self.nytx), (-2, 1), (-2, 2)
        )
        # w-ty: (5x4)
        self.wty_rowp, self.wty_cols, self.wty_nnz = self._build_pattern_from_stencil(
            self.nw, (self.nxw, self.nyw), (self.nxty, self.nyty), (-2, 2), (-2, 1)
        )

        # tx-w: (4x5) “transpose-ish”
        self.txw_rowp, self.txw_cols, self.txw_nnz = self._build_pattern_from_stencil(
            self.ntx, (self.nxtx, self.nytx), (self.nxw, self.nyw), (-1, 2), (-2, 2)
        )
        # tx-tx: (3x5)
        self.txtx_rowp, self.txtx_cols, self.txtx_nnz = self._build_pattern_from_stencil(
            self.ntx, (self.nxtx, self.nytx), (self.nxtx, self.nytx), (-1, 1), (-2, 2)
        )
        # tx-ty: (4x4)
        self.txty_rowp, self.txty_cols, self.txty_nnz = self._build_pattern_from_stencil(
            self.ntx, (self.nxtx, self.nytx), (self.nxty, self.nyty), (-1, 2), (-2, 1)
        )

        # ty-w: (5x4)
        self.tyw_rowp, self.tyw_cols, self.tyw_nnz = self._build_pattern_from_stencil(
            self.nty, (self.nxty, self.nyty), (self.nxw, self.nyw), (-2, 2), (-1, 2)
        )
        # ty-tx: (4x4)
        self.tytx_rowp, self.tytx_cols, self.tytx_nnz = self._build_pattern_from_stencil(
            self.nty, (self.nxty, self.nyty), (self.nxtx, self.nytx), (-2, 1), (-1, 2)
        )
        # ty-ty: (5x3)
        self.tyty_rowp, self.tyty_cols, self.tyty_nnz = self._build_pattern_from_stencil(
            self.nty, (self.nxty, self.nyty), (self.nxty, self.nyty), (-2, 2), (-1, 1)
        )

    def _alloc_blocks(self):
        # allocate CSR blocks
        self.k_ww   = sp.csr_matrix((np.zeros(self.ww_nnz),   self.ww_cols,   self.ww_rowp),   shape=(self.nw,  self.nw))
        self.k_wtx  = sp.csr_matrix((np.zeros(self.wtx_nnz),  self.wtx_cols,  self.wtx_rowp),  shape=(self.nw,  self.ntx))
        self.k_wty  = sp.csr_matrix((np.zeros(self.wty_nnz),  self.wty_cols,  self.wty_rowp),  shape=(self.nw,  self.nty))

        self.k_txw  = sp.csr_matrix((np.zeros(self.txw_nnz),  self.txw_cols,  self.txw_rowp),  shape=(self.ntx, self.nw))
        self.k_txtx = sp.csr_matrix((np.zeros(self.txtx_nnz), self.txtx_cols, self.txtx_rowp), shape=(self.ntx, self.ntx))
        self.k_txty = sp.csr_matrix((np.zeros(self.txty_nnz), self.txty_cols, self.txty_rowp), shape=(self.ntx, self.nty))

        self.k_tyw  = sp.csr_matrix((np.zeros(self.tyw_nnz),  self.tyw_cols,  self.tyw_rowp),  shape=(self.nty, self.nw))
        self.k_tytx = sp.csr_matrix((np.zeros(self.tytx_nnz), self.tytx_cols, self.tytx_rowp), shape=(self.nty, self.ntx))
        self.k_tyty = sp.csr_matrix((np.zeros(self.tyty_nnz), self.tyty_cols, self.tyty_rowp), shape=(self.nty, self.nty))

        # global bmat view
        self.kmat = sp.bmat(
            [
                [self.k_ww,   self.k_wtx,  self.k_wty],
                [self.k_txw,  self.k_txtx, self.k_txty],
                [self.k_tyw,  self.k_tytx, self.k_tyty],
            ],
            format="csr",
        )

        self.fw  = np.zeros(self.nw)
        self.ftx = np.zeros(self.ntx)
        self.fty = np.zeros(self.nty)

    # -----------------------------
    # Assembly
    # -----------------------------
    def _assemble_system(self):
        # zero stiffness
        for A in [self.k_ww, self.k_wtx, self.k_wty,
                  self.k_txw, self.k_txtx, self.k_txty,
                  self.k_tyw, self.k_tytx, self.k_tyty]:
            A.data[:] = 0.0

        # zero rhs
        self.fw[:] = 0.0
        self.ftx[:] = 0.0
        self.fty[:] = 0.0

        # loop elements
        elem_id = 0
        for ey in range(self.nye):
            for ex in range(self.nxe):
                w_dofs  = self.conn_w[elem_id]   # 9
                tx_dofs = self.conn_tx[elem_id]  # 6
                ty_dofs = self.conn_ty[elem_id]  # 6

                left_b  = (ex == 0)
                right_b = (ex == self.nxe - 1)
                bot_b   = (ey == 0)
                top_b   = (ey == self.nye - 1)

                (Kww, Kwtx, Kwty,
                 Ktxw, Ktxtx, Ktxty,
                 Ktyw, Ktytx, Ktyty) = self.element.get_kelem(
                    E=self.E, nu=self.nu, thick=self.thick,
                    dx=self.dx, dy=self.dy,
                    left_bndry=left_b, right_bndry=right_b,
                    bot_bndry=bot_b, top_bndry=top_b
                 )

                x0 = ex * self.dx
                y0 = ey * self.dy
                fw, ftx, fty = self.element.get_felem(
                    self.load_fcn, x0, y0, self.dx, self.dy,
                    left_b, right_b, bot_b, top_b
                )

                # scatter-add
                for a, I in enumerate(w_dofs):
                    self.fw[I] += fw[a]
                    for b, J in enumerate(w_dofs):
                        self._add_to_block(self.k_ww, self.ww_rowp, self.k_ww.indices, I, J, Kww[a, b])
                    for b, J in enumerate(tx_dofs):
                        self._add_to_block(self.k_wtx, self.wtx_rowp, self.k_wtx.indices, I, J, Kwtx[a, b])
                    for b, J in enumerate(ty_dofs):
                        self._add_to_block(self.k_wty, self.wty_rowp, self.k_wty.indices, I, J, Kwty[a, b])

                for a, I in enumerate(tx_dofs):
                    self.ftx[I] += ftx[a]
                    for b, J in enumerate(w_dofs):
                        self._add_to_block(self.k_txw, self.txw_rowp, self.k_txw.indices, I, J, Ktxw[a, b])
                    for b, J in enumerate(tx_dofs):
                        self._add_to_block(self.k_txtx, self.txtx_rowp, self.k_txtx.indices, I, J, Ktxtx[a, b])
                    for b, J in enumerate(ty_dofs):
                        self._add_to_block(self.k_txty, self.txty_rowp, self.k_txty.indices, I, J, Ktxty[a, b])

                for a, I in enumerate(ty_dofs):
                    self.fty[I] += fty[a]
                    for b, J in enumerate(w_dofs):
                        self._add_to_block(self.k_tyw, self.tyw_rowp, self.k_tyw.indices, I, J, Ktyw[a, b])
                    for b, J in enumerate(tx_dofs):
                        self._add_to_block(self.k_tytx, self.tytx_rowp, self.k_tytx.indices, I, J, Ktytx[a, b])
                    for b, J in enumerate(ty_dofs):
                        self._add_to_block(self.k_tyty, self.tyty_rowp, self.k_tyty.indices, I, J, Ktyty[a, b])

                elem_id += 1

        # rebuild global view
        self.kmat = sp.bmat(
            [
                [self.k_ww,   self.k_wtx,  self.k_wty],
                [self.k_txw,  self.k_txtx, self.k_txty],
                [self.k_tyw,  self.k_tytx, self.k_tyty],
            ],
            format="csr",
        )

        self.force[:self.nw] = self.fw
        self.force[self.nw:self.nw+self.ntx] = self.ftx
        self.force[self.nw+self.ntx:] = self.fty

        self._apply_bcs()

    def _apply_bcs(self):
        if self.kmat is None:
            raise RuntimeError("Assemble first.")

        K = self.kmat.tolil()
        f = self.force.copy()

        for dof in self.bcs:
            K.rows[dof] = [dof]
            K.data[dof] = [1.0]
            f[dof] = 0.0

        # symmetric column zeroing
        for dof in self.bcs:
            K[:, dof] = 0.0
            K[dof, dof] = 1.0

        self.kmat = K.tocsr()
        self.force = f

    # -----------------------------
    # Solve + MG hooks
    # -----------------------------
    def direct_solve(self):
        self._assemble_system()
        self.u = sp.linalg.spsolve(self.kmat, self.force)
        return self.u

    def prolongate(self, coarse_soln: np.ndarray):
        nxe_c = self.nxe // 2
        nye_c = self.nye // 2
        return self.element.prolongate(coarse_soln, nxe_c, nye_c)

    def restrict_defect(self, fine_defect: np.ndarray):
        # called on coarse grid object; pass its (nxe,nye) as coarse sizes
        return self.element.restrict_defect(fine_defect, self.nxe, self.nye)

    def plot_disp(self, combine_split=True):
        """3D surface plot of w(x,y) on the CONTROL grid (debug)."""
        if self.u is None:
            raise RuntimeError("Run direct_solve() first.")

        w = self.u[:self.nw]

        # IMPORTANT: inode = ix + nnx*iy  -> W[iy, ix]
        W = w.reshape((self.nxw, self.nxw))  # row=iy, col=ix

        wmin = float(W.min())
        wmax = float(W.max())
        # print(f"w range: [{wmin:.6e}, {wmax:.6e}], ptp={wmax-wmin:.6e}")

        # physical coordinates consistent with how you build elem_xpts
        x = np.arange(self.nxw) * self.dx
        y = np.arange(self.nxw) * self.dy
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