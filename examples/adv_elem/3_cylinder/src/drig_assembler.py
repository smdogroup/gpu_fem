import numpy as np
import scipy.sparse as sp

# =============================================================================
# 2D Assembler: structured grid, block-CSR, mixed DRIG cylinder (open hoop)
# Updated spaces:
#   w   ~ (p2,p2)  "2x2 IGA"
#   thx ~ (p1,p2)  "1x2 IGA"
#   thy ~ (p2,p1)  "2x1 IGA"
#   u   ~ (p3,p2)  "3x2 IGA"
#   v   ~ (p2,p3)  "2x3 IGA"
# Global unknown ordering:
#   U = [ w, u, v, thx, thy ]
# =============================================================================
class DeRhamIGACylinderAssembler:
    """
    Structured 2D IGA cylinder discretization on (x,s) for the mixed element:

    Global unknown ordering:
      U = [ w, u, v, thx, thy ]

    Field spaces:
      w   : (p2,p2) -> (nxe+2) x (nye+2)
      u   : (p3,p2) -> (nxe+3) x (nye+2)
      v   : (p2,p3) -> (nxe+2) x (nye+3)
      thx : (p1,p2) -> (nxe+1) x (nye+2)
      thy : (p2,p1) -> (nxe+2) x (nye+1)

    Element interface expected:
      element.get_kelem(E, nu, thick, dx, dy, ixe, nxe, iye, nye) -> 25 blocks (5x5)
      element.get_felem(load_fcn, x0, y0, dx, dy, ixe, nxe, iye, nye) -> (fw, fu, fv, fthx, fthy)

    BCs (strong):
      - clamped=False:
          w = 0 on all 4 sides  (w grid)
          u = 0 on x=0 edge     (u grid)
          v = 0 on y=0 edge     (v grid)
          (thx, thy free)
      - clamped=True:
          w,u,v,thx,thy = 0 on all 4 sides (each on its own grid)
    """

    def __init__(
        self,
        ELEMENT,
        nxe: int,
        nye: int = None,
        E=70e9,
        nu=0.3,
        thick=1e-2,
        length: float = 1.0,
        radius: float = 1.0,
        hoop_length: float = np.pi,
        load_fcn=lambda x, s: 0.0,
        clamped: bool = False,
        geometry="" # unused temp arg for std assembler
    ):
        self.element = ELEMENT if not callable(ELEMENT) else ELEMENT(r=radius, clamped=clamped)

        self.nxe = int(nxe)
        self.nye = int(nye if nye is not None else nxe)

        self.E = float(E)
        self.nu_poiss = float(nu)
        self.thick = float(thick)
        self.Lx = float(length)
        self.radius = float(radius)
        self.Ly = float(hoop_length)
        self.clamped = bool(clamped)

        # change load function from standard x,y,z to x,s coordinates
        # def xs_load_fcn(x,s):
        #     y = -radius * np.cos(s / radius)
        #     z = radius * np.sin(s / radius)
        #     return load_fcn(x,y,z)
        # self.load_fcn = xs_load_fcn
        self.load_fcn = load_fcn

        # keep element flag in sync
        if hasattr(self.element, "clamped"):
            self.element.clamped = self.clamped

        self.dof_per_node = 5

        # -----------------------------
        # Grid sizes (match spaces)
        # -----------------------------
        self.nx_w,   self.ny_w   = self.nxe + 2, self.nye + 2  # w   (p2,p2)
        self.nx_u,   self.ny_u   = self.nxe + 3, self.nye + 2  # u   (p3,p2)
        self.nx_v,   self.ny_v   = self.nxe + 2, self.nye + 3  # v   (p2,p3)
        self.nx_thx, self.ny_thx = self.nxe + 1, self.nye + 2  # thx (p1,p2)
        self.nx_thy, self.ny_thy = self.nxe + 2, self.nye + 1  # thy (p2,p1)

        self.nw   = self.nx_w   * self.ny_w
        self.nu_d = self.nx_u   * self.ny_u
        self.nv   = self.nx_v   * self.ny_v
        self.nthx = self.nx_thx * self.ny_thx
        self.nthy = self.nx_thy * self.ny_thy

        self.N = self.nw + self.nu_d + self.nv + self.nthx + self.nthy

        # offsets
        self.off_w   = 0
        self.off_u   = self.off_w   + self.nw
        self.off_v   = self.off_u   + self.nu_d
        self.off_thx = self.off_v   + self.nv
        self.off_thy = self.off_thx + self.nthx

        # element sizes
        self.dx = self.Lx / self.nxe
        self.dy = self.Ly / self.nye

        # system containers
        self.kmat = None
        self.force = np.zeros(self.N)
        self.u = None

        # strong BC dof list
        self.bcs = self._build_bcs()

        # connectivity
        (self.conn_w, self.conn_u,
         self.conn_v, self.conn_thx, self.conn_thy) = self._build_connectivity()

        # block CSR patterns + allocations
        self._build_block_patterns()
        self._alloc_blocks()

    # -------------------------------------------------------------------------
    # Index helper
    # -------------------------------------------------------------------------
    @staticmethod
    def _node(i: int, j: int, nx: int) -> int:
        return i + nx * j

    # -------------------------------------------------------------------------
    # Connectivity (open hoop, no wrap)
    # -------------------------------------------------------------------------
    def _build_connectivity(self):
        conn_w, conn_u, conn_v, conn_thx, conn_thy = ([] for _ in range(5))

        elem_id = 0
        for ey in range(self.nye):
            for ex in range(self.nxe):

                # w: (p2,p2) => 3x3 on (nx_w, ny_w)
                wloc = []
                for ly in range(3):
                    jy = ey + ly
                    for lx in range(3):
                        ix = ex + lx
                        wloc.append(self._node(ix, jy, self.nx_w))
                conn_w.append(wloc)

                # u: (p3,p2) => 4x3 on (nx_u, ny_u)
                uloc = []
                for ly in range(3):
                    jy = ey + ly
                    for lx in range(4):
                        ix = ex + lx
                        uloc.append(self._node(ix, jy, self.nx_u))
                conn_u.append(uloc)

                # v: (p2,p3) => 3x4 on (nx_v, ny_v)
                vloc = []
                for ly in range(4):
                    jy = ey + ly
                    for lx in range(3):
                        ix = ex + lx
                        vloc.append(self._node(ix, jy, self.nx_v))
                conn_v.append(vloc)

                # thx: (p1,p2) => 2x3 on (nx_thx, ny_thx)
                txloc = []
                for ly in range(3):
                    jy = ey + ly
                    for lx in range(2):
                        ix = ex + lx
                        txloc.append(self._node(ix, jy, self.nx_thx))
                conn_thx.append(txloc)

                # thy: (p2,p1) => 3x2 on (nx_thy, ny_thy)
                tyloc = []
                for ly in range(2):
                    jy = ey + ly
                    for lx in range(3):
                        ix = ex + lx
                        tyloc.append(self._node(ix, jy, self.nx_thy))
                conn_thy.append(tyloc)

                elem_id += 1

        return conn_w, conn_u, conn_v, conn_thx, conn_thy

    # -------------------------------------------------------------------------
    # Boundary conditions (strong, via row/col elimination)
    # -------------------------------------------------------------------------
    def _build_bcs(self):
        bcs = []

        def on_bndry(i, j, nx, ny):
            return (i == 0) or (i == nx - 1) or (j == 0) or (j == ny - 1)

        # w always on all sides
        for j in range(self.ny_w):
            for i in range(self.nx_w):
                if on_bndry(i, j, self.nx_w, self.ny_w):
                    bcs.append(self.off_w + self._node(i, j, self.nx_w))

        if self.clamped:
            # u all sides
            for j in range(self.ny_u):
                for i in range(self.nx_u):
                    if on_bndry(i, j, self.nx_u, self.ny_u):
                        bcs.append(self.off_u + self._node(i, j, self.nx_u))

            # v all sides
            for j in range(self.ny_v):
                for i in range(self.nx_v):
                    if on_bndry(i, j, self.nx_v, self.ny_v):
                        bcs.append(self.off_v + self._node(i, j, self.nx_v))

            # thx all sides
            for j in range(self.ny_thx):
                for i in range(self.nx_thx):
                    if on_bndry(i, j, self.nx_thx, self.ny_thx):
                        bcs.append(self.off_thx + self._node(i, j, self.nx_thx))

            # thy all sides
            for j in range(self.ny_thy):
                for i in range(self.nx_thy):
                    if on_bndry(i, j, self.nx_thy, self.ny_thy):
                        bcs.append(self.off_thy + self._node(i, j, self.nx_thy))
        else:
            # u: x=0 edge (on u grid)
            for j in range(self.ny_u):
                bcs.append(self.off_u + self._node(0, j, self.nx_u))

            # v: y=0 edge (on v grid)
            for i in range(self.nx_v):
                bcs.append(self.off_v + self._node(i, 0, self.nx_v))

        return sorted(set(bcs))

    # -------------------------------------------------------------------------
    # CSR pattern helpers
    # -------------------------------------------------------------------------
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

    def _build_pattern_from_stencil(
        self,
        nrow: int,
        row_grid, col_grid,
        stencil_ix, stencil_iy,
        periodic_y: bool = False,
    ):
        nx_r, ny_r = row_grid
        nx_c, ny_c = col_grid

        rowp = [0]
        cols = []
        nnz = 0

        for jr in range(ny_r):
            for ir in range(nx_r):
                cand = []
                for dj in range(stencil_iy[0], stencil_iy[1] + 1):
                    jc = jr + dj
                    if periodic_y:
                        jc = jc % ny_c
                    else:
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

    # -------------------------------------------------------------------------
    # Block patterns (open hoop -> periodic_y=False)
    # -------------------------------------------------------------------------
    def _build_block_patterns(self):
        per = False  # open hoop patch

        # Grids
        gw  = (self.nx_w,  self.ny_w)
        gu  = (self.nx_u,  self.ny_u)
        gv  = (self.nx_v,  self.ny_v)
        gtx = (self.nx_thx, self.ny_thx)
        gty = (self.nx_thy, self.ny_thy)

        # Support radii (B-spline p: support over p+1 elems => +/-p in index-ish)
        # Use conservative stencils:
        #   p3 -> +/-3, p2 -> +/-2, p1 -> +/-1

        # --- w rows (w grid is p2,p2)
        self.ww_rowp,   self.ww_cols,   self.ww_nnz   = self._build_pattern_from_stencil(self.nw,  gw,  gw,  (-2, 2), (-2, 2), per)
        self.wu_rowp,   self.wu_cols,   self.wu_nnz   = self._build_pattern_from_stencil(self.nw,  gw,  gu,  (-2, 3), (-2, 2), per)  # u has p3 in x
        self.wv_rowp,   self.wv_cols,   self.wv_nnz   = self._build_pattern_from_stencil(self.nw,  gw,  gv,  (-2, 2), (-2, 3), per)  # v has p3 in y
        self.wthx_rowp, self.wthx_cols, self.wthx_nnz = self._build_pattern_from_stencil(self.nw,  gw,  gtx, (-2, 1), (-2, 2), per)  # thx p1 in x
        self.wthy_rowp, self.wthy_cols, self.wthy_nnz = self._build_pattern_from_stencil(self.nw,  gw,  gty, (-2, 2), (-2, 1), per)  # thy p1 in y

        # --- u rows (u grid is p3,p2)
        self.uw_rowp,   self.uw_cols,   self.uw_nnz   = self._build_pattern_from_stencil(self.nu_d, gu, gw,  (-3, 2), (-2, 2), per)
        self.uu_rowp,   self.uu_cols,   self.uu_nnz   = self._build_pattern_from_stencil(self.nu_d, gu, gu,  (-3, 3), (-2, 2), per)
        self.uv_rowp,   self.uv_cols,   self.uv_nnz   = self._build_pattern_from_stencil(self.nu_d, gu, gv,  (-3, 2), (-2, 3), per)
        self.uthx_rowp, self.uthx_cols, self.uthx_nnz = self._build_pattern_from_stencil(self.nu_d, gu, gtx, (-3, 1), (-2, 2), per)
        self.uthy_rowp, self.uthy_cols, self.uthy_nnz = self._build_pattern_from_stencil(self.nu_d, gu, gty, (-3, 2), (-2, 1), per)

        # --- v rows (v grid is p2,p3)
        self.vw_rowp,   self.vw_cols,   self.vw_nnz   = self._build_pattern_from_stencil(self.nv, gv, gw,  (-2, 2), (-3, 2), per)
        self.vu_rowp,   self.vu_cols,   self.vu_nnz   = self._build_pattern_from_stencil(self.nv, gv, gu,  (-2, 3), (-3, 2), per)
        self.vv_rowp,   self.vv_cols,   self.vv_nnz   = self._build_pattern_from_stencil(self.nv, gv, gv,  (-2, 2), (-3, 3), per)
        self.vthx_rowp, self.vthx_cols, self.vthx_nnz = self._build_pattern_from_stencil(self.nv, gv, gtx, (-2, 1), (-3, 2), per)
        self.vthy_rowp, self.vthy_cols, self.vthy_nnz = self._build_pattern_from_stencil(self.nv, gv, gty, (-2, 2), (-3, 1), per)

        # --- thx rows (thx grid is p1,p2)
        self.thxw_rowp,   self.thxw_cols,   self.thxw_nnz   = self._build_pattern_from_stencil(self.nthx, gtx, gw,  (-1, 2), (-2, 2), per)
        self.thxu_rowp,   self.thxu_cols,   self.thxu_nnz   = self._build_pattern_from_stencil(self.nthx, gtx, gu,  (-1, 3), (-2, 2), per)
        self.thxv_rowp,   self.thxv_cols,   self.thxv_nnz   = self._build_pattern_from_stencil(self.nthx, gtx, gv,  (-1, 2), (-2, 3), per)
        self.thxthx_rowp, self.thxthx_cols, self.thxthx_nnz = self._build_pattern_from_stencil(self.nthx, gtx, gtx, (-1, 1), (-2, 2), per)
        self.thxthy_rowp, self.thxthy_cols, self.thxthy_nnz = self._build_pattern_from_stencil(self.nthx, gtx, gty, (-1, 2), (-2, 1), per)

        # --- thy rows (thy grid is p2,p1)
        self.thyw_rowp,   self.thyw_cols,   self.thyw_nnz   = self._build_pattern_from_stencil(self.nthy, gty, gw,  (-2, 2), (-1, 2), per)
        self.thyu_rowp,   self.thyu_cols,   self.thyu_nnz   = self._build_pattern_from_stencil(self.nthy, gty, gu,  (-2, 3), (-1, 2), per)
        self.thyv_rowp,   self.thyv_cols,   self.thyv_nnz   = self._build_pattern_from_stencil(self.nthy, gty, gv,  (-2, 2), (-1, 3), per)
        self.thythx_rowp, self.thythx_cols, self.thythx_nnz = self._build_pattern_from_stencil(self.nthy, gty, gtx, (-2, 1), (-1, 2), per)
        self.thythy_rowp, self.thythy_cols, self.thythy_nnz = self._build_pattern_from_stencil(self.nthy, gty, gty, (-2, 2), (-1, 1), per)

    # -------------------------------------------------------------------------
    # Allocate blocks + global bmat view
    # -------------------------------------------------------------------------
    def _alloc_blocks(self):
        z = np.zeros

        # w-row
        self.k_ww   = sp.csr_matrix((z(self.ww_nnz),   self.ww_cols,   self.ww_rowp),   shape=(self.nw,  self.nw))
        self.k_wu   = sp.csr_matrix((z(self.wu_nnz),   self.wu_cols,   self.wu_rowp),   shape=(self.nw,  self.nu_d))
        self.k_wv   = sp.csr_matrix((z(self.wv_nnz),   self.wv_cols,   self.wv_rowp),   shape=(self.nw,  self.nv))
        self.k_wthx = sp.csr_matrix((z(self.wthx_nnz), self.wthx_cols, self.wthx_rowp), shape=(self.nw,  self.nthx))
        self.k_wthy = sp.csr_matrix((z(self.wthy_nnz), self.wthy_cols, self.wthy_rowp), shape=(self.nw,  self.nthy))

        # u-row
        self.k_uw   = sp.csr_matrix((z(self.uw_nnz),   self.uw_cols,   self.uw_rowp),   shape=(self.nu_d, self.nw))
        self.k_uu   = sp.csr_matrix((z(self.uu_nnz),   self.uu_cols,   self.uu_rowp),   shape=(self.nu_d, self.nu_d))
        self.k_uv   = sp.csr_matrix((z(self.uv_nnz),   self.uv_cols,   self.uv_rowp),   shape=(self.nu_d, self.nv))
        self.k_uthx = sp.csr_matrix((z(self.uthx_nnz), self.uthx_cols, self.uthx_rowp), shape=(self.nu_d, self.nthx))
        self.k_uthy = sp.csr_matrix((z(self.uthy_nnz), self.uthy_cols, self.uthy_rowp), shape=(self.nu_d, self.nthy))

        # v-row
        self.k_vw   = sp.csr_matrix((z(self.vw_nnz),   self.vw_cols,   self.vw_rowp),   shape=(self.nv, self.nw))
        self.k_vu   = sp.csr_matrix((z(self.vu_nnz),   self.vu_cols,   self.vu_rowp),   shape=(self.nv, self.nu_d))
        self.k_vv   = sp.csr_matrix((z(self.vv_nnz),   self.vv_cols,   self.vv_rowp),   shape=(self.nv, self.nv))
        self.k_vthx = sp.csr_matrix((z(self.vthx_nnz), self.vthx_cols, self.vthx_rowp), shape=(self.nv, self.nthx))
        self.k_vthy = sp.csr_matrix((z(self.vthy_nnz), self.vthy_cols, self.vthy_rowp), shape=(self.nv, self.nthy))

        # thx-row
        self.k_thxw   = sp.csr_matrix((z(self.thxw_nnz),   self.thxw_cols,   self.thxw_rowp),   shape=(self.nthx, self.nw))
        self.k_thxu   = sp.csr_matrix((z(self.thxu_nnz),   self.thxu_cols,   self.thxu_rowp),   shape=(self.nthx, self.nu_d))
        self.k_thxv   = sp.csr_matrix((z(self.thxv_nnz),   self.thxv_cols,   self.thxv_rowp),   shape=(self.nthx, self.nv))
        self.k_thxthx = sp.csr_matrix((z(self.thxthx_nnz), self.thxthx_cols, self.thxthx_rowp), shape=(self.nthx, self.nthx))
        self.k_thxthy = sp.csr_matrix((z(self.thxthy_nnz), self.thxthy_cols, self.thxthy_rowp), shape=(self.nthx, self.nthy))

        # thy-row
        self.k_thyw   = sp.csr_matrix((z(self.thyw_nnz),   self.thyw_cols,   self.thyw_rowp),   shape=(self.nthy, self.nw))
        self.k_thyu   = sp.csr_matrix((z(self.thyu_nnz),   self.thyu_cols,   self.thyu_rowp),   shape=(self.nthy, self.nu_d))
        self.k_thyv   = sp.csr_matrix((z(self.thyv_nnz),   self.thyv_cols,   self.thyv_rowp),   shape=(self.nthy, self.nv))
        self.k_thythx = sp.csr_matrix((z(self.thythx_nnz), self.thythx_cols, self.thythx_rowp), shape=(self.nthy, self.nthx))
        self.k_thythy = sp.csr_matrix((z(self.thythy_nnz), self.thythy_cols, self.thythy_rowp), shape=(self.nthy, self.nthy))

        # RHS blocks
        self.fw   = np.zeros(self.nw)
        self.fu   = np.zeros(self.nu_d)
        self.fv   = np.zeros(self.nv)
        self.fthx = np.zeros(self.nthx)
        self.fthy = np.zeros(self.nthy)

        self._rebuild_global_bmat()

    def _rebuild_global_bmat(self):
        self.kmat = sp.bmat(
            [
                [self.k_ww,    self.k_wu,   self.k_wv,   self.k_wthx,   self.k_wthy],
                [self.k_uw,    self.k_uu,   self.k_uv,   self.k_uthx,   self.k_uthy],
                [self.k_vw,    self.k_vu,   self.k_vv,   self.k_vthx,   self.k_vthy],
                [self.k_thxw,  self.k_thxu, self.k_thxv, self.k_thxthx, self.k_thxthy],
                [self.k_thyw,  self.k_thyu, self.k_thyv, self.k_thythx, self.k_thythy],
            ],
            format="csr",
        )

    # -------------------------------------------------------------------------
    # Assembly
    # -------------------------------------------------------------------------
    def _zero_system(self):
        for A in [
            self.k_ww, self.k_wu, self.k_wv, self.k_wthx, self.k_wthy,
            self.k_uw, self.k_uu, self.k_uv, self.k_uthx, self.k_uthy,
            self.k_vw, self.k_vu, self.k_vv, self.k_vthx, self.k_vthy,
            self.k_thxw, self.k_thxu, self.k_thxv, self.k_thxthx, self.k_thxthy,
            self.k_thyw, self.k_thyu, self.k_thyv, self.k_thythx, self.k_thythy,
        ]:
            A.data[:] = 0.0

        self.fw[:] = 0.0
        self.fu[:] = 0.0
        self.fv[:] = 0.0
        self.fthx[:] = 0.0
        self.fthy[:] = 0.0

    def _scatter_block(self, A, rowp, cols, row_dofs, col_dofs, Kloc):
        for a, I in enumerate(row_dofs):
            for b, J in enumerate(col_dofs):
                self._add_to_block(A, rowp, cols, I, J, float(Kloc[a, b]))

    def _assemble_system(self):
        self._zero_system()

        elem_id = 0
        for iye in range(self.nye):
            for ixe in range(self.nxe):
                w_dofs   = self.conn_w[elem_id]     # 9
                u_dofs   = self.conn_u[elem_id]     # 12
                v_dofs   = self.conn_v[elem_id]     # 12
                thx_dofs = self.conn_thx[elem_id]   # 6
                thy_dofs = self.conn_thy[elem_id]   # 6

                blocks = self.element.get_kelem(
                    E=self.E, nu=self.nu_poiss, thick=self.thick,
                    dx=self.dx, dy=self.dy,
                    ixe=ixe, nxe=self.nxe, iye=iye, nye=self.nye
                )

                (
                    Kww,  Kwu,  Kwv,  Kwtx,  Kwty,
                    Kuw,  Kuu,  Kuv,  Kutx,  Kuty,
                    Kvw,  Kvu,  Kvv,  Kvtx,  Kvty,
                    Ktxw, Ktxu, Ktxv, Ktxtx, Ktxty,
                    Ktyw, Ktyu, Ktyv, Ktytx, Ktyty
                ) = blocks

                # element rhs (optional)
                fw = np.zeros(9)
                fu = np.zeros(12)
                fv = np.zeros(12)
                ftx = np.zeros(6)
                fty = np.zeros(6)
                if hasattr(self.element, "get_felem"):
                    x0 = ixe * self.dx
                    y0 = iye * self.dy
                    out = self.element.get_felem(
                        self.load_fcn, x0, y0, self.dx, self.dy,
                        ixe=ixe, nxe=self.nxe, iye=iye, nye=self.nye
                    )
                    if isinstance(out, tuple) and len(out) == 5:
                        fw, fu, fv, ftx, fty = out
                    else:
                        fw = out

                # add rhs
                for a, I in enumerate(w_dofs):   self.fw[I]   += fw[a]
                for a, I in enumerate(u_dofs):   self.fu[I]   += fu[a]
                for a, I in enumerate(v_dofs):   self.fv[I]   += fv[a]
                for a, I in enumerate(thx_dofs): self.fthx[I] += ftx[a]
                for a, I in enumerate(thy_dofs): self.fthy[I] += fty[a]

                # scatter 25 blocks
                # w row
                self._scatter_block(self.k_ww,   self.ww_rowp,   self.k_ww.indices,   w_dofs,   w_dofs,   Kww)
                self._scatter_block(self.k_wu,   self.wu_rowp,   self.k_wu.indices,   w_dofs,   u_dofs,   Kwu)
                self._scatter_block(self.k_wv,   self.wv_rowp,   self.k_wv.indices,   w_dofs,   v_dofs,   Kwv)
                self._scatter_block(self.k_wthx, self.wthx_rowp, self.k_wthx.indices, w_dofs,   thx_dofs, Kwtx)
                self._scatter_block(self.k_wthy, self.wthy_rowp, self.k_wthy.indices, w_dofs,   thy_dofs, Kwty)

                # u row
                self._scatter_block(self.k_uw,   self.uw_rowp,   self.k_uw.indices,   u_dofs,   w_dofs,   Kuw)
                self._scatter_block(self.k_uu,   self.uu_rowp,   self.k_uu.indices,   u_dofs,   u_dofs,   Kuu)
                self._scatter_block(self.k_uv,   self.uv_rowp,   self.k_uv.indices,   u_dofs,   v_dofs,   Kuv)
                self._scatter_block(self.k_uthx, self.uthx_rowp, self.k_uthx.indices, u_dofs,   thx_dofs, Kutx)
                self._scatter_block(self.k_uthy, self.uthy_rowp, self.k_uthy.indices, u_dofs,   thy_dofs, Kuty)

                # v row
                self._scatter_block(self.k_vw,   self.vw_rowp,   self.k_vw.indices,   v_dofs,   w_dofs,   Kvw)
                self._scatter_block(self.k_vu,   self.vu_rowp,   self.k_vu.indices,   v_dofs,   u_dofs,   Kvu)
                self._scatter_block(self.k_vv,   self.vv_rowp,   self.k_vv.indices,   v_dofs,   v_dofs,   Kvv)
                self._scatter_block(self.k_vthx, self.vthx_rowp, self.k_vthx.indices, v_dofs,   thx_dofs, Kvtx)
                self._scatter_block(self.k_vthy, self.vthy_rowp, self.k_vthy.indices, v_dofs,   thy_dofs, Kvty)

                # thx row
                self._scatter_block(self.k_thxw,   self.thxw_rowp,   self.k_thxw.indices,   thx_dofs, w_dofs,   Ktxw)
                self._scatter_block(self.k_thxu,   self.thxu_rowp,   self.k_thxu.indices,   thx_dofs, u_dofs,   Ktxu)
                self._scatter_block(self.k_thxv,   self.thxv_rowp,   self.k_thxv.indices,   thx_dofs, v_dofs,   Ktxv)
                self._scatter_block(self.k_thxthx, self.thxthx_rowp, self.k_thxthx.indices, thx_dofs, thx_dofs, Ktxtx)
                self._scatter_block(self.k_thxthy, self.thxthy_rowp, self.k_thxthy.indices, thx_dofs, thy_dofs, Ktxty)

                # thy row
                self._scatter_block(self.k_thyw,   self.thyw_rowp,   self.k_thyw.indices,   thy_dofs, w_dofs,   Ktyw)
                self._scatter_block(self.k_thyu,   self.thyu_rowp,   self.k_thyu.indices,   thy_dofs, u_dofs,   Ktyu)
                self._scatter_block(self.k_thyv,   self.thyv_rowp,   self.k_thyv.indices,   thy_dofs, v_dofs,   Ktyv)
                self._scatter_block(self.k_thythx, self.thythx_rowp, self.k_thythx.indices, thy_dofs, thx_dofs, Ktytx)
                self._scatter_block(self.k_thythy, self.thythy_rowp, self.k_thythy.indices, thy_dofs, thy_dofs, Ktyty)

                elem_id += 1

        # rebuild global view
        self._rebuild_global_bmat()

        # assemble global force vector
        self.force[:] = 0.0
        self.force[self.off_w:self.off_u] = self.fw
        self.force[self.off_u:self.off_v] = self.fu
        self.force[self.off_v:self.off_thx] = self.fv
        self.force[self.off_thx:self.off_thy] = self.fthx
        self.force[self.off_thy:] = self.fthy

        # print(f"{self.force=}")

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

        for dof in self.bcs:
            K[:, dof] = 0.0
            K[dof, dof] = 1.0

        self.kmat = K.tocsr()
        self.force = f

    # -------------------------------------------------------------------------
    # Solve hook
    # -------------------------------------------------------------------------
    def direct_solve(self):
        self._assemble_system()
        self.u = sp.linalg.spsolve(self.kmat.copy(), self.force)
        return self.u

    # -------------------------------------------------------------------------
    # Multigrid transfer wrappers (delegate to element)
    # -------------------------------------------------------------------------
    def _assemble_prolongation(self):
        self.element._assemble_prolongation(self.nxe)

    def prolongate(self, coarse_soln: np.ndarray):
        nxe_c = self.nxe // 2
        nye_c = self.nye // 2
        return self.element.prolongate(coarse_soln, nxe_c, nye_c)

    def restrict_defect(self, fine_defect: np.ndarray):
        return self.element.restrict_defect(fine_defect, self.nxe, self.nye)

    # -------------------------------------------------------------------------
    # Plot (updated layout: no w2)
    # -------------------------------------------------------------------------
    def _plot_disp(self, disp_mag: float = 0.2, mode: str = "w", ax=None):
        if self.u is None:
            raise RuntimeError("Run direct_solve() first.")

        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import matplotlib.cm as cm

        mode = mode.lower()

        off_w   = 0
        off_u   = off_w   + self.nw
        off_v   = off_u   + self.nu_d
        off_thx = off_v   + self.nv
        off_thy = off_thx + self.nthx

        if mode == "w":
            vec = self.u[off_w:off_w + self.nw]
            nx, ny = self.nx_w, self.ny_w
            label = "w"
        elif mode == "u":
            vec = self.u[off_u:off_u + self.nu_d]
            nx, ny = self.nx_u, self.ny_u
            label = "u"
        elif mode == "v":
            vec = self.u[off_v:off_v + self.nv]
            nx, ny = self.nx_v, self.ny_v
            label = "v"
        elif mode == "thx":
            vec = self.u[off_thx:off_thx + self.nthx]
            nx, ny = self.nx_thx, self.ny_thx
            label = "thx"
        elif mode == "thy":
            vec = self.u[off_thy:off_thy + self.nthy]
            nx, ny = self.nx_thy, self.ny_thy
            label = "thy"
        else:
            raise ValueError("Unknown mode. Use one of ['w','u','v','thx','thy'].")

        vec_nrm = np.max(np.abs(vec))
        print(f"{label} disp nrm: {vec_nrm:.4e}")

        V = vec.reshape((ny, nx))

        x = np.linspace(0.0, self.Lx, nx)
        # s = np.linspace(-self.Ly, 0.0, ny)
        s = np.arange(nx) * self.dy
        X, S = np.meshgrid(x, s, indexing="xy")
        Phi = S / self.radius

        X = 1.0 - X

        R = V
        orig_mag = float(np.max(np.abs(R)))
        scale_factor = (disp_mag / orig_mag) if orig_mag > 0 else 1.0
        Rdef = R * scale_factor

        # Y = (self.radius + Rdef) * np.sin(Phi)
        # Z = (self.radius + Rdef) * np.cos(Phi)
        Y = (self.radius + Rdef) * -np.cos(Phi)
        Z = (self.radius + Rdef) * np.sin(Phi)

        C = np.abs(V)
        C_face = 0.25 * (C[:-1, :-1] + C[1:, :-1] + C[:-1, 1:] + C[1:, 1:])

        norm = mcolors.Normalize(vmin=float(C_face.min()), vmax=float(C_face.max()))
        cmap = cm.get_cmap("viridis")
        facecolors = cmap(norm(C_face))

        created_fig = False
        if ax is None:
            fig = plt.figure(figsize=(9, 6))
            ax = fig.add_subplot(111, projection="3d")
            created_fig = True
        else:
            fig = ax.figure

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
        ax.set_title(f"color={label}")
        ax.view_init(elev=25, azim=-135)
        ax.set_box_aspect((1,1,1))

        if created_fig:
            mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array([])
            fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.08, label=f"|{label}|")
            plt.tight_layout()
            plt.show()

        return ax

    def plot_disp(self, disp_mag: float = 0.2, mode: str = "all"):
        import matplotlib.pyplot as plt

        if isinstance(mode, str) and mode.lower() == "all":
            modes = ["w", "u", "v", "thx", "thy"]

            fig = plt.figure(figsize=(16, 7))
            axs = [fig.add_subplot(2, 3, k + 1, projection="3d") for k in range(5)]

            for i, m in enumerate(modes):
                self._plot_disp(disp_mag=disp_mag, mode=m, ax=axs[i])

            plt.tight_layout()
            plt.show()
            return

        self._plot_disp(disp_mag=disp_mag, mode=mode, ax=None)