import numpy as np
import scipy.sparse as sp

# =============================================================================
# 2D Assembler: structured grid, block-CSR (6x6 blocks), mixed DRIG cylinder (open hoop)
# =============================================================================
class MixedDeRhamIGACylinderAssembler:
    """
    Structured 2D IGA cylinder discretization on (x,s) for the mixed p=3 element:

    Global unknown ordering:
      U = [ w, w2, u, v, thx, thy ]

    Field spaces (match MixedDeRhamIGACylinderElement docstring):
      w    : (p3,p3)  -> (nxe+3) x (nye+3)
      w2   : (p3,p1)  -> (nxe+3) x (nye+1)
      u    : (p2,p3)  -> (nxe+2) x (nye+3)
      v    : (p3,p2)  -> (nxe+3) x (nye+2)
      thx  : (p3,p2)  -> (nxe+3) x (nye+2)
      thy  : (p2,p3)  -> (nxe+2) x (nye+3)

    Element interface expected:
      element.get_kelem(E, nu, thick, dx, dy, ixe, nxe, iye, nye) -> 36 blocks (6x6)
      element.get_felem(load_fcn, x0, y0, dx, dy, ixe, nxe, iye, nye) -> (fw, fw2, fu, fv, ftx, fty)

    BCs (strong):
      - clamped=False:
          w = 0 on all 4 sides  (w grid)
          u = 0 on x=0 edge     (u grid)
          v = 0 on y=0 edge     (v grid)
          (w2, thx, thy free)
      - clamped=True:
          w = 0 on all 4 sides
          u,v,thx,thy = 0 on all 4 sides (on their own grids)
          (w2 left free by default; consistent with your element’s apply_bcs_2d)
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
        self.load_fcn = load_fcn
        self.clamped = bool(clamped)

        self.dof_per_node = 6 # not really able to be used though.. cause vertex+edges not nodes

        # keep element flag in sync
        if hasattr(self.element, "clamped"):
            self.element.clamped = self.clamped

        # -----------------------------
        # Grid sizes (match spaces)
        # -----------------------------
        self.nx_w,   self.ny_w   = self.nxe + 3, self.nye + 3   # (p3,p3)
        self.nx_w2,  self.ny_w2  = self.nxe + 3, self.nye + 1   # (p3,p1)
        self.nx_u,   self.ny_u   = self.nxe + 2, self.nye + 3   # (p2,p3)
        self.nx_v,   self.ny_v   = self.nxe + 3, self.nye + 2   # (p3,p2)
        self.nx_thx, self.ny_thx = self.nx_v,    self.ny_v      # (p3,p2)
        self.nx_thy, self.ny_thy = self.nx_u,    self.ny_u      # (p2,p3)

        self.nw   = self.nx_w   * self.ny_w
        self.nw2  = self.nx_w2  * self.ny_w2
        self.nu_d = self.nx_u   * self.ny_u
        self.nv   = self.nx_v   * self.ny_v
        self.nthx = self.nx_thx * self.ny_thx
        self.nthy = self.nx_thy * self.ny_thy

        self.N = self.nw + self.nw2 + self.nu_d + self.nv + self.nthx + self.nthy

        # offsets
        self.off_w   = 0
        self.off_w2  = self.off_w  + self.nw
        self.off_u   = self.off_w2 + self.nw2
        self.off_v   = self.off_u  + self.nu_d
        self.off_thx = self.off_v  + self.nv
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
        (self.conn_w, self.conn_w2, self.conn_u,
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
        conn_w, conn_w2, conn_u, conn_v, conn_thx, conn_thy = ([] for _ in range(6))

        elem_id = 0
        for ey in range(self.nye):
            for ex in range(self.nxe):

                # w: 4x4 on (nx_w, ny_w)
                wloc = []
                for ly in range(4):
                    jy = ey + ly
                    for lx in range(4):
                        ix = ex + lx
                        wloc.append(self._node(ix, jy, self.nx_w))
                conn_w.append(wloc)

                # w2: 4x2 on (nx_w2, ny_w2)
                w2loc = []
                for ly in range(2):
                    jy = ey + ly
                    for lx in range(4):
                        ix = ex + lx
                        w2loc.append(self._node(ix, jy, self.nx_w2))
                conn_w2.append(w2loc)

                # v, thx: 4x3 on (nx_v, ny_v)
                vloc = []
                for ly in range(3):
                    jy = ey + ly
                    for lx in range(4):
                        ix = ex + lx
                        vloc.append(self._node(ix, jy, self.nx_v))
                conn_v.append(vloc)
                conn_thx.append(list(vloc))

                # u, thy: 3x4 on (nx_u, ny_u)
                uloc = []
                for ly in range(4):
                    jy = ey + ly
                    for lx in range(3):
                        ix = ex + lx
                        uloc.append(self._node(ix, jy, self.nx_u))
                conn_u.append(uloc)
                conn_thy.append(list(uloc))

                elem_id += 1

        return conn_w, conn_w2, conn_u, conn_v, conn_thx, conn_thy

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

            # NOTE: w2 intentionally NOT constrained (aux field)
        else:
            # u: x=0 edge
            for j in range(self.ny_u):
                i = 0
                bcs.append(self.off_u + self._node(i, j, self.nx_u))

            # v: y=0 edge
            j = 0
            for i in range(self.nx_v):
                bcs.append(self.off_v + self._node(i, j, self.nx_v))

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
        gw2 = (self.nx_w2, self.ny_w2)
        gu  = (self.nx_u,  self.ny_u)
        gv  = (self.nx_v,  self.ny_v)
        gtx = (self.nx_thx, self.ny_thx)
        gty = (self.nx_thy, self.ny_thy)

        # Stencils (generous / safe)
        # Think: p3 support -> +/-3, p2 support -> +/-2, p1 support -> +/-1.
        # Use “row vs col” conservative ranges, matching your old strategy.

        # --- w rows
        self.ww_rowp,   self.ww_cols,   self.ww_nnz   = self._build_pattern_from_stencil(self.nw,  gw,  gw,  (-3, 3), (-3, 3), per)
        self.ww2_rowp,  self.ww2_cols,  self.ww2_nnz  = self._build_pattern_from_stencil(self.nw,  gw,  gw2, (-3, 3), (-3, 1), per)
        self.wu_rowp,   self.wu_cols,   self.wu_nnz   = self._build_pattern_from_stencil(self.nw,  gw,  gu,  (-3, 2), (-3, 3), per)
        self.wv_rowp,   self.wv_cols,   self.wv_nnz   = self._build_pattern_from_stencil(self.nw,  gw,  gv,  (-3, 3), (-3, 2), per)
        self.wthx_rowp, self.wthx_cols, self.wthx_nnz = self._build_pattern_from_stencil(self.nw,  gw,  gtx, (-3, 3), (-3, 2), per)
        self.wthy_rowp, self.wthy_cols, self.wthy_nnz = self._build_pattern_from_stencil(self.nw,  gw,  gty, (-3, 2), (-3, 3), per)

        # --- w2 rows
        self.w2w_rowp,   self.w2w_cols,   self.w2w_nnz   = self._build_pattern_from_stencil(self.nw2, gw2, gw,  (-3, 3), (-1, 3), per)
        self.w2w2_rowp,  self.w2w2_cols,  self.w2w2_nnz  = self._build_pattern_from_stencil(self.nw2, gw2, gw2, (-3, 3), (-1, 1), per)
        self.w2u_rowp,   self.w2u_cols,   self.w2u_nnz   = self._build_pattern_from_stencil(self.nw2, gw2, gu,  (-3, 2), (-1, 3), per)
        self.w2v_rowp,   self.w2v_cols,   self.w2v_nnz   = self._build_pattern_from_stencil(self.nw2, gw2, gv,  (-3, 3), (-1, 2), per)
        self.w2thx_rowp, self.w2thx_cols, self.w2thx_nnz = self._build_pattern_from_stencil(self.nw2, gw2, gtx, (-3, 3), (-1, 2), per)
        self.w2thy_rowp, self.w2thy_cols, self.w2thy_nnz = self._build_pattern_from_stencil(self.nw2, gw2, gty, (-3, 2), (-1, 3), per)

        # --- u rows
        self.uw_rowp,   self.uw_cols,   self.uw_nnz   = self._build_pattern_from_stencil(self.nu_d, gu, gw,  (-2, 3), (-3, 3), per)
        self.uw2_rowp,  self.uw2_cols,  self.uw2_nnz  = self._build_pattern_from_stencil(self.nu_d, gu, gw2, (-2, 3), (-3, 1), per)
        self.uu_rowp,   self.uu_cols,   self.uu_nnz   = self._build_pattern_from_stencil(self.nu_d, gu, gu,  (-2, 2), (-3, 3), per)
        self.uv_rowp,   self.uv_cols,   self.uv_nnz   = self._build_pattern_from_stencil(self.nu_d, gu, gv,  (-2, 3), (-3, 2), per)
        self.uthx_rowp, self.uthx_cols, self.uthx_nnz = self._build_pattern_from_stencil(self.nu_d, gu, gtx, (-2, 3), (-3, 2), per)
        self.uthy_rowp, self.uthy_cols, self.uthy_nnz = self._build_pattern_from_stencil(self.nu_d, gu, gty, (-2, 2), (-3, 3), per)

        # --- v rows
        self.vw_rowp,   self.vw_cols,   self.vw_nnz   = self._build_pattern_from_stencil(self.nv, gv, gw,  (-3, 3), (-2, 3), per)
        self.vw2_rowp,  self.vw2_cols,  self.vw2_nnz  = self._build_pattern_from_stencil(self.nv, gv, gw2, (-3, 3), (-2, 1), per)
        self.vu_rowp,   self.vu_cols,   self.vu_nnz   = self._build_pattern_from_stencil(self.nv, gv, gu,  (-3, 2), (-2, 3), per)
        self.vv_rowp,   self.vv_cols,   self.vv_nnz   = self._build_pattern_from_stencil(self.nv, gv, gv,  (-3, 3), (-2, 2), per)
        self.vthx_rowp, self.vthx_cols, self.vthx_nnz = self._build_pattern_from_stencil(self.nv, gv, gtx, (-3, 3), (-2, 2), per)
        self.vthy_rowp, self.vthy_cols, self.vthy_nnz = self._build_pattern_from_stencil(self.nv, gv, gty, (-3, 2), (-2, 3), per)

        # --- thx rows (same grid as v)
        self.thxw_rowp,   self.thxw_cols,   self.thxw_nnz   = self._build_pattern_from_stencil(self.nthx, gtx, gw,  (-3, 3), (-2, 3), per)
        self.thxw2_rowp,  self.thxw2_cols,  self.thxw2_nnz  = self._build_pattern_from_stencil(self.nthx, gtx, gw2, (-3, 3), (-2, 1), per)
        self.thxu_rowp,   self.thxu_cols,   self.thxu_nnz   = self._build_pattern_from_stencil(self.nthx, gtx, gu,  (-3, 2), (-2, 3), per)
        self.thxv_rowp,   self.thxv_cols,   self.thxv_nnz   = self._build_pattern_from_stencil(self.nthx, gtx, gv,  (-3, 3), (-2, 2), per)
        self.thxthx_rowp, self.thxthx_cols, self.thxthx_nnz = self._build_pattern_from_stencil(self.nthx, gtx, gtx, (-3, 3), (-2, 2), per)
        self.thxthy_rowp, self.thxthy_cols, self.thxthy_nnz = self._build_pattern_from_stencil(self.nthx, gtx, gty, (-3, 2), (-2, 3), per)

        # --- thy rows (same grid as u)
        self.thyw_rowp,   self.thyw_cols,   self.thyw_nnz   = self._build_pattern_from_stencil(self.nthy, gty, gw,  (-2, 3), (-3, 3), per)
        self.thyw2_rowp,  self.thyw2_cols,  self.thyw2_nnz  = self._build_pattern_from_stencil(self.nthy, gty, gw2, (-2, 3), (-3, 1), per)
        self.thyu_rowp,   self.thyu_cols,   self.thyu_nnz   = self._build_pattern_from_stencil(self.nthy, gty, gu,  (-2, 2), (-3, 3), per)
        self.thyv_rowp,   self.thyv_cols,   self.thyv_nnz   = self._build_pattern_from_stencil(self.nthy, gty, gv,  (-2, 3), (-3, 2), per)
        self.thythx_rowp, self.thythx_cols, self.thythx_nnz = self._build_pattern_from_stencil(self.nthy, gty, gtx, (-2, 3), (-3, 2), per)
        self.thythy_rowp, self.thythy_cols, self.thythy_nnz = self._build_pattern_from_stencil(self.nthy, gty, gty, (-2, 2), (-3, 3), per)

    # -------------------------------------------------------------------------
    # Allocate blocks + global bmat view
    # -------------------------------------------------------------------------
    def _alloc_blocks(self):
        z = np.zeros

        # w-row
        self.k_ww   = sp.csr_matrix((z(self.ww_nnz),   self.ww_cols,   self.ww_rowp),   shape=(self.nw,  self.nw))
        self.k_ww2  = sp.csr_matrix((z(self.ww2_nnz),  self.ww2_cols,  self.ww2_rowp),  shape=(self.nw,  self.nw2))
        self.k_wu   = sp.csr_matrix((z(self.wu_nnz),   self.wu_cols,   self.wu_rowp),   shape=(self.nw,  self.nu_d))
        self.k_wv   = sp.csr_matrix((z(self.wv_nnz),   self.wv_cols,   self.wv_rowp),   shape=(self.nw,  self.nv))
        self.k_wthx = sp.csr_matrix((z(self.wthx_nnz), self.wthx_cols, self.wthx_rowp), shape=(self.nw,  self.nthx))
        self.k_wthy = sp.csr_matrix((z(self.wthy_nnz), self.wthy_cols, self.wthy_rowp), shape=(self.nw,  self.nthy))

        # w2-row
        self.k_w2w   = sp.csr_matrix((z(self.w2w_nnz),   self.w2w_cols,   self.w2w_rowp),   shape=(self.nw2, self.nw))
        self.k_w2w2  = sp.csr_matrix((z(self.w2w2_nnz),  self.w2w2_cols,  self.w2w2_rowp),  shape=(self.nw2, self.nw2))
        self.k_w2u   = sp.csr_matrix((z(self.w2u_nnz),   self.w2u_cols,   self.w2u_rowp),   shape=(self.nw2, self.nu_d))
        self.k_w2v   = sp.csr_matrix((z(self.w2v_nnz),   self.w2v_cols,   self.w2v_rowp),   shape=(self.nw2, self.nv))
        self.k_w2thx = sp.csr_matrix((z(self.w2thx_nnz), self.w2thx_cols, self.w2thx_rowp), shape=(self.nw2, self.nthx))
        self.k_w2thy = sp.csr_matrix((z(self.w2thy_nnz), self.w2thy_cols, self.w2thy_rowp), shape=(self.nw2, self.nthy))

        # u-row
        self.k_uw   = sp.csr_matrix((z(self.uw_nnz),   self.uw_cols,   self.uw_rowp),   shape=(self.nu_d, self.nw))
        self.k_uw2  = sp.csr_matrix((z(self.uw2_nnz),  self.uw2_cols,  self.uw2_rowp),  shape=(self.nu_d, self.nw2))
        self.k_uu   = sp.csr_matrix((z(self.uu_nnz),   self.uu_cols,   self.uu_rowp),   shape=(self.nu_d, self.nu_d))
        self.k_uv   = sp.csr_matrix((z(self.uv_nnz),   self.uv_cols,   self.uv_rowp),   shape=(self.nu_d, self.nv))
        self.k_uthx = sp.csr_matrix((z(self.uthx_nnz), self.uthx_cols, self.uthx_rowp), shape=(self.nu_d, self.nthx))
        self.k_uthy = sp.csr_matrix((z(self.uthy_nnz), self.uthy_cols, self.uthy_rowp), shape=(self.nu_d, self.nthy))

        # v-row
        self.k_vw   = sp.csr_matrix((z(self.vw_nnz),   self.vw_cols,   self.vw_rowp),   shape=(self.nv, self.nw))
        self.k_vw2  = sp.csr_matrix((z(self.vw2_nnz),  self.vw2_cols,  self.vw2_rowp),  shape=(self.nv, self.nw2))
        self.k_vu   = sp.csr_matrix((z(self.vu_nnz),   self.vu_cols,   self.vu_rowp),   shape=(self.nv, self.nu_d))
        self.k_vv   = sp.csr_matrix((z(self.vv_nnz),   self.vv_cols,   self.vv_rowp),   shape=(self.nv, self.nv))
        self.k_vthx = sp.csr_matrix((z(self.vthx_nnz), self.vthx_cols, self.vthx_rowp), shape=(self.nv, self.nthx))
        self.k_vthy = sp.csr_matrix((z(self.vthy_nnz), self.vthy_cols, self.vthy_rowp), shape=(self.nv, self.nthy))

        # thx-row
        self.k_thxw   = sp.csr_matrix((z(self.thxw_nnz),   self.thxw_cols,   self.thxw_rowp),   shape=(self.nthx, self.nw))
        self.k_thxw2  = sp.csr_matrix((z(self.thxw2_nnz),  self.thxw2_cols,  self.thxw2_rowp),  shape=(self.nthx, self.nw2))
        self.k_thxu   = sp.csr_matrix((z(self.thxu_nnz),   self.thxu_cols,   self.thxu_rowp),   shape=(self.nthx, self.nu_d))
        self.k_thxv   = sp.csr_matrix((z(self.thxv_nnz),   self.thxv_cols,   self.thxv_rowp),   shape=(self.nthx, self.nv))
        self.k_thxthx = sp.csr_matrix((z(self.thxthx_nnz), self.thxthx_cols, self.thxthx_rowp), shape=(self.nthx, self.nthx))
        self.k_thxthy = sp.csr_matrix((z(self.thxthy_nnz), self.thxthy_cols, self.thxthy_rowp), shape=(self.nthx, self.nthy))

        # thy-row
        self.k_thyw   = sp.csr_matrix((z(self.thyw_nnz),   self.thyw_cols,   self.thyw_rowp),   shape=(self.nthy, self.nw))
        self.k_thyw2  = sp.csr_matrix((z(self.thyw2_nnz),  self.thyw2_cols,  self.thyw2_rowp),  shape=(self.nthy, self.nw2))
        self.k_thyu   = sp.csr_matrix((z(self.thyu_nnz),   self.thyu_cols,   self.thyu_rowp),   shape=(self.nthy, self.nu_d))
        self.k_thyv   = sp.csr_matrix((z(self.thyv_nnz),   self.thyv_cols,   self.thyv_rowp),   shape=(self.nthy, self.nv))
        self.k_thythx = sp.csr_matrix((z(self.thythx_nnz), self.thythx_cols, self.thythx_rowp), shape=(self.nthy, self.nthx))
        self.k_thythy = sp.csr_matrix((z(self.thythy_nnz), self.thythy_cols, self.thythy_rowp), shape=(self.nthy, self.nthy))

        # RHS blocks
        self.fw   = np.zeros(self.nw)
        self.fw2  = np.zeros(self.nw2)
        self.fu   = np.zeros(self.nu_d)
        self.fv   = np.zeros(self.nv)
        self.fthx = np.zeros(self.nthx)
        self.fthy = np.zeros(self.nthy)

        self._rebuild_global_bmat()

    def _rebuild_global_bmat(self):
        self.kmat = sp.bmat(
            [
                [self.k_ww,    self.k_ww2,   self.k_wu,   self.k_wv,   self.k_wthx,   self.k_wthy],
                [self.k_w2w,   self.k_w2w2,  self.k_w2u,  self.k_w2v,  self.k_w2thx,  self.k_w2thy],
                [self.k_uw,    self.k_uw2,   self.k_uu,   self.k_uv,   self.k_uthx,   self.k_uthy],
                [self.k_vw,    self.k_vw2,   self.k_vu,   self.k_vv,   self.k_vthx,   self.k_vthy],
                [self.k_thxw,  self.k_thxw2, self.k_thxu, self.k_thxv, self.k_thxthx, self.k_thxthy],
                [self.k_thyw,  self.k_thyw2, self.k_thyu, self.k_thyv, self.k_thythx, self.k_thythy],
            ],
            format="csr",
        )

    # -------------------------------------------------------------------------
    # Assembly
    # -------------------------------------------------------------------------
    def _zero_system(self):
        for A in [
            self.k_ww, self.k_ww2, self.k_wu, self.k_wv, self.k_wthx, self.k_wthy,
            self.k_w2w, self.k_w2w2, self.k_w2u, self.k_w2v, self.k_w2thx, self.k_w2thy,
            self.k_uw, self.k_uw2, self.k_uu, self.k_uv, self.k_uthx, self.k_uthy,
            self.k_vw, self.k_vw2, self.k_vu, self.k_vv, self.k_vthx, self.k_vthy,
            self.k_thxw, self.k_thxw2, self.k_thxu, self.k_thxv, self.k_thxthx, self.k_thxthy,
            self.k_thyw, self.k_thyw2, self.k_thyu, self.k_thyv, self.k_thythx, self.k_thythy,
        ]:
            A.data[:] = 0.0

        self.fw[:] = 0.0
        self.fw2[:] = 0.0
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
                w_dofs   = self.conn_w[elem_id]     # 16
                w2_dofs  = self.conn_w2[elem_id]    # 8
                u_dofs   = self.conn_u[elem_id]     # 12
                v_dofs   = self.conn_v[elem_id]     # 12
                thx_dofs = self.conn_thx[elem_id]   # 12
                thy_dofs = self.conn_thy[elem_id]   # 12

                # element stiffness (36 blocks)
                blocks = self.element.get_kelem(
                    E=self.E, nu=self.nu_poiss, thick=self.thick,
                    dx=self.dx, dy=self.dy,
                    ixe=ixe, nxe=self.nxe, iye=iye, nye=self.nye
                )

                (
                    Kww,  Kww2, Kwu,  Kwv,  Kwtx,  Kwty,
                    Kw2w, Kw2w2, Kw2u, Kw2v, Kw2tx, Kw2ty,
                    Kuw,  Kuw2, Kuu,  Kuv,  Kutx,  Kuty,
                    Kvw,  Kvw2, Kvu,  Kvv,  Kvtx,  Kvty,
                    Ktxw, Ktxw2, Ktxu, Ktxv, Ktxtx, Ktxty,
                    Ktyw, Ktyw2, Ktyu, Ktyv, Ktytx, Ktyty
                ) = blocks

                # element rhs (optional)
                fw = np.zeros(16)
                fw2 = np.zeros(8)
                fu = np.zeros(12)
                fv = np.zeros(12)
                ftx = np.zeros(12)
                fty = np.zeros(12)
                if hasattr(self.element, "get_felem"):
                    x0 = ixe * self.dx
                    y0 = iye * self.dy
                    out = self.element.get_felem(
                        self.load_fcn, x0, y0, self.dx, self.dy,
                        ixe=ixe, nxe=self.nxe, iye=iye, nye=self.nye
                    )
                    if isinstance(out, tuple) and len(out) == 6:
                        fw, fw2, fu, fv, ftx, fty = out
                    else:
                        # best-effort fallback: if only fw returned
                        fw = out

                # add rhs
                for a, I in enumerate(w_dofs):   self.fw[I]  += fw[a]
                for a, I in enumerate(w2_dofs):  self.fw2[I] += fw2[a]
                for a, I in enumerate(u_dofs):   self.fu[I]  += fu[a]
                for a, I in enumerate(v_dofs):   self.fv[I]  += fv[a]
                for a, I in enumerate(thx_dofs): self.fthx[I] += ftx[a]
                for a, I in enumerate(thy_dofs): self.fthy[I] += fty[a]

                # scatter all 36 blocks
                # w row
                self._scatter_block(self.k_ww,   self.ww_rowp,   self.k_ww.indices,   w_dofs,  w_dofs,  Kww)
                self._scatter_block(self.k_ww2,  self.ww2_rowp,  self.k_ww2.indices,  w_dofs,  w2_dofs, Kww2)
                self._scatter_block(self.k_wu,   self.wu_rowp,   self.k_wu.indices,   w_dofs,  u_dofs,  Kwu)
                self._scatter_block(self.k_wv,   self.wv_rowp,   self.k_wv.indices,   w_dofs,  v_dofs,  Kwv)
                self._scatter_block(self.k_wthx, self.wthx_rowp, self.k_wthx.indices, w_dofs,  thx_dofs, Kwtx)
                self._scatter_block(self.k_wthy, self.wthy_rowp, self.k_wthy.indices, w_dofs,  thy_dofs, Kwty)

                # w2 row
                self._scatter_block(self.k_w2w,   self.w2w_rowp,   self.k_w2w.indices,   w2_dofs, w_dofs,   Kw2w)
                self._scatter_block(self.k_w2w2,  self.w2w2_rowp,  self.k_w2w2.indices,  w2_dofs, w2_dofs,  Kw2w2)
                self._scatter_block(self.k_w2u,   self.w2u_rowp,   self.k_w2u.indices,   w2_dofs, u_dofs,   Kw2u)
                self._scatter_block(self.k_w2v,   self.w2v_rowp,   self.k_w2v.indices,   w2_dofs, v_dofs,   Kw2v)
                self._scatter_block(self.k_w2thx, self.w2thx_rowp, self.k_w2thx.indices, w2_dofs, thx_dofs, Kw2tx)
                self._scatter_block(self.k_w2thy, self.w2thy_rowp, self.k_w2thy.indices, w2_dofs, thy_dofs, Kw2ty)

                # u row
                self._scatter_block(self.k_uw,   self.uw_rowp,   self.k_uw.indices,   u_dofs, w_dofs,  Kuw)
                self._scatter_block(self.k_uw2,  self.uw2_rowp,  self.k_uw2.indices,  u_dofs, w2_dofs, Kuw2)
                self._scatter_block(self.k_uu,   self.uu_rowp,   self.k_uu.indices,   u_dofs, u_dofs,  Kuu)
                self._scatter_block(self.k_uv,   self.uv_rowp,   self.k_uv.indices,   u_dofs, v_dofs,  Kuv)
                self._scatter_block(self.k_uthx, self.uthx_rowp, self.k_uthx.indices, u_dofs, thx_dofs, Kutx)
                self._scatter_block(self.k_uthy, self.uthy_rowp, self.k_uthy.indices, u_dofs, thy_dofs, Kuty)

                # v row
                self._scatter_block(self.k_vw,   self.vw_rowp,   self.k_vw.indices,   v_dofs, w_dofs,  Kvw)
                self._scatter_block(self.k_vw2,  self.vw2_rowp,  self.k_vw2.indices,  v_dofs, w2_dofs, Kvw2)
                self._scatter_block(self.k_vu,   self.vu_rowp,   self.k_vu.indices,   v_dofs, u_dofs,  Kvu)
                self._scatter_block(self.k_vv,   self.vv_rowp,   self.k_vv.indices,   v_dofs, v_dofs,  Kvv)
                self._scatter_block(self.k_vthx, self.vthx_rowp, self.k_vthx.indices, v_dofs, thx_dofs, Kvtx)
                self._scatter_block(self.k_vthy, self.vthy_rowp, self.k_vthy.indices, v_dofs, thy_dofs, Kvty)

                # thx row
                self._scatter_block(self.k_thxw,   self.thxw_rowp,   self.k_thxw.indices,   thx_dofs, w_dofs,  Ktxw)
                self._scatter_block(self.k_thxw2,  self.thxw2_rowp,  self.k_thxw2.indices,  thx_dofs, w2_dofs, Ktxw2)
                self._scatter_block(self.k_thxu,   self.thxu_rowp,   self.k_thxu.indices,   thx_dofs, u_dofs,  Ktxu)
                self._scatter_block(self.k_thxv,   self.thxv_rowp,   self.k_thxv.indices,   thx_dofs, v_dofs,  Ktxv)
                self._scatter_block(self.k_thxthx, self.thxthx_rowp, self.k_thxthx.indices, thx_dofs, thx_dofs, Ktxtx)
                self._scatter_block(self.k_thxthy, self.thxthy_rowp, self.k_thxthy.indices, thx_dofs, thy_dofs, Ktxty)

                # thy row
                self._scatter_block(self.k_thyw,   self.thyw_rowp,   self.k_thyw.indices,   thy_dofs, w_dofs,  Ktyw)
                self._scatter_block(self.k_thyw2,  self.thyw2_rowp,  self.k_thyw2.indices,  thy_dofs, w2_dofs, Ktyw2)
                self._scatter_block(self.k_thyu,   self.thyu_rowp,   self.k_thyu.indices,   thy_dofs, u_dofs,  Ktyu)
                self._scatter_block(self.k_thyv,   self.thyv_rowp,   self.k_thyv.indices,   thy_dofs, v_dofs,  Ktyv)
                self._scatter_block(self.k_thythx, self.thythx_rowp, self.k_thythx.indices, thy_dofs, thx_dofs, Ktytx)
                self._scatter_block(self.k_thythy, self.thythy_rowp, self.k_thythy.indices, thy_dofs, thy_dofs, Ktyty)

                elem_id += 1

        # rebuild global view
        self._rebuild_global_bmat()

        # assemble global force vector
        self.force[:] = 0.0
        self.force[self.off_w:self.off_w2] = self.fw
        self.force[self.off_w2:self.off_u] = self.fw2
        self.force[self.off_u:self.off_v] = self.fu
        self.force[self.off_v:self.off_thx] = self.fv
        self.force[self.off_thx:self.off_thy] = self.fthx
        self.force[self.off_thy:] = self.fthy

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
    def prolongate(self, coarse_soln: np.ndarray):
        nxe_c = self.nxe // 2
        nye_c = self.nye // 2
        return self.element.prolongate(coarse_soln, nxe_c, nye_c)

    def restrict_defect(self, fine_defect: np.ndarray):
        # called on coarse grid object; pass its (nxe,nye) as coarse sizes
        return self.element.restrict_defect(fine_defect, self.nxe, self.nye)

    def _plot_disp(self, disp_mag: float = 0.2, mode: str = "w", ax=None):
        """
        PRIVATE: plot a single field onto `ax` (3D) if provided, else create a figure.

        Fixed deformation rule (NO INPUT):
        - mode == "w": deform with w
        - else:        undeformed (Rdef = 0)  [grids differ, so no u/v/etc deformation]
        """
        if self.u is None:
            raise RuntimeError("Run direct_solve() first.")

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import matplotlib.cm as cm

        mode = mode.lower()

        # global ordering: [w, w2, u, v, thx, thy]
        off_w   = 0
        off_w2  = off_w   + self.nw
        off_u   = off_w2  + self.nw2
        off_v   = off_u   + self.nu_d
        off_thx = off_v   + self.nv
        off_thy = off_thx + self.nthx

        if mode == "w":
            vec = self.u[off_w:off_w + self.nw]
            nx, ny = self.nx_w, self.ny_w
            label = "w"
        elif mode == "w2":
            vec = self.u[off_w2:off_w2 + self.nw2]
            nx, ny = self.nx_w2, self.ny_w2
            label = "w2"
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
            raise ValueError("Unknown mode='{0}'. Use one of ['w','w2','u','v','thx','thy'].".format(mode))

        vec_nrm = np.max(np.abs(vec))
        print(f"{label} disp nrm: {vec_nrm:.4e}")

        V = vec.reshape((ny, nx))

        # ---- build (x, theta) grid matching that field ----
        x = np.linspace(0.0, self.Lx, nx)
        s = np.linspace(-self.Ly, 0.0, ny)
        X, S = np.meshgrid(x, s, indexing="xy")
        Phi = S / self.radius

        # keep your flip
        X = 1.0 - X

        # ---- deformation (FIXED) ----
        R = V
        orig_mag = float(np.max(np.abs(R)))
        scale_factor = (disp_mag / orig_mag) if orig_mag > 0 else 1.0
        Rdef = R * scale_factor

        # ---- geometry ----
        Y = (self.radius + Rdef) * np.sin(Phi)
        Z = (self.radius + Rdef) * np.cos(Phi)

        # ---- color by selected field ----
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

        if created_fig:
            mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array([])
            fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.08, label=f"|{label}|")
            plt.tight_layout()
            plt.show()

        return ax


    def plot_disp(self, disp_mag: float = 0.2, mode: str = "all"):
        """
        Public entrypoint.

        - mode in {'w','w2','u','v','thx','thy'}: plot one
        - mode == 'all': plot all 6 in a (2,3) grid
        """
        import matplotlib.pyplot as plt

        if isinstance(mode, str) and mode.lower() == "all":
            modes = ["w", "w2", "u", "v", "thx", "thy"]

            fig = plt.figure(figsize=(16, 10))
            axs = [fig.add_subplot(2, 3, k + 1, projection="3d") for k in range(6)]

            for i, m in enumerate(modes):
                self._plot_disp(disp_mag=disp_mag, mode=m, ax=axs[i])

            plt.tight_layout()
            plt.show()
            return

        self._plot_disp(disp_mag=disp_mag, mode=mode, ax=None)
