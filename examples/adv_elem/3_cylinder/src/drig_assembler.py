import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# =============================================================================
# 2D Assembler: structured grid, block-CSR (5x5 blocks), deRham cylinder (periodic hoop)
# =============================================================================
class DeRhamIGACylinderAssembler:
    """
    Structured 2D IGA cylinder discretization (de Rham compatible) on (x,s) where:
      x = axial in [0, Lx]
      s = hoop arc-length in [0, Ly] with periodic wrap (s=0 identified with s=Ly)

    Unknown ordering (global):
      U = [ w(0..nw-1), u(0..nu-1), v(0..nv-1), thx(0..nthx-1), thy(0..nthy-1) ]

    Field spaces (matching your DeRhamIsogeometricCylinderElement docstring):
      w   : (p2,p2)  -> 3x3 per elem
      u   : (p1,p2)  -> 2x3 per elem
      v   : (p2,p1)  -> 3x2 per elem
      thx : (p2,p1)  -> 3x2 per elem
      thy : (p1,p2)  -> 2x3 per elem

    Periodicity:
      - Hoop direction (the second param direction, "y" in your element, i.e. s) is periodic.
      - All field grids wrap in the hoop direction, and element connectivity wraps accordingly.
      - There are NO "bot/top" boundaries in hoop; only axial ends can be essential BCs.

    Notes:
      - This assumes your element.get_kelem(...) uses (dx, dy) where dy is hoop arc-length span.
      - RHS: if element has get_felem(...), we use it; otherwise RHS stays zero (common for eigen/buckling).
    """

    def __init__(
        self,
        ELEMENT,
        nxe: int,
        nye: int = None,
        E=70e9,
        nu=0.3,
        thick=1e-2,
        length: float = 1.0,          # axial length
        radius:float = 1.0,
        hoop_length:float=np.pi, # semi-cylinder
        load_fcn=lambda x, s: 0.0,    # radial load on w (optional)
        clamped: bool = False,
    ):
        self.element = ELEMENT(clamped=clamped) if callable(ELEMENT) else ELEMENT

        self.nxe = int(nxe)
        self.nye = int(nye if nye is not None else nxe)

        self.E = float(E)
        self.nu_poiss = float(nu)   # material Poisson
        self.thick = float(thick)
        self.Lx = float(length)
        self.radius = float(radius)
        self.Ly = hoop_length
        self.load_fcn = load_fcn
        self.clamped = bool(clamped)
        self.dof_per_node = 5

        # copy clamped input to element
        self.element.clamped = self.clamped

        # -----------------------------
        # Grid sizes (match spaces)
        # -----------------------------
        # (p2,p2)
        self.nx_w = self.nxe + 2
        self.ny_w = self.nye + 2

        # (p2,p1)
        self.nx_thx = self.nxe + 2
        self.ny_thx = self.nye + 1

        # (p1,p2)
        self.nx_thy  = self.nxe + 1
        self.ny_thy  = self.nye + 2

        # counts
        self.nw   = self.nx_w * self.ny_w
        self.nu  = self.nx_thy * self.ny_thy
        self.nv   = self.nx_thx * self.ny_thx
        self.nthx = self.nx_thx * self.ny_thx
        self.nthy = self.nx_thy  * self.ny_thy

        # print(f"{self.nw=}\n{self.nu=}\n{self.nv=}\n{self.nthx=}\n{self.nthy=}")
        self.N = self.nw + self.nu + self.nv + self.nthx + self.nthy

        # offsets
        self.off_w   = 0
        self.off_u   = self.off_w + self.nw
        self.off_v   = self.off_u + self.nu
        self.off_thx = self.off_v + self.nv
        self.off_thy = self.off_thx + self.nthx

        # element sizes
        self.dx = self.Lx / self.nxe
        self.dy = self.Ly / self.nye  # hoop arc-length per element

        self.dx_coord = self.Lx / (self.nx_w - 1)
        self.dy_coord = self.Ly / (self.ny_w - 1)

        # system
        self.kmat = None
        self.force = np.zeros(self.N)
        self.u = None

        # build BC list (strongly enforced u[dof]=0)
        self.bcs = self._build_bcs()

        # connectivity (structured, periodic in hoop)
        self.conn_w, self.conn_u, self.conn_v, self.conn_thx, self.conn_thy = self._build_connectivity()

        # build block-CSR patterns (safe stencils, periodic in hoop)
        self._build_block_patterns()
        self._alloc_blocks()

    # -------------------------------------------------------------------------
    # Index helpers (periodic hoop)
    # -------------------------------------------------------------------------
    @staticmethod
    def _node(i, j, nx):
        return i + nx * j

    # @staticmethod
    # def _wrap_j(j, ny):
    #     return j % ny

    # -------------------------------------------------------------------------
    # Connectivity (periodic in hoop direction)
    # -------------------------------------------------------------------------
    def _build_connectivity(self):
        conn_w   = []
        conn_u   = []
        conn_v   = []
        conn_thx = []
        conn_thy = []

        for ey in range(self.nye):
            for ex in range(self.nxe):

                # ---- w : 3x3 on (nx_w, ny_w)  (NO WRAP)
                wloc = []
                for ly in range(3):
                    jy = ey + ly
                    for lx in range(3):
                        ix = ex + lx
                        wloc.append(self._node(ix, jy, self.nx_w))
                conn_w.append(wloc)
                # conn_u.append(list(wloc))

                # ---- v, thx : 3x2 on (nx_thx, ny_thx)  (NO WRAP)
                vloc = []
                for ly in range(2):
                    jy = ey + ly
                    for lx in range(3):
                        ix = ex + lx
                        vloc.append(self._node(ix, jy, self.nx_thx))
                conn_v.append(vloc)
                conn_thx.append(list(vloc))

                # ---- u, thy : 2x3 on (nx_thy, ny_thy)  (NO WRAP)
                tyloc = []
                for ly in range(3):
                    jy = ey + ly
                    for lx in range(2):
                        ix = ex + lx
                        tyloc.append(self._node(ix, jy, self.nx_thy))
                conn_u.append(tyloc)
                conn_thy.append(list(tyloc))
                

        return conn_w, conn_u, conn_v, conn_thx, conn_thy

    def _build_bcs(self):
        """
        Boundary conditions for an open-hoop cylindrical patch in (x,s).

        Modes:
        - self.clamped == True  (Fully clamped on ALL 4 sides):
            constrain w, u, v, thx, thy on all boundary nodes
        - self.clamped == False (Simply supported-style, as you had):
            constrain w on all 4 sides
            constrain u on x=0 edge
            constrain v on y=0 edge
            rotations are NOT constrained

        Notes:
        - This assumes your global ordering is [w, u, v, thx, thy] with offsets
            self.off_w, self.off_u, self.off_v, self.off_thx, self.off_thy.
        - Uses each field’s own grid (nx_*, ny_*) and node map (_node).
        """
        bcs = []

        def on_bndry(i, j, nx, ny):
            return (i == 0) or (i == nx - 1) or (j == 0) or (j == ny - 1)

        # -------------------------
        # Fully clamped: all fields on all sides
        # -------------------------
        if self.clamped:
            # w
            for j in range(self.ny_w):
                for i in range(self.nx_w):
                    if on_bndry(i, j, self.nx_w, self.ny_w):
                        bcs.append(self.off_w + self._node(i, j, self.nx_w))

            # u
            for j in range(self.ny_thy):
                for i in range(self.nx_thy):
                    if on_bndry(i, j, self.nx_thy, self.ny_thy):
                        bcs.append(self.off_u + self._node(i, j, self.nx_thy))

            # v
            for j in range(self.ny_thx):
                for i in range(self.nx_thx):
                    if on_bndry(i, j, self.nx_thx, self.ny_thx):
                        bcs.append(self.off_v + self._node(i, j, self.nx_thx))

            # thx
            for j in range(self.ny_thx):
                for i in range(self.nx_thx):
                    if on_bndry(i, j, self.nx_thx, self.ny_thx):
                        bcs.append(self.off_thx + self._node(i, j, self.nx_thx))

            # thy
            for j in range(self.ny_thy):
                for i in range(self.nx_thy):
                    if on_bndry(i, j, self.nx_thy, self.ny_thy):
                        bcs.append(self.off_thy + self._node(i, j, self.nx_thy))

        # -------------------------
        # Simply supported (your original intent)
        # -------------------------
        else:

            # w boundary: all edges
            for j in range(self.ny_w):
                for i in range(self.nx_w):
                    if on_bndry(i, j, self.nx_w, self.ny_w):
                        bcs.append(self.off_w + self._node(i, j, self.nx_w))

            # u boundary: on x=0 edge
            for j in range(self.ny_thy):
                for i in range(self.nx_thy):
                    if i == 0:
                        bcs.append(self.off_u + self._node(i, j, self.nx_thy))

            # v boundary: on y=0 edge
            for j in range(self.ny_thx):
                for i in range(self.nx_thx):
                    if j == 0:
                        bcs.append(self.off_v + self._node(i, j, self.nx_thx))

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
        """
        row_grid: (nx_r, ny_r)
        col_grid: (nx_c, ny_c)
        stencil_ix: (imin, imax) inclusive offsets in i
        stencil_iy: (jmin, jmax) inclusive offsets in j
        periodic_y: wrap the hoop index j for columns (and effectively rows via looping all jr)
        """
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
    # Block patterns (generous stencils; periodic in hoop)
    # -------------------------------------------------------------------------
    # In _build_block_patterns(): set per=False (NO periodic hoop)
    def _build_block_patterns(self):
        per = False  # open hoop (no wrap)

        # --- w rows
        self.ww_rowp,   self.ww_cols,   self.ww_nnz   = self._build_pattern_from_stencil(self.nw,  (self.nx_w, self.ny_w), (self.nx_w, self.ny_w), (-2, 2), (-2, 2), periodic_y=per)
        self.wu_rowp,   self.wu_cols,   self.wu_nnz   = self._build_pattern_from_stencil(self.nw,  (self.nx_w, self.ny_w), (self.nx_thy, self.ny_thy), (-2, 1), (-2, 2), periodic_y=per)
        self.wv_rowp,   self.wv_cols,   self.wv_nnz   = self._build_pattern_from_stencil(self.nw,  (self.nx_w, self.ny_w), (self.nx_thx, self.ny_thx), (-2, 2), (-2, 1), periodic_y=per)
        self.wthx_rowp, self.wthx_cols, self.wthx_nnz = self._build_pattern_from_stencil(self.nw,  (self.nx_w, self.ny_w), (self.nx_thx, self.ny_thx), (-2, 2), (-2, 1), periodic_y=per)
        self.wthy_rowp, self.wthy_cols, self.wthy_nnz = self._build_pattern_from_stencil(self.nw,  (self.nx_w, self.ny_w), (self.nx_thy,  self.ny_thy ), (-2, 1), (-2, 2), periodic_y=per)

        # --- u rows
                # --- u rows  (u is on (nx_thy, ny_thy) now)
        self.uw_rowp,   self.uw_cols,   self.uw_nnz   = self._build_pattern_from_stencil(
            self.nu, (self.nx_thy, self.ny_thy), (self.nx_w,   self.ny_w),   (-1, 2), (-2, 2), periodic_y=per
        )
        self.uu_rowp,   self.uu_cols,   self.uu_nnz   = self._build_pattern_from_stencil(
            self.nu, (self.nx_thy, self.ny_thy), (self.nx_thy, self.ny_thy), (-1, 1), (-2, 2), periodic_y=per
        )
        self.uv_rowp,   self.uv_cols,   self.uv_nnz   = self._build_pattern_from_stencil(
            self.nu, (self.nx_thy, self.ny_thy), (self.nx_thx, self.ny_thx), (-1, 2), (-2, 1), periodic_y=per
        )
        self.uthx_rowp, self.uthx_cols, self.uthx_nnz = self._build_pattern_from_stencil(
            self.nu, (self.nx_thy, self.ny_thy), (self.nx_thx, self.ny_thx), (-1, 2), (-2, 1), periodic_y=per
        )
        self.uthy_rowp, self.uthy_cols, self.uthy_nnz = self._build_pattern_from_stencil(
            self.nu, (self.nx_thy, self.ny_thy), (self.nx_thy, self.ny_thy), (-1, 1), (-2, 2), periodic_y=per
        )

        # --- v rows
        self.vw_rowp,   self.vw_cols,   self.vw_nnz   = self._build_pattern_from_stencil(self.nv,  (self.nx_thx, self.ny_thx), (self.nx_w, self.ny_w), (-2, 2), (-1, 2), periodic_y=per)
        self.vu_rowp,   self.vu_cols,   self.vu_nnz   = self._build_pattern_from_stencil(
            self.nv, (self.nx_thx, self.ny_thx), (self.nx_thy, self.ny_thy), (-2, 1), (-1, 2), periodic_y=per
        )
        self.vv_rowp,   self.vv_cols,   self.vv_nnz   = self._build_pattern_from_stencil(self.nv,  (self.nx_thx, self.ny_thx), (self.nx_thx, self.ny_thx), (-2, 2), (-1, 1), periodic_y=per)
        self.vthx_rowp, self.vthx_cols, self.vthx_nnz = self._build_pattern_from_stencil(self.nv,  (self.nx_thx, self.ny_thx), (self.nx_thx, self.ny_thx), (-2, 2), (-1, 1), periodic_y=per)
        self.vthy_rowp, self.vthy_cols, self.vthy_nnz = self._build_pattern_from_stencil(self.nv,  (self.nx_thx, self.ny_thx), (self.nx_thy,  self.ny_thy ), (-2, 1), (-1, 2), periodic_y=per)

        # --- thx rows
        self.thxw_rowp,   self.thxw_cols,   self.thxw_nnz   = self._build_pattern_from_stencil(self.nthx, (self.nx_thx, self.ny_thx), (self.nx_w, self.ny_w), (-2, 2), (-1, 2), periodic_y=per)
        self.thxu_rowp,   self.thxu_cols,   self.thxu_nnz   = self._build_pattern_from_stencil(
            self.nthx, (self.nx_thx, self.ny_thx), (self.nx_thy, self.ny_thy), (-2, 1), (-1, 2), periodic_y=per
        )
        self.thxv_rowp,   self.thxv_cols,   self.thxv_nnz   = self._build_pattern_from_stencil(self.nthx, (self.nx_thx, self.ny_thx), (self.nx_thx, self.ny_thx), (-2, 2), (-1, 1), periodic_y=per)
        self.thxthx_rowp, self.thxthx_cols, self.thxthx_nnz = self._build_pattern_from_stencil(self.nthx, (self.nx_thx, self.ny_thx), (self.nx_thx, self.ny_thx), (-2, 2), (-1, 1), periodic_y=per)
        self.thxthy_rowp, self.thxthy_cols, self.thxthy_nnz = self._build_pattern_from_stencil(self.nthx, (self.nx_thx, self.ny_thx), (self.nx_thy,  self.ny_thy ), (-2, 1), (-1, 2), periodic_y=per)

        # --- thy rows
        self.thyw_rowp,   self.thyw_cols,   self.thyw_nnz   = self._build_pattern_from_stencil(self.nthy, (self.nx_thy,  self.ny_thy ), (self.nx_w, self.ny_w), (-1, 2), (-2, 2), periodic_y=per)
        self.thyu_rowp,   self.thyu_cols,   self.thyu_nnz   = self._build_pattern_from_stencil(
            self.nthy, (self.nx_thy, self.ny_thy), (self.nx_thy, self.ny_thy), (-1, 1), (-2, 2), periodic_y=per
        )
        self.thyv_rowp,   self.thyv_cols,   self.thyv_nnz   = self._build_pattern_from_stencil(self.nthy, (self.nx_thy,  self.ny_thy ), (self.nx_thx, self.ny_thx), (-1, 2), (-2, 1), periodic_y=per)
        self.thythx_rowp, self.thythx_cols, self.thythx_nnz = self._build_pattern_from_stencil(self.nthy, (self.nx_thy,  self.ny_thy ), (self.nx_thx, self.ny_thx), (-1, 2), (-2, 1), periodic_y=per)
        self.thythy_rowp, self.thythy_cols, self.thythy_nnz = self._build_pattern_from_stencil(self.nthy, (self.nx_thy,  self.ny_thy ), (self.nx_thy,  self.ny_thy ), (-1, 1), (-2, 2), periodic_y=per)
    # -------------------------------------------------------------------------
    # Allocate blocks + global bmat view
    # -------------------------------------------------------------------------
    def _alloc_blocks(self):
        z = np.zeros

        # w-row blocks
        self.k_ww   = sp.csr_matrix((z(self.ww_nnz),   self.ww_cols,   self.ww_rowp),   shape=(self.nw,   self.nw))
        self.k_wu   = sp.csr_matrix((z(self.wu_nnz),   self.wu_cols,   self.wu_rowp),   shape=(self.nw,   self.nu))
        self.k_wv   = sp.csr_matrix((z(self.wv_nnz),   self.wv_cols,   self.wv_rowp),   shape=(self.nw,   self.nv))
        self.k_wthx = sp.csr_matrix((z(self.wthx_nnz), self.wthx_cols, self.wthx_rowp), shape=(self.nw,   self.nthx))
        self.k_wthy = sp.csr_matrix((z(self.wthy_nnz), self.wthy_cols, self.wthy_rowp), shape=(self.nw,   self.nthy))

        # u-row blocks
        self.k_uw   = sp.csr_matrix((z(self.uw_nnz),   self.uw_cols,   self.uw_rowp),   shape=(self.nu,  self.nw))
        self.k_uu   = sp.csr_matrix((z(self.uu_nnz),   self.uu_cols,   self.uu_rowp),   shape=(self.nu,  self.nu))
        self.k_uv   = sp.csr_matrix((z(self.uv_nnz),   self.uv_cols,   self.uv_rowp),   shape=(self.nu,  self.nv))
        self.k_uthx = sp.csr_matrix((z(self.uthx_nnz), self.uthx_cols, self.uthx_rowp), shape=(self.nu,  self.nthx))
        self.k_uthy = sp.csr_matrix((z(self.uthy_nnz), self.uthy_cols, self.uthy_rowp), shape=(self.nu,  self.nthy))

        # v-row blocks
        self.k_vw   = sp.csr_matrix((z(self.vw_nnz),   self.vw_cols,   self.vw_rowp),   shape=(self.nv,   self.nw))
        self.k_vu   = sp.csr_matrix((z(self.vu_nnz),   self.vu_cols,   self.vu_rowp),   shape=(self.nv,   self.nu))
        self.k_vv   = sp.csr_matrix((z(self.vv_nnz),   self.vv_cols,   self.vv_rowp),   shape=(self.nv,   self.nv))
        self.k_vthx = sp.csr_matrix((z(self.vthx_nnz), self.vthx_cols, self.vthx_rowp), shape=(self.nv,   self.nthx))
        self.k_vthy = sp.csr_matrix((z(self.vthy_nnz), self.vthy_cols, self.vthy_rowp), shape=(self.nv,   self.nthy))

        # thx-row blocks
        self.k_thxw   = sp.csr_matrix((z(self.thxw_nnz),   self.thxw_cols,   self.thxw_rowp),   shape=(self.nthx, self.nw))
        self.k_thxu   = sp.csr_matrix((z(self.thxu_nnz),   self.thxu_cols,   self.thxu_rowp),   shape=(self.nthx, self.nu))
        self.k_thxv   = sp.csr_matrix((z(self.thxv_nnz),   self.thxv_cols,   self.thxv_rowp),   shape=(self.nthx, self.nv))
        self.k_thxthx = sp.csr_matrix((z(self.thxthx_nnz), self.thxthx_cols, self.thxthx_rowp), shape=(self.nthx, self.nthx))
        self.k_thxthy = sp.csr_matrix((z(self.thxthy_nnz), self.thxthy_cols, self.thxthy_rowp), shape=(self.nthx, self.nthy))

        # thy-row blocks
        self.k_thyw   = sp.csr_matrix((z(self.thyw_nnz),   self.thyw_cols,   self.thyw_rowp),   shape=(self.nthy, self.nw))
        self.k_thyu   = sp.csr_matrix((z(self.thyu_nnz),   self.thyu_cols,   self.thyu_rowp),   shape=(self.nthy, self.nu))
        self.k_thyv   = sp.csr_matrix((z(self.thyv_nnz),   self.thyv_cols,   self.thyv_rowp),   shape=(self.nthy, self.nv))
        self.k_thythx = sp.csr_matrix((z(self.thythx_nnz), self.thythx_cols, self.thythx_rowp), shape=(self.nthy, self.nthx))
        self.k_thythy = sp.csr_matrix((z(self.thythy_nnz), self.thythy_cols, self.thythy_rowp), shape=(self.nthy, self.nthy))

        # RHS blocks
        self.fw   = np.zeros(self.nw)
        self.fu   = np.zeros(self.nu)
        self.fv   = np.zeros(self.nv)
        self.fthx = np.zeros(self.nthx)
        self.fthy = np.zeros(self.nthy)

        # global view
        self.kmat = sp.bmat(
            [
                [self.k_ww,   self.k_wu,   self.k_wv,   self.k_wthx,   self.k_wthy],
                [self.k_uw,   self.k_uu,   self.k_uv,   self.k_uthx,   self.k_uthy],
                [self.k_vw,   self.k_vu,   self.k_vv,   self.k_vthx,   self.k_vthy],
                [self.k_thxw, self.k_thxu, self.k_thxv, self.k_thxthx, self.k_thxthy],
                [self.k_thyw, self.k_thyu, self.k_thyv, self.k_thythx, self.k_thythy],
            ],
            format="csr",
        )

    # -------------------------------------------------------------------------
    # Assembly
    # -------------------------------------------------------------------------
    def _assemble_system(self):
        # zero stiffness
        for A in [
            self.k_ww, self.k_wu, self.k_wv, self.k_wthx, self.k_wthy,
            self.k_uw, self.k_uu, self.k_uv, self.k_uthx, self.k_uthy,
            self.k_vw, self.k_vu, self.k_vv, self.k_vthx, self.k_vthy,
            self.k_thxw, self.k_thxu, self.k_thxv, self.k_thxthx, self.k_thxthy,
            self.k_thyw, self.k_thyu, self.k_thyv, self.k_thythx, self.k_thythy,
        ]:
            A.data[:] = 0.0

        # zero rhs
        self.fw[:] = 0.0
        self.fu[:] = 0.0
        self.fv[:] = 0.0
        self.fthx[:] = 0.0
        self.fthy[:] = 0.0

        elem_id = 0
        for ey in range(self.nye):
            for ex in range(self.nxe):
                w_dofs   = self.conn_w[elem_id]    # 9
                u_dofs   = self.conn_u[elem_id]    # 6
                v_dofs   = self.conn_v[elem_id]    # 6
                thx_dofs = self.conn_thx[elem_id]  # 6
                thy_dofs = self.conn_thy[elem_id]  # 6

                left_b  = (ex == 0)
                right_b = (ex == self.nxe - 1)

                # # Hoop is periodic: never treat as boundary
                # bot_b = False
                # top_b = False
                bot_b = ey == 0
                top_b = ey == self.nye - 1

                # print(f"({ex=},{ey=}) : {left_b=} {right_b=} {bot_b=} {top_b=}")

                # element stiffness (w,u,v,thx,thy)
                (
                    Kww,  Kwu,  Kwv,  Kwtx,  Kwty,
                    Kuw,  Kuu,  Kuv,  Kutx,  Kuty,
                    Kvw,  Kvu,  Kvv,  Kvtx,  Kvty,
                    Ktxw, Ktxu, Ktxv, Ktxtx, Ktxty,
                    Ktyw, Ktyu, Ktyv, Ktytx, Ktyty
                ) = self.element.get_kelem(
                    E=self.E, nu=self.nu_poiss, thick=self.thick,
                    dx=self.dx, dy=self.dy,
                    left_bndry=left_b, right_bndry=right_b,
                    bot_bndry=bot_b, top_bndry=top_b
                )

                # RHS (optional)
                fw = np.zeros(9)
                fu = np.zeros(6)
                fv = np.zeros(6)
                fthx = np.zeros(6)
                fthy = np.zeros(6)
                if hasattr(self.element, "get_felem"):
                    x0 = ex * self.dx
                    s0 = ey * self.dy
                    out = self.element.get_felem(
                        self.load_fcn, x0, s0, self.dx, self.dy,
                        left_b, right_b, bot_b, top_b
                    )
                    # accept either 5 or 1-vector return conventions
                    if isinstance(out, tuple) and len(out) == 5:
                        fw, fu, fv, fthx, fthy = out
                    elif isinstance(out, tuple) and len(out) == 1:
                        fw = out[0]
                    else:
                        fw = out  # best effort
                # else:
                #     if ex == 0 and ey == 0: print("WARNING: felem not implemented, so doing flat loading\n")
                #     fw = self.load_fcn(0.5, 0.5) * np.ones(9)

                # -----------------------------
                # Scatter-add (blocks)
                # -----------------------------
                # w row
                for a, I in enumerate(w_dofs):
                    self.fw[I] += fw[a]
                    for b, J in enumerate(w_dofs):
                        self._add_to_block(self.k_ww, self.ww_rowp, self.k_ww.indices, I, J, Kww[a, b])
                    for b, J in enumerate(u_dofs):
                        self._add_to_block(self.k_wu, self.wu_rowp, self.k_wu.indices, I, J, Kwu[a, b])
                    for b, J in enumerate(v_dofs):
                        self._add_to_block(self.k_wv, self.wv_rowp, self.k_wv.indices, I, J, Kwv[a, b])
                    for b, J in enumerate(thx_dofs):
                        self._add_to_block(self.k_wthx, self.wthx_rowp, self.k_wthx.indices, I, J, Kwtx[a, b])
                    for b, J in enumerate(thy_dofs):
                        self._add_to_block(self.k_wthy, self.wthy_rowp, self.k_wthy.indices, I, J, Kwty[a, b])

                # u row
                for a, I in enumerate(u_dofs):
                    self.fu[I] += fu[a]
                    for b, J in enumerate(w_dofs):
                        self._add_to_block(self.k_uw, self.uw_rowp, self.k_uw.indices, I, J, Kuw[a, b])
                    for b, J in enumerate(u_dofs):
                        self._add_to_block(self.k_uu, self.uu_rowp, self.k_uu.indices, I, J, Kuu[a, b])
                    for b, J in enumerate(v_dofs):
                        self._add_to_block(self.k_uv, self.uv_rowp, self.k_uv.indices, I, J, Kuv[a, b])
                    for b, J in enumerate(thx_dofs):
                        self._add_to_block(self.k_uthx, self.uthx_rowp, self.k_uthx.indices, I, J, Kutx[a, b])
                    for b, J in enumerate(thy_dofs):
                        self._add_to_block(self.k_uthy, self.uthy_rowp, self.k_uthy.indices, I, J, Kuty[a, b])

                # v row
                for a, I in enumerate(v_dofs):
                    self.fv[I] += fv[a]
                    for b, J in enumerate(w_dofs):
                        self._add_to_block(self.k_vw, self.vw_rowp, self.k_vw.indices, I, J, Kvw[a, b])
                    for b, J in enumerate(u_dofs):
                        self._add_to_block(self.k_vu, self.vu_rowp, self.k_vu.indices, I, J, Kvu[a, b])
                    for b, J in enumerate(v_dofs):
                        self._add_to_block(self.k_vv, self.vv_rowp, self.k_vv.indices, I, J, Kvv[a, b])
                    for b, J in enumerate(thx_dofs):
                        self._add_to_block(self.k_vthx, self.vthx_rowp, self.k_vthx.indices, I, J, Kvtx[a, b])
                    for b, J in enumerate(thy_dofs):
                        self._add_to_block(self.k_vthy, self.vthy_rowp, self.k_vthy.indices, I, J, Kvty[a, b])

                # thx row
                for a, I in enumerate(thx_dofs):
                    self.fthx[I] += fthx[a]
                    for b, J in enumerate(w_dofs):
                        self._add_to_block(self.k_thxw, self.thxw_rowp, self.k_thxw.indices, I, J, Ktxw[a, b])
                    for b, J in enumerate(u_dofs):
                        self._add_to_block(self.k_thxu, self.thxu_rowp, self.k_thxu.indices, I, J, Ktxu[a, b])
                    for b, J in enumerate(v_dofs):
                        self._add_to_block(self.k_thxv, self.thxv_rowp, self.k_thxv.indices, I, J, Ktxv[a, b])
                    for b, J in enumerate(thx_dofs):
                        self._add_to_block(self.k_thxthx, self.thxthx_rowp, self.k_thxthx.indices, I, J, Ktxtx[a, b])
                    for b, J in enumerate(thy_dofs):
                        self._add_to_block(self.k_thxthy, self.thxthy_rowp, self.k_thxthy.indices, I, J, Ktxty[a, b])

                # thy row
                for a, I in enumerate(thy_dofs):
                    self.fthy[I] += fthy[a]
                    for b, J in enumerate(w_dofs):
                        self._add_to_block(self.k_thyw, self.thyw_rowp, self.k_thyw.indices, I, J, Ktyw[a, b])
                    for b, J in enumerate(u_dofs):
                        self._add_to_block(self.k_thyu, self.thyu_rowp, self.k_thyu.indices, I, J, Ktyu[a, b])
                    for b, J in enumerate(v_dofs):
                        self._add_to_block(self.k_thyv, self.thyv_rowp, self.k_thyv.indices, I, J, Ktyv[a, b])
                    for b, J in enumerate(thx_dofs):
                        self._add_to_block(self.k_thythx, self.thythx_rowp, self.k_thythx.indices, I, J, Ktytx[a, b])
                    for b, J in enumerate(thy_dofs):
                        self._add_to_block(self.k_thythy, self.thythy_rowp, self.k_thythy.indices, I, J, Ktyty[a, b])

                elem_id += 1

        # rebuild global view
        self.kmat = sp.bmat(
            [
                [self.k_ww,   self.k_wu,   self.k_wv,   self.k_wthx,   self.k_wthy],
                [self.k_uw,   self.k_uu,   self.k_uv,   self.k_uthx,   self.k_uthy],
                [self.k_vw,   self.k_vu,   self.k_vv,   self.k_vthx,   self.k_vthy],
                [self.k_thxw, self.k_thxu, self.k_thxv, self.k_thxthx, self.k_thxthy],
                [self.k_thyw, self.k_thyu, self.k_thyv, self.k_thythx, self.k_thythy],
            ],
            format="csr",
        )

        self.force[self.off_w:self.off_u] = self.fw
        self.force[self.off_u:self.off_v] = self.fu
        self.force[self.off_v:self.off_thx] = self.fv
        self.force[self.off_thx:self.off_thy] = self.fthx
        self.force[self.off_thy:] = self.fthy

        self._apply_bcs()

        # plt.imshow(self.kmat.toarray())
        # plt.show()


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

        # # add one to the whole diagonal matrix so invertible..
        # eps = 1e-12   # or 1e-10 / 1e-8 depending how singular it is
        # self.kmat = self.kmat + eps * sp.eye(self.kmat.shape[0], format="csr")

        # print(f"{f=}")
        # print(f"{self.force[:self.off_u]=}")

    # -------------------------------------------------------------------------
    # Solve hook
    # -------------------------------------------------------------------------
    def direct_solve(self):
        self._assemble_system()
        self.u = sp.linalg.spsolve(self.kmat.copy(), self.force)

        # check residual
        res = self.force - self.kmat.dot(self.u)
        force_nrm = np.linalg.norm(self.force)
        res_nrm = np.linalg.norm(res)
        rel_nrm = res_nrm / force_nrm
        # print(f"{rel_nrm=:.4e} of direct solve")

        return self.u
    
    def prolongate(self, coarse_soln: np.ndarray):
        nxe_c = self.nxe // 2
        nye_c = self.nye // 2
        return self.element.prolongate(coarse_soln, nxe_c, nye_c)

    def restrict_defect(self, fine_defect: np.ndarray):
        # called on coarse grid object; pass its (nxe,nye) as coarse sizes
        return self.element.restrict_defect(fine_defect, self.nxe, self.nye)

    # Also: plot reshape is flipped; fix it:
    # def plot_disp(self, disp_mag: float = 0.2, mode: str = "w", deform: str = "none"):
    #     """
    #     mode:   which field to visualize in color and (optionally) geometry
    #             one of ["w", "u", "v", "thx", "thy"]
    #     deform: how to deform the cylinder geometry
    #             - "w":      always deform with w (safe default)
    #             - "radial": deform with the selected `mode` field
    #             - "none":   no deformation, just plot on undeformed cylinder
    #     """
    #     if self.u is None:
    #         raise RuntimeError("Run direct_solve() first.")

    #     # ---- helper to slice the chosen field ----
    #     mode = mode.lower()
    #     deform = deform.lower()

    #     # global ordering: [w, u, v, thx, thy]
    #     off_w   = 0
    #     off_u   = off_w   + self.nw
    #     off_v   = off_u   + self.nu
    #     off_thx = off_v   + self.nv
    #     off_thy = off_thx + self.nthx

    #     if mode == "w":
    #         vec = self.u[off_w:off_w + self.nw]
    #         nx, ny = self.nx_w, self.ny_w
    #         label = "w"
    #     elif mode == "u":
    #         vec = self.u[off_u:off_u + self.nu]
    #         nx, ny = self.nx_thy, self.ny_thy
    #         label = "u"
    #     elif mode == "v":
    #         vec = self.u[off_v:off_v + self.nv]
    #         nx, ny = self.nx_thx, self.ny_thx
    #         label = "v"
    #     elif mode == "thx":
    #         vec = self.u[off_thx:off_thx + self.nthx]
    #         nx, ny = self.nx_thx, self.ny_thx
    #         label = "thx"
    #     elif mode == "thy":
    #         vec = self.u[off_thy:off_thy + self.nthy]
    #         nx, ny = self.nx_thy, self.ny_thy
    #         label = "thy"
    #     else:
    #         raise ValueError(f"Unknown mode='{mode}'. Use one of ['w','u','v','thx','thy'].")

    #     # vec_nrm = np.linalg.norm(vec)
    #     vec_nrm = np.max(np.abs(vec))
    #     print(f"disp nrm: {vec_nrm:.4e}")

    #     V = vec.reshape((ny, nx))
    #     # if mode == 'u':
    #     #     V = V.T
    #     # V = vec.reshape((ny, nx)).T
    #     # print(f"{V=}")
    #     # print(f"{vec=}")
    #     # print(f"{V=}")

    #     # ---- build (x, theta) grid matching that field ----
    #     x = np.linspace(0.0, self.Lx, nx)
    #     th = np.linspace(-self.Ly, 0.0, ny)
    #     X, TH = np.meshgrid(x, th, indexing="xy")
    #     Phi = TH / self.radius

    #     # flip X coords for some reason (plotting issue)
    #     X = 1.0 - X

    #     # print(f"{X=}")

    #     # ---- choose deformation field ----
    #     # if deform == "none":
    #     #     Rdef = np.zeros_like(V)
    #     #     deform_label = "none"
    #     # else:
    #     #     if deform == "radial":
    #     #         # deform using the selected field
    #     #         rad_vec = vec
    #     #         rad_nx, rad_ny = nx, ny
    #     #         deform_label = label
    #     #     elif deform == "w":
    #     #         # always deform using w field (recommended when mode != w)
    #     #         rad_vec = self.u[off_w:off_w + self.nw]
    #     #         rad_nx, rad_ny = self.nx_w, self.ny_w
    #     #         deform_label = "w"
    #     #     else:
    #     #         raise ValueError("deform must be one of ['w','radial','none'].")

    #     #     # If deform grid differs from plot grid, you probably want to only use deform="w"
    #     #     # unless you *know* they share the same nx/ny.
    #     #     if (rad_nx, rad_ny) != (nx, ny):
    #     #         raise ValueError(
    #     #             f"Deformation grid ({rad_ny}x{rad_nx}) != plot grid ({ny}x{nx}). "
    #     #             f"Use deform='w' or deform='none' for mode='{mode}'."
    #     #         )

    #     R = vec.reshape((ny, nx))
    #     orig_mag = float(np.max(np.abs((R))))
    #     scale_factor = (disp_mag / orig_mag) if orig_mag > 0 else 1.0
    #     Rdef = R * scale_factor

    #     # Rdef *= -1

    #     # ---- geometry ----
    #     Y = (self.radius + Rdef) * np.sin(Phi)
    #     Z = (self.radius + Rdef) * np.cos(Phi)

    #     # ---- color by selected field ----
    #     import matplotlib.pyplot as plt
    #     import matplotlib.colors as mcolors
    #     import matplotlib.cm as cm

    #     C = np.abs(V)  # vertex values (ny, nx)
    #     C_face = 0.25 * (C[:-1, :-1] + C[1:, :-1] + C[:-1, 1:] + C[1:, 1:])

    #     norm = mcolors.Normalize(vmin=float(C_face.min()), vmax=float(C_face.max()))
    #     cmap = cm.get_cmap("viridis")
    #     facecolors = cmap(norm(C_face))  # (ny-1, nx-1, 4)

    #     fig = plt.figure(figsize=(9, 6))
    #     ax = fig.add_subplot(111, projection="3d")

    #     ax.plot_surface(
    #         X, Y, Z,
    #         facecolors=facecolors,
    #         linewidth=0,
    #         antialiased=True,
    #         shade=False,
    #     )

    #     ax.set_xlabel("x")
    #     ax.set_ylabel("y")
    #     ax.set_zlabel("radial")
    #     ax.set_title(f"color={label}")

    #     ax.view_init(elev=25, azim=-135)

    #     mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    #     mappable.set_array([])
    #     fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.08, label=f"|{label}|")

    #     plt.tight_layout()
    #     plt.show()

    def _plot_disp(self, disp_mag: float = 0.2, mode: str = "w", ax=None):
        """
        PRIVATE: plot a single field onto `ax` (3D) if provided, else create a figure.

        Fixed deformation rule (NO INPUT):
        - mode == "w": deform with w
        - else:        undeformed (Rdef = 0)  [grids differ, so no u/u etc]
        """
        if self.u is None:
            raise RuntimeError("Run direct_solve() first.")

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import matplotlib.cm as cm

        mode = mode.lower()

        # global ordering: [w, u, v, thx, thy]
        off_w   = 0
        off_u   = off_w   + self.nw
        off_v   = off_u   + self.nu
        off_thx = off_v   + self.nv
        off_thy = off_thx + self.nthx

        if mode == "w":
            vec = self.u[off_w:off_w + self.nw]
            nx, ny = self.nx_w, self.ny_w
            label = "w"
        elif mode == "u":
            vec = self.u[off_u:off_u + self.nu]
            nx, ny = self.nx_thy, self.ny_thy
            label = "u"
        elif mode == "v":
            vec = self.u[off_v:off_v + self.nv]
            nx, ny = self.nx_thx, self.ny_thx
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
            raise ValueError(f"Unknown mode='{mode}'. Use one of ['w','u','v','thx','thy'].")

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

        # ---- deformation (FIXED, no user input) ----
        R = V
        orig_mag = float(np.max(np.abs(R)))
        scale_factor = (disp_mag / orig_mag) if orig_mag > 0 else 1.0
        Rdef = R * scale_factor

        # ---- geometry ----
        Y = (self.radius + Rdef) * np.sin(Phi)
        Z = (self.radius + Rdef) * np.cos(Phi)

        # ---- color by selected field ----
        C = np.abs(V)  # vertex values (ny, nx)
        C_face = 0.25 * (C[:-1, :-1] + C[1:, :-1] + C[:-1, 1:] + C[1:, 1:])

        norm = mcolors.Normalize(vmin=float(C_face.min()), vmax=float(C_face.max()))
        cmap = cm.get_cmap("viridis")
        facecolors = cmap(norm(C_face))  # (ny-1, nx-1, 4)

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

        # only colorbar on standalone figure (otherwise subplots become a mess)
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

        - mode in {'w','u','v','thx','thy'}: plot one
        - mode == 'all': plot all 5 in a (3,2) grid (last slot empty)
        """
        import matplotlib.pyplot as plt

        if isinstance(mode, str) and mode.lower() == "all":
            modes = ["w", "u", "v", "thx", "thy"]

            fig = plt.figure(figsize=(14, 10))
            axs = [fig.add_subplot(2, 3, k + 1, projection="3d") for k in range(6)]

            for i, m in enumerate(modes):
                self._plot_disp(disp_mag=disp_mag, mode=m, ax=axs[i])

            axs[5].set_axis_off()
            plt.tight_layout()
            plt.show()
            return

        self._plot_disp(disp_mag=disp_mag, mode=mode, ax=None)
