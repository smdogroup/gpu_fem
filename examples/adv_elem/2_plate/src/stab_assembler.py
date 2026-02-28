import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
from _sparse_utils import build_csr_from_conn
from std_assembler import StandardPlateAssembler

# ============================================================
# Stabilized (CIP-edge) Structured Assembler for Q1 RM plates
# ============================================================

class StabilizedPlateAssembler(StandardPlateAssembler):
    """
    Same as StandardPlateAssembler but additionally assembles
    *interior-edge* CIP stabilization terms that depend on jumps
    of (n·∇theta), div(theta), and (n·∇w).

    This assumes:
      - structured axis-aligned mesh on rectangle
      - Q1 4-node quads
      - C0 dofs shared across elements (standard conforming mesh)
      - element provides: get_edge_stab_kelem(...)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_stab = bool(self.element.edge_stab)

        # quick local-edge -> local-node-pair map for Q4
        # local nodes: 0:(ex,ey), 1:(ex+1,ey), 2:(ex+1,ey+1), 3:(ex,ey+1)
        self._edge_lnodes = {
            0: (0, 1),  # bottom  eta=-1
            1: (1, 2),  # right   xi=+1
            2: (2, 3),  # top     eta=+1
            3: (3, 0),  # left    xi=-1
        }

    def _add_block(self, br: int, bc: int, Kblk: np.ndarray):
        """Add a dpn x dpn block into the BSR data for node-row br, node-col bc."""
        dpn = self.dof_per_node
        assert Kblk.shape == (dpn, dpn)
        for colp in range(self.rowp[br], self.rowp[br + 1]):
            if self.cols[colp] == bc:
                self.data[colp, :, :] += Kblk
                return
        raise RuntimeError("BCSR pattern missing a required (br,bc) entry. Pattern build is inconsistent.")

    def _assemble_edge_stabilization(self):
        """
        Assemble CIP edge stabilization over all interior edges.
        For structured mesh this is just:
          - vertical edges between (ex,ey) and (ex+1,ey)
          - horizontal edges between (ex,ey) and (ex,ey+1)
        """
        if not self.edge_stab:
            return

        if not hasattr(self.element, "get_edge_stab_kelem"):
            raise AttributeError(
                "Element must implement get_edge_stab_kelem(E,nu,thick, elem_xpts_L, elem_xpts_R, edge_L, edge_R, nx, ny)"
            )

        dpn = self.dof_per_node

        # --- vertical interior edges: between left elem and right elem
        # left elem uses local edge 1 (right), right elem uses local edge 3 (left)
        # normal choose +x (from left -> right)
        nx, ny = 1.0, 0.0
        for ey in range(self.nxe):
            for ex in range(self.nxe - 1):
                eL = ex + self.nxe * ey
                eR = (ex + 1) + self.nxe * ey

                locL = self.conn[eL]
                locR = self.conn[eR]
                xptsL = self._elem_xpts_from_loc(locL)
                xptsR = self._elem_xpts_from_loc(locR)

                edge_L = 1
                edge_R = 3

                # returns a (6x6) matrix on the two shared edge nodes, ordered by L's local edge node order
                Kedge6, gnodes2 = self.element.get_edge_stab_kelem(
                    self.E, self.nu, self.thick, xptsL, xptsR, edge_L, edge_R, nx, ny,
                    # optional: pass global node ids so element can ensure consistent ordering
                    loc_conn_L=locL, loc_conn_R=locR
                )

                # scatter 6x6 into BSR blocks (2 nodes)
                g0, g1 = gnodes2
                for (ir, gr) in enumerate((g0, g1)):
                    for (ic, gc) in enumerate((g0, g1)):
                        blk = Kedge6[dpn*ir:dpn*(ir+1), dpn*ic:dpn*(ic+1)]
                        self._add_block(gr, gc, blk)

        # --- horizontal interior edges: between bottom elem and top elem
        # bottom elem uses local edge 2 (top), top elem uses local edge 0 (bottom)
        # normal choose +y (from bottom -> top)
        nx, ny = 0.0, 1.0
        for ey in range(self.nxe - 1):
            for ex in range(self.nxe):
                eB = ex + self.nxe * ey
                eT = ex + self.nxe * (ey + 1)

                locB = self.conn[eB]
                locT = self.conn[eT]
                xptsB = self._elem_xpts_from_loc(locB)
                xptsT = self._elem_xpts_from_loc(locT)

                edge_B = 2
                edge_T = 0

                Kedge6, gnodes2 = self.element.get_edge_stab_kelem(
                    self.E, self.nu, self.thick, xptsB, xptsT, edge_B, edge_T, nx, ny,
                    loc_conn_L=locB, loc_conn_R=locT
                )

                g0, g1 = gnodes2
                for (ir, gr) in enumerate((g0, g1)):
                    for (ic, gc) in enumerate((g0, g1)):
                        blk = Kedge6[dpn*ir:dpn*(ir+1), dpn*ic:dpn*(ic+1)]
                        self._add_block(gr, gc, blk)

    def _assemble_system(self, bcs: bool = True):
        # volume terms (same as StandardPlateAssembler)
        dpn = self.dof_per_node
        self.data = np.zeros((self.nnzb, dpn, dpn), dtype=np.double)
        self.force = np.zeros(self.N, dtype=np.double)

        for ielem in range(self.num_elements):
            loc_conn = self.conn[ielem]
            loc_dofs = self.dof_conn[ielem]
            elem_xpts = self._elem_xpts_from_loc(loc_conn)

            kelem = self.element.get_kelem(self.E, self.nu, self.thick, elem_xpts)
            felem = self.element.get_felem(mag=self.load_fcn, elem_xpts=elem_xpts)

            # LHS scatter into block-CSR
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

            # RHS scatter
            np.add.at(self.force, loc_dofs, felem)

        # NEW: assemble interior-edge stabilization (CIP)
        self._assemble_edge_stabilization()

        if bcs:
            self._apply_bcs(dpn)

        self.kmat = sp.bsr_matrix((self.data, self.cols, self.rowp), shape=(self.N, self.N))

