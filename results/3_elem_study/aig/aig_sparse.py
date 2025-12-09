import numpy as np
from dataclasses import dataclass
from math import comb
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# --------------------------
# Utility: make open knot
# --------------------------
def make_open_knot_vector(n_ctrl, degree):
    m = n_ctrl + degree + 1
    n_elems = n_ctrl - degree
    kv = [0.0]*(degree+1)
    if n_elems > 0:
        for i in range(1, n_elems):
            kv.append(i / n_elems)
    kv += [1.0]*(degree+1)
    return np.array(kv)

# --------------------------
# Cox-de Boor basis + first derivative
# (expects span to be supplied)
# --------------------------
def basis_functions_and_derivatives(span, u, p, U):
    left = np.zeros(p+1)
    right = np.zeros(p+1)
    ndu = np.zeros((p+1, p+1))
    ndu[0,0] = 1.0
    for j in range(1, p+1):
        left[j] = u - U[span+1-j]
        right[j] = U[span+j] - u
        saved = 0.0
        for r in range(j):
            ndu[j,r] = right[r+1] + left[j-r]
            temp = ndu[r,j-1] / ndu[j,r]
            ndu[r,j] = saved + right[r+1]*temp
            saved = left[j-r]*temp
        ndu[j,j] = saved
    N = ndu[:, p].copy()
    ders1 = np.zeros(p+1)
    a = np.zeros((2, p+1))
    for r in range(p+1):
        a[0,0] = 1.0
        s1 = 0; s2 = 1
        for k in (1,):
            d = 0.0
            rk = r - k
            pk = p - k
            if rk >= 0:
                a[s2,0] = a[s1,0] / ndu[pk+1, rk]
                d = a[s2,0] * ndu[rk, pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k-1 if r-1 <= pk else p - r
            for j in range(j1, j2+1):
                a[s2,j] = (a[s1,j] - a[s1,j-1]) / ndu[pk+1, rk+j]
                d += a[s2,j] * ndu[rk+j, pk]
            if r <= pk:
                a[s2,k] = -a[s1,k-1] / ndu[pk+1, r]
                d += a[s2,k] * ndu[r, pk]
            ders1[r] = d
            s1, s2 = s2, s1
    ders1 *= p
    return N, ders1

# --------------------------
# Patch dataclass
# --------------------------
@dataclass
class Patch:
    Lx: float = 1.0
    Ly: float = 1.0
    nxe: int = 4
    nye: int = 4
    p: int = 2
    q: int = 2
    thickness: float = 0.01
    E: float = 70e9
    nu: float = 0.33
    shear_correction: float = 5/6
    scale_coords_by_thickness: bool = True
    scale_bending_by_1_over_th2: bool = True

    def __post_init__(self):
        self.n_ctrl_x = self.nxe + self.p
        self.n_ctrl_y = self.nye + self.q
        self.U = make_open_knot_vector(self.n_ctrl_x, self.p)
        self.V = make_open_knot_vector(self.n_ctrl_y, self.q)
        x_coords = np.linspace(0, self.Lx, self.n_ctrl_x)
        y_coords = np.linspace(0, self.Ly, self.n_ctrl_y)
        scale = self.thickness if self.scale_coords_by_thickness else 1.0
        self.ctrl_pts = np.zeros((self.n_ctrl_x, self.n_ctrl_y, 2))
        for i,x in enumerate(x_coords):
            for j,y in enumerate(y_coords):
                self.ctrl_pts[i,j,0] = x*scale
                self.ctrl_pts[i,j,1] = y*scale
        self.weights = np.ones((self.n_ctrl_x, self.n_ctrl_y))

# --------------------------
# BSR assembly (6 DOF per node: u,v,w,thx,thy,thz)
# --------------------------
def assemble_FSDT_patch_bsr_6dof(patch: Patch, q_load=1.0, quad_order=None):
    p = patch.p; q = patch.q
    nctrl = patch.n_ctrl_x * patch.n_ctrl_y
    dof_per_node = 6
    ndof = dof_per_node * nctrl
    if quad_order is None:
        quad_order = max(p+1, q+1)

    # material matrices (same as before)
    t = patch.thickness
    E = patch.E; nu = patch.nu
    G = E/(2*(1+nu))
    D_b = E * t**3 / (12*(1-nu**2)) * np.array([[1, nu, 0],[nu,1,0],[0,0,(1-nu)/2]])
    if patch.scale_bending_by_1_over_th2:
        D_b *= 1.0/(t**2)
    D_s = patch.shear_correction * G * t * np.eye(2)

    # element spans
    U = patch.U; V = patch.V
    spans_x = [(a, U[a], U[a+1]) for a in range(len(U)-1) if U[a+1] - U[a] > 1e-12]
    spans_y = [(b, V[b], V[b+1]) for b in range(len(V)-1) if V[b+1] - V[b] > 1e-12]

    from numpy.polynomial.legendre import leggauss
    xg, wg = leggauss(quad_order)
    xi_pts = 0.5*(xg + 1.0); xi_w = 0.5*wg
    eta_pts = xi_pts; eta_w = xi_w

    def ctrl_gid(i,j):
        return i*patch.n_ctrl_y + j

    # First pass: sparsity from element connectivities (per control-point blocks)
    row_to_cols = [set() for _ in range(nctrl)]
    element_connectivities = []
    element_param_ranges = []
    for sx, u0, u1 in spans_x:
        for sy, v0, v1 in spans_y:
            i_start = sx - p; i_end = sx
            j_start = sy - q; j_end = sy
            elem_ctrls = []
            for ii in range(i_start, i_end+1):
                for jj in range(j_start, j_end+1):
                    g = ctrl_gid(ii,jj)
                    elem_ctrls.append(g)
            element_connectivities.append(elem_ctrls)
            element_param_ranges.append((sx, u0, u1, sy, v0, v1))
            for Ia in elem_ctrls:
                for Jb in elem_ctrls:
                    row_to_cols[Ia].add(Jb)

    indptr = np.zeros(nctrl+1, dtype=np.int32)
    indices_list = []
    for i in range(nctrl):
        cols = sorted(row_to_cols[i])
        indices_list.append(np.array(cols, dtype=np.int32))
        indptr[i+1] = indptr[i] + len(cols)
    total_blocks = int(indptr[-1])

    # map (row,col) -> linear block index
    block_pos = {}
    pos = 0
    for i in range(nctrl):
        cols = indices_list[i]
        for col in cols:
            block_pos[(i,int(col))] = pos
            pos += 1

    block_size = dof_per_node
    data = np.zeros((total_blocks, block_size, block_size), dtype=float)
    F = np.zeros(ndof)

    # mapping from small 3-DOF ordering [w,thx,thy] -> indices inside 6-DOF node
    # small index 0-> w, 1->thx, 2->thy
    def local3_to_local6_index(node_index, small_idx):
        base = node_index * dof_per_node
        if small_idx == 0:
            return base + 2  # w
        elif small_idx == 1:
            return base + 3  # thx
        elif small_idx == 2:
            return base + 4  # thy
        else:
            raise IndexError

    # Second pass: assemble element contributions (compute 3*nn Ke then embed into 6*nn)
    for elem_idx, elem_ctrls in enumerate(element_connectivities):
        sx, u0, u1, sy, v0, v1 = element_param_ranges[elem_idx]
        nn = len(elem_ctrls)
        Ke3 = np.zeros((3*nn, 3*nn))  # using [w,thx,thy] local ordering
        Fe3 = np.zeros(3*nn)

        for i_g, xi in enumerate(xi_pts):
            u = u0 + xi*(u1-u0)
            wu = xi_w[i_g]*(u1-u0)
            Nu, dNu = basis_functions_and_derivatives(sx, u, p, U)
            for j_g, eta in enumerate(eta_pts):
                v = v0 + eta*(v1-v0)
                wv = eta_w[j_g]*(v1-v0)
                Nv, dNv = basis_functions_and_derivatives(sy, v, q, V)

                # tensor-product basis (local)
                R = np.zeros(nn); dR_du = np.zeros(nn); dR_dv = np.zeros(nn)
                k = 0
                for a in range(p+1):
                    for b in range(q+1):
                        R[k] = Nu[a] * Nv[b]
                        dR_du[k] = dNu[a] * Nv[b]
                        dR_dv[k] = Nu[a] * dNv[b]
                        k += 1

                # map to physical
                x = np.zeros(2); dx_du = np.zeros(2); dx_dv = np.zeros(2)
                for k, g in enumerate(elem_ctrls):
                    ii = g // patch.n_ctrl_y; jj = g % patch.n_ctrl_y
                    cp = patch.ctrl_pts[ii, jj]
                    x += R[k] * cp
                    dx_du += dR_du[k] * cp
                    dx_dv += dR_dv[k] * cp
                J = np.column_stack([dx_du, dx_dv])
                detJ = np.linalg.det(J)
                if detJ <= 0:
                    raise RuntimeError("Non-positive Jacobian during assembly")
                invJ = np.linalg.inv(J)
                dR_dx = invJ[0,0]*dR_du + invJ[0,1]*dR_dv
                dR_dy = invJ[1,0]*dR_du + invJ[1,1]*dR_dv

                # Build small B matrices (using small ordering [w, thx, thy])
                Bb = np.zeros((3, 3*nn))
                Bs = np.zeros((2, 3*nn))
                for a_local in range(nn):
                    # small ordering indices offset = 3*a_local
                    Bb[0, 3*a_local+1] = dR_dx[a_local]    # d(thx)/dx
                    Bb[1, 3*a_local+2] = dR_dy[a_local]    # d(thy)/dy
                    Bb[2, 3*a_local+1] = dR_dy[a_local]    # d(thx)/dy
                    Bb[2, 3*a_local+2] = dR_dx[a_local]    # d(thy)/dx
                    # shear: [thx + dw/dx; thy + dw/dy]
                    Bs[0, 3*a_local+0] = dR_dx[a_local]    # dw/dx
                    Bs[0, 3*a_local+1] = R[a_local]        # thx
                    Bs[1, 3*a_local+0] = dR_dy[a_local]    # dw/dy
                    Bs[1, 3*a_local+2] = R[a_local]        # thy

                weight = wu * wv * detJ
                Ke3 += (Bb.T @ D_b @ Bb + Bs.T @ D_s @ Bs) * weight
                for a_local in range(nn):
                    Fe3[3*a_local + 0] += R[a_local] * q_load * weight

        # Now embed Ke3 (3*nn x 3*nn) into Ke6 (6*nn x 6*nn) by mapping [w,thx,thy] -> indices [2,3,4]
        Ke6 = np.zeros((dof_per_node*nn, dof_per_node*nn))
        Fe6 = np.zeros(dof_per_node*nn)
        # map Fe3
        for a_local in range(nn):
            idx3 = 3*a_local + 0  # w DOF in small ordering
            idx6 = local3_to_local6_index(a_local, 0)
            Fe6[idx6] += Fe3[idx3]
        # embed Ke3 blocks
        for i3 in range(3*nn):
            inode_i = i3 // 3; iloc = i3 % 3
            I6 = local3_to_local6_index(inode_i, iloc)
            for j3 in range(3*nn):
                jnode = j3 // 3; jloc = j3 % 3
                J6 = local3_to_local6_index(jnode, jloc)
                Ke6[I6, J6] += Ke3[i3, j3]

        # Scatter Fe6 -> global F
        for a_local, I_ctrl in enumerate(elem_ctrls):
            # global dof base
            base6 = I_ctrl * dof_per_node
            local_base6 = a_local * dof_per_node
            F[base6:base6+dof_per_node] += Fe6[local_base6:local_base6+dof_per_node]

        # For Ke6: iterate local block pairs and add to global data blocks
        for a_local, I_ctrl in enumerate(elem_ctrls):
            for b_local, J_ctrl in enumerate(elem_ctrls):
                # extract local 6x6 block
                li = a_local * dof_per_node; lj = b_local * dof_per_node
                block_local6 = Ke6[li:li+dof_per_node, lj:lj+dof_per_node]
                pos = block_pos[(I_ctrl, J_ctrl)]
                data[pos] += block_local6

    # Add the +1 diagonal regularization for u (idx 0), v (idx 1), thz (idx 5) once per node
    for node in range(nctrl):
        pos = block_pos[(node, node)]
        data[pos][0,0] += 1.0   # u
        data[pos][1,1] += 1.0   # v
        data[pos][5,5] += 1.0   # thz

    # build indices array
    indices = np.empty(total_blocks, dtype=np.int32)
    cur = 0
    for i in range(nctrl):
        cols = indices_list[i]
        L = len(cols)
        indices[cur:cur+L] = cols
        cur += L

    # create BSR with blocksize = (6,6)
    bsr = sp.bsr_matrix((data, indices, indptr), shape=(ndof, ndof))
    return bsr, F, patch

# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    nxe = 8
    patch = Patch(Lx=1.0, Ly=1.0, nxe=nxe, nye=nxe, p=2, q=2, thickness=0.02,
                  E=70e9, nu=0.33, shear_correction=5/6,
                  scale_coords_by_thickness=True, scale_bending_by_1_over_th2=True)

    K_bsr, F, patch = assemble_FSDT_patch_bsr_6dof(patch, q_load=1.0)
    print("BSR built. data shape:", K_bsr.data.shape, "indices len:", K_bsr.indices.size, "indptr len:", K_bsr.indptr.size)

    # apply clamped BC (outer boundary) -- now each node has 6 DOFs
    nctrl = patch.n_ctrl_x * patch.n_ctrl_y
    dof_per_node = 6
    ndof = dof_per_node * nctrl
    fixed = []
    for i in range(patch.n_ctrl_x):
        for j in range(patch.n_ctrl_y):
            if i==0 or i==patch.n_ctrl_x-1 or j==0 or j==patch.n_ctrl_y-1:
                gid = i*patch.n_ctrl_y + j
                fixed += [dof_per_node*gid + d for d in range(dof_per_node)]
    all_dofs = np.arange(ndof)
    free = np.setdiff1d(all_dofs, np.array(fixed, dtype=int))

    K_csr = K_bsr.tocsr()
    K_red = K_csr[free,:][:,free]
    F_red = F[free]
    u_red = spla.spsolve(K_red, F_red)
    u = np.zeros(ndof)
    u[free] = u_red
    # vertical deflections at control points (w is index 2 in each 6-block)
    w = u[2::6]
    print("max deflection (control pts):", w.max())
