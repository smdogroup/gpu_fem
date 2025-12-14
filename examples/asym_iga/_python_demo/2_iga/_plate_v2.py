import numpy as np
from dataclasses import dataclass

# hard-coded quadratic Bernstein basis and derivatives
def quad_bernstein(xi):
    N = np.array([(1-xi)**2, 2*xi*(1-xi), xi**2])
    dN = np.array([-2*(1-xi), 2*(1-2*xi), 2*xi])
    return N, dN

# ------------------ Geometry / patch ------------------
def make_open_knot_vector(n_ctrl, degree):
    # open knot vector with clamped ends, uniform interior knots
    m = n_ctrl + degree + 1
    n_elems = n_ctrl - degree
    kv = [0.0]*(degree+1)
    if n_elems>0:
        for i in range(1, n_elems):
            kv.append(i/n_elems)
    kv += [1.0]*(degree+1)
    return np.array(kv)


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

        print(f"{self.U=}\n{self.V=}")

        # create grid of control points (uniform) in parametric domain mapped to physical rectangle
        x_coords = np.linspace(0, self.Lx, self.n_ctrl_x)
        y_coords = np.linspace(0, self.Ly, self.n_ctrl_y)
        # possibly scale coordinates by thickness for the asymptotic transform
        if self.scale_coords_by_thickness:
            scale = self.thickness
        else:
            scale = 1.0
        self.ctrl_pts = np.zeros((self.n_ctrl_x, self.n_ctrl_y, 2))
        for i,x in enumerate(x_coords):
            for j,y in enumerate(y_coords):
                self.ctrl_pts[i,j,0] = x*scale
                self.ctrl_pts[i,j,1] = y*scale
        # weights (all ones => B-spline). NURBS could be added by specifying w != 1.
        self.weights = np.ones((self.n_ctrl_x, self.n_ctrl_y))

def assemble_FSDT_patch_quadratic(patch, q_load=1.0, quad_order=3):
    nctrl_x, nctrl_y = patch.n_ctrl_x, patch.n_ctrl_y
    ndof = 3 * nctrl_x * nctrl_y
    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)

    # material matrices
    t = patch.thickness
    E, nu = patch.E, patch.nu
    G = E/(2*(1+nu))
    D_b = E * t**3 / (12*(1-nu**2)) * np.array([[1, nu, 0],[nu,1,0],[0,0,(1-nu)/2]])
    if patch.scale_bending_by_1_over_th2:
        D_b *= 1.0/(t**2)
    D_s = patch.shear_correction * G * t * np.eye(2)

    # quadrature points on [0,1]
    from numpy.polynomial.legendre import leggauss
    xg, wg = leggauss(quad_order)
    xi_pts = 0.5*(xg+1); xi_w = 0.5*wg
    eta_pts, eta_w = xi_pts, xi_w

    # ------------------ determine non-zero knot spans ------------------
    elem_spans_u = []
    for i in range(patch.p, len(patch.U)-patch.p-1):
        if patch.U[i+1] > patch.U[i]:
            elem_spans_u.append((i, patch.U[i], patch.U[i+1]))

    elem_spans_v = []
    for j in range(patch.q, len(patch.V)-patch.q-1):
        if patch.V[j+1] > patch.V[j]:
            elem_spans_v.append((j, patch.V[j], patch.V[j+1]))

    # ------------------ loop over elements ------------------
    for i_span, u_left, u_right in elem_spans_u:
        for j_span, v_left, v_right in elem_spans_v:

            # get control point indices for current element
            ctrl_i_start = i_span - patch.p
            ctrl_j_start = j_span - patch.q

            elem_idx = []
            for di in range(patch.p+1):
                for dj in range(patch.q+1):
                    ii = ctrl_i_start + di
                    jj = ctrl_j_start + dj
                    elem_idx.append((ii,jj))  # do NOT clamp indices

            nn = len(elem_idx)
            Ke = np.zeros((3*nn,3*nn))
            Fe = np.zeros(3*nn)

            # something is still slightly wrong in this version of the method near the boundary

            for xi_q, w_xi in zip(xi_pts, xi_w):
                xi_hat = xi_q  # quadrature in [0,1]
                N_xi, dN_xi_hat = quad_bernstein(xi_hat)
                dN_xi = dN_xi_hat / (u_right - u_left)  # chain rule

                for eta_q, w_eta in zip(eta_pts, eta_w):
                    eta_hat = eta_q
                    N_eta, dN_eta_hat = quad_bernstein(eta_hat)
                    dN_deta = dN_eta_hat / (v_right - v_left)

                    # tensor-product basis
                    R = np.zeros(nn)
                    dR_dxi = np.zeros(nn)
                    dR_deta = np.zeros(nn)
                    coords = np.zeros((nn,2))
                    # for k, (ii,jj) in enumerate(elem_idx):
                    #     local_i = ii - ctrl_i_start
                    #     local_j = jj - ctrl_j_start
                    #     R[k] = N_xi[local_i] * N_eta[local_j]
                    #     dR_dxi[k] = dN_xi[local_i] * N_eta[local_j]
                    #     dR_deta[k] = N_xi[local_i] * dN_deta[local_j]
                    #     coords[k,:] = patch.ctrl_pts[ii,jj]

                    for di in range(patch.p+1):
                        for dj in range(patch.q+1):
                            ii = ctrl_i_start + di
                            jj = ctrl_j_start + dj
                            k = di*(patch.q+1) + dj
                            R[k] = N_xi[di] * N_eta[dj]
                            dR_dxi[k] = dN_xi[di] * N_eta[dj]
                            dR_deta[k] = N_xi[di] * dN_deta[dj]
                            coords[k,:] = patch.ctrl_pts[ii,jj]


                    # Jacobian
                    dx_dxi = np.dot(dR_dxi, coords[:,0])
                    dy_dxi = np.dot(dR_dxi, coords[:,1])
                    dx_deta = np.dot(dR_deta, coords[:,0])
                    dy_deta = np.dot(dR_deta, coords[:,1])
                    J = np.array([[dx_dxi, dx_deta],[dy_dxi, dy_deta]])
                    detJ = np.linalg.det(J)
                    invJ = np.linalg.inv(J)

                    # derivatives wrt physical x,y
                    dR_dx = invJ[0,0]*dR_dxi + invJ[0,1]*dR_deta
                    dR_dy = invJ[1,0]*dR_dxi + invJ[1,1]*dR_deta

                    # B matrices
                    Bb = np.zeros((3,3*nn))
                    Bs = np.zeros((2,3*nn))
                    for a in range(nn):
                        Bb[0,3*a+1] = dR_dx[a]
                        Bb[1,3*a+2] = dR_dy[a]
                        Bb[2,3*a+1] = dR_dy[a]
                        Bb[2,3*a+2] = dR_dx[a]

                        Bs[0,3*a+0] = dR_dx[a]
                        Bs[0,3*a+1] = R[a]
                        Bs[1,3*a+0] = dR_dy[a]
                        Bs[1,3*a+2] = R[a]

                    weight = w_xi * w_eta * detJ * (u_right-u_left)*(v_right-v_left)
                    Ke += (Bb.T @ D_b @ Bb + Bs.T @ D_s @ Bs) * weight
                    for a in range(nn):
                        Fe[3*a+0] += R[a] * q_load * weight

            # assemble into global K,F
            dof_map = []
            for ii,jj in elem_idx:
                gidx = ii*nctrl_y + jj
                dof_map += [3*gidx, 3*gidx+1, 3*gidx+2]
            for ii_local, I in enumerate(dof_map):
                F[I] += Fe[ii_local]
                for jj_local, Jidx in enumerate(dof_map):
                    K[I,Jidx] += Ke[ii_local, jj_local]

    return K, F, patch



# ------------------ Boundary conditions & solve ------------------
def apply_clamped_bc(K, F, patch: Patch):
    # clamp all control points on outer boundary (w=0, thx=0, thy=0)
    nctrlx = patch.n_ctrl_x; nctrly = patch.n_ctrl_y
    fixed = []
    for i in range(nctrlx):
        for j in range(nctrly):
            if i==0 or i==nctrlx-1 or j==0 or j==nctrly-1:
                gid = (i*nctrly + j)
                fixed += [3*gid+0, 3*gid+1, 3*gid+2]
    free = np.setdiff1d(np.arange(K.shape[0]), np.array(fixed, dtype=int))
    K_reduced = K[np.ix_(free, free)]
    F_reduced = F[free]
    return K_reduced, F_reduced, free, fixed

# ------------------ Example run ------------------

# nxe = 4
nxe = 16
# nxe = 32

# User-configurable parameters
patch = Patch(Lx=1.0, Ly=1.0, nxe=nxe, nye=nxe, p=2, q=2, thickness=0.02,
              E=70e9, nu=0.33, shear_correction=5/6,
              scale_coords_by_thickness=True, scale_bending_by_1_over_th2=True)
K, F, patch = assemble_FSDT_patch_quadratic(patch, q_load=1.0e7)
K_red, F_red, free, fixed = apply_clamped_bc(K, F, patch)
# solve
u_red = np.linalg.solve(K_red, F_red)
u = np.zeros(K.shape[0])
u[free] = u_red
# extract transverse deflections at control points
nctrl = patch.n_ctrl_x*patch.n_ctrl_y
w = np.zeros(nctrl)
for i in range(nctrl):
    w[i] = u[3*i+0]
w_mat = w.reshape((patch.n_ctrl_x, patch.n_ctrl_y))
print("Solved. control grid size:", patch.n_ctrl_x, "x", patch.n_ctrl_y)
print("Max vertical deflection (control pts):", w.max())

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # optional, needed for 3D plotting

# Create meshgrid of control point coordinates
X = patch.ctrl_pts[:,:,0]
Y = patch.ctrl_pts[:,:,1]

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Plot surface
surf = ax.plot_surface(X, Y, w_mat, cmap='viridis', edgecolor='k')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('w')
ax.set_title('Vertical deflection w (control points)')
fig.colorbar(surf, shrink=0.6, aspect=10)
plt.show()
