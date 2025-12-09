# I will assemble and solve a simple FSDT isogeometric plate solver (single-patch NURBS / B-spline).
# Unknowns per control point: [w, thx, thy]. Features:
# - degree p,q (default 2)
# - structured rectangular domain [0,Lx]x[0,Ly]
# - user sets nxe,nye (elements in each direction) -> control points = nxe+p x nye+q (open knot vector)
# - option to apply asymptotic scaling: scale_coords_by_thickness (multiplies physical coordinates by thickness),
#   and scale_bending_by_1_over_th2 (multiplies bending stiffness by 1/thickness^2).
# - assembles global stiffness, applies clamped BC on outer boundary, applies uniform transverse load q (default 1)
# - computes bending stresses (sigma_xx, sigma_yy, sigma_xy) at quad points and transverse shear stresses (tau_xz, tau_yz)
#
# This is a compact, readable implementation intended for clarity and experimentation rather than peak performance.
# Run the cell to compute and solve an example. Adjust parameters below as needed.

import numpy as np
from math import comb
from dataclasses import dataclass

# ------------------ Utilities: B-spline basis and derivatives ------------------
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

def find_span(n_ctrl, degree, u, U):
    # Cox-de Boor span search (n_ctrl = number of control points)
    if u >= U[-1] - 1e-12:
        return n_ctrl - 1
    low = degree
    high = len(U)-degree-1
    mid = (low+high)//2
    while True:
        if u < U[mid]:
            high = mid
        elif u >= U[mid+1]:
            low = mid
        else:
            return mid
        mid = (low+high)//2

def basis_functions_and_derivatives(span, u, degree, U, n_deriv=1):
    # Compute nonzero basis functions and first derivatives using Cox-de Boor + derivative formula
    # Returns arrays N[0:degree] and dN[0:degree]
    left = np.zeros(degree+1)
    right = np.zeros(degree+1)
    ndu = np.zeros((degree+1, degree+1))
    ndu[0,0] = 1.0
    for j in range(1, degree+1):
        left[j] = u - U[span+1-j]
        right[j] = U[span+j] - u
        saved = 0.0
        for r in range(j):
            ndu[j,r] = right[r+1] + left[j-r]
            temp = ndu[r,j-1]/ndu[j,r]
            ndu[r,j] = saved + right[r+1]*temp
            saved = left[j-r]*temp
        ndu[j,j] = saved
    N = ndu[:,degree].copy()
    # derivatives
    ders = np.zeros((n_deriv+1, degree+1))
    a = np.zeros((2, degree+1))
    # compute a triangular table of derivatives
    for r in range(degree+1):
        s1 = 0; s2 = 1
        a[0,0] = 1.0
        for k in range(1, n_deriv+1):
            d = 0.0
            rk = r - k
            pk = degree - k
            if r >= k:
                a[s2,0] = a[s1,0]/ndu[pk+1,rk]
                d = a[s2,0]*ndu[rk,pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k-1 if r-1 <= pk else degree - r
            for j in range(j1, j2+1):
                a[s2,j] = (a[s1,j] - a[s1,j-1]) / ndu[pk+1, rk+j]
                d += a[s2,j]*ndu[rk+j, pk]
            if r <= pk:
                a[s2,k] = -a[s1,k-1]/ndu[pk+1, r]
                d += a[s2,k]*ndu[r, pk]
            ders[k,r] = d
            s1, s2 = s2, s1
    # Multiply by correct factors
    for k in range(1, n_deriv+1):
        for j in range(degree+1):
            ders[k,j] *= degree
    return N, ders[1]

# ------------------ Geometry / patch ------------------
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

# ------------------ Assembly ------------------
def assemble_FSDT_patch(patch: Patch, q_load=1.0, quad_order=None):
    p = patch.p; q = patch.q
    nctrl = patch.n_ctrl_x * patch.n_ctrl_y
    ndof = 3 * nctrl  # [w, thx, thy] per control point
    if quad_order is None:
        quad_order = max(p+1, q+1)
    # material matrices
    t = patch.thickness
    E = patch.E; nu = patch.nu
    G = E/(2*(1+nu))
    D_b = E * t**3 / (12*(1-nu**2)) * np.array([[1, nu, 0],[nu,1,0],[0,0,(1-nu)/2]])
    if patch.scale_bending_by_1_over_th2:
        D_b *= 1.0/(t**2)
    D_s = patch.shear_correction * G * t * np.eye(2)
    # global stiffness and load
    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)
    # helper to map local (i,j) ctrl indices to global index
    def idx(i,j):
        return (i*patch.n_ctrl_y + j)
    # Precompute element knot spans (non-zero measure spans)
    U = patch.U; V = patch.V
    spans_x = []
    for a in range(len(U)-1):
        if U[a+1] - U[a] > 1e-12:
            spans_x.append((a, U[a], U[a+1]))
    spans_y = []
    for b in range(len(V)-1):
        if V[b+1] - V[b] > 1e-12:
            spans_y.append((b, V[b], V[b+1]))
    # Gauss points and weights (1D)
    def gauss(n):
        from numpy.polynomial.legendre import leggauss
        x,w = leggauss(n)
        # map from [-1,1] to [0,1]
        pts = 0.5*(x+1)
        w = 0.5*w
        return pts, w
    xi_pts, xi_w = gauss(quad_order)
    eta_pts, eta_w = gauss(quad_order)
    # Loop elements (knot spans)
    for sx, u0, u1 in spans_x:
        for sy, v0, v1 in spans_y:
            he = (u1-u0)*(v1-v0)  # parametric area scale (we map using jacobian)
            # element-active control points indices (span index = a means span between U[a],U[a+1]; active ctrl indices a-p ... a)
            i_start = sx - p; i_end = sx
            j_start = sy - q; j_end = sy
            # collect control pts and weights for this element
            elem_ctrl_idx = []
            elem_ctrl_coords = []
            elem_weights = []
            for ii in range(i_start, i_end+1):
                for jj in range(j_start, j_end+1):
                    elem_ctrl_idx.append((ii, jj))
                    elem_ctrl_coords.append(patch.ctrl_pts[ii, jj])
                    elem_weights.append(patch.weights[ii,jj])
            nel_dofs = len(elem_ctrl_idx)*3
            Ke = np.zeros((nel_dofs, nel_dofs))
            Fe = np.zeros(nel_dofs)
            # quadrature
            for i_g, xi in enumerate(xi_pts):
                u = u0 + xi*(u1-u0)
                wu = xi_w[i_g]*(u1-u0)
                span_u = find_span(patch.n_ctrl_x, p, u, U)
                Nu, dNu = basis_functions_and_derivatives(span_u, u, p, U, n_deriv=1)
                for j_g, eta in enumerate(eta_pts):
                    v = v0 + eta*(v1-v0)
                    wv = eta_w[j_g]*(v1-v0)
                    span_v = find_span(patch.n_ctrl_y, q, v, V)
                    Nv, dNv = basis_functions_and_derivatives(span_v, v, q, V, n_deriv=1)
                    # form tensor product shape functions for the patch-element local basis
                    # ordering must match elem_ctrl_idx build order: ii loop outside jj loop inside
                    R = []
                    dR_du = []
                    dR_dv = []
                    # basis indexes correspond to support range in each direction
                    iu0 = span_u - p
                    jv0 = span_v - q
                    for a in range(p+1):
                        for b in range(q+1):
                            Nu_a = Nu[a]
                            Nv_b = Nv[b]
                            R.append(Nu_a * Nv_b)
                            dR_du.append(dNu[a]*Nv_b)
                            dR_dv.append(Nu_a * dNv[b])
                    R = np.array(R); dR_du = np.array(dR_du); dR_dv = np.array(dR_dv)
                    # Map to physical coordinates (rational weighting if NURBS, here weights=1)
                    x = np.zeros(2)
                    dx_du = np.zeros(2); dx_dv = np.zeros(2)
                    for k,(ii,jj) in enumerate(elem_ctrl_idx):
                        cp = patch.ctrl_pts[ii,jj]
                        x += R[k]*cp
                        dx_du += dR_du[k]*cp
                        dx_dv += dR_dv[k]*cp
                    J = np.column_stack([dx_du, dx_dv])  # 2x2
                    detJ = np.linalg.det(J)
                    if detJ <= 0:
                        raise RuntimeError("Non-positive Jacobian det")
                    invJ = np.linalg.inv(J)
                    # derivatives wrt physical x,y
                    dR_dx = invJ[0,0]*dR_du + invJ[0,1]*dR_dv
                    dR_dy = invJ[1,0]*dR_du + invJ[1,1]*dR_dv
                    # Build B matrices
                    # ordering per control point DOFs: [w, thx, thy]
                    nn = len(R)
                    Bb = np.zeros((3, 3*nn))  # curvature from rotations: [d(phx)/dx; d(phy)/dy; d(phx)/dy + d(phy)/dx]
                    Bs = np.zeros((2, 3*nn))  # shear: [phi_x + dw/dx; phi_y + dw/dy]
                    for a in range(nn):
                        # w DOF index = 3*a, thx = 3*a+1, thy = 3*a+2
                        # curvature contributions from rotation DOFs derivatives
                        Bb[0, 3*a+1] = dR_dx[a]    # d(phx)/dx
                        Bb[1, 3*a+2] = dR_dy[a]    # d(phy)/dy
                        Bb[2, 3*a+1] = dR_dy[a]    # d(phx)/dy
                        Bb[2, 3*a+2] = dR_dx[a]    # d(phy)/dx
                        # shear
                        Bs[0, 3*a+0] = dR_dx[a]    # dw/dx coefficient
                        Bs[0, 3*a+1] = R[a]       # phi_x coefficient
                        Bs[1, 3*a+0] = dR_dy[a]    # dw/dy
                        Bs[1, 3*a+2] = R[a]       # phi_y
                    # element stiffness contribution
                    weight = wu*wv*detJ
                    Ke += (Bb.T @ D_b @ Bb + Bs.T @ D_s @ Bs) * weight
                    # load vector (transverse q applied to w DOF), note physical area scale = detJ
                    # if user wants load scaling by thickness or mu they can multiply q_load externally; here q_load is provided as argument
                    # contribution to w DOFs only:
                    for a in range(nn):
                        Fe[3*a+0] += R[a] * q_load * weight
            # Assemble into global K,F
            # Map local dofs to global
            dof_map = []
            for (ii,jj) in elem_ctrl_idx:
                gidx = idx(ii,jj)
                dof_map += [3*gidx+0, 3*gidx+1, 3*gidx+2]
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

# ------------------ Stress computation ------------------
def compute_stresses_at_quadrature(patch: Patch, u_global, quad_order=None):
    # evaluate bending curvatures and shear strains at quadrature points and convert to stresses
    if quad_order is None:
        quad_order = max(patch.p+1, patch.q+1)
    E = patch.E; nu = patch.nu
    G = E/(2*(1+nu))
    D_b = E * patch.thickness**3 / (12*(1-nu**2)) * np.array([[1, nu, 0],[nu,1,0],[0,0,(1-nu)/2]])
    if patch.scale_bending_by_1_over_th2:
        D_b *= 1.0/(patch.thickness**2)
    # iterate same as assembly but only compute strains at quad points
    U = patch.U; V = patch.V
    p = patch.p; q = patch.q
    spans_x = [(a,U[a],U[a+1]) for a in range(len(U)-1) if U[a+1]-U[a]>1e-12]
    spans_y = [(b,V[b],V[b+1]) for b in range(len(V)-1) if V[b+1]-V[b]>1e-12]
    from numpy.polynomial.legendre import leggauss
    xg, wg = leggauss(quad_order)
    xi_pts = 0.5*(xg+1); xi_w = 0.5*wg
    eta_pts = xi_pts; eta_w = xi_w
    stresses = []  # list of dicts per quad point
    def idx(i,j): return (i*patch.n_ctrl_y + j)
    for sx, u0, u1 in spans_x:
        for sy, v0, v1 in spans_y:
            i_start = sx - p; i_end = sx
            j_start = sy - q; j_end = sy
            elem_ctrl_idx = []
            for ii in range(i_start, i_end+1):
                for jj in range(j_start, j_end+1):
                    elem_ctrl_idx.append((ii,jj))
            nn = len(elem_ctrl_idx)
            for i_g, xi in enumerate(xi_pts):
                u = u0 + xi*(u1-u0)
                span_u = find_span(patch.n_ctrl_x, p, u, U)
                Nu, dNu = basis_functions_and_derivatives(span_u, u, p, U, n_deriv=1)
                for j_g, eta in enumerate(eta_pts):
                    v = v0 + eta*(v1-v0)
                    span_v = find_span(patch.n_ctrl_y, q, v, V)
                    Nv, dNv = basis_functions_and_derivatives(span_v, v, q, V, n_deriv=1)
                    R = []; dR_du = []; dR_dv = []
                    for a in range(p+1):
                        for b in range(q+1):
                            R.append(Nu[a]*Nv[b])
                            dR_du.append(dNu[a]*Nv[b])
                            dR_dv.append(Nu[a]*dNv[b])
                    R=np.array(R); dR_du=np.array(dR_du); dR_dv=np.array(dR_dv)
                    x = np.zeros(2); dx_du=np.zeros(2); dx_dv=np.zeros(2)
                    for k,(ii,jj) in enumerate(elem_ctrl_idx):
                        cp = patch.ctrl_pts[ii,jj]
                        x += R[k]*cp
                        dx_du += dR_du[k]*cp
                        dx_dv += dR_dv[k]*cp
                    J = np.column_stack([dx_du, dx_dv])
                    detJ=np.linalg.det(J); invJ=np.linalg.inv(J)
                    dR_dx = invJ[0,0]*dR_du + invJ[0,1]*dR_dv
                    dR_dy = invJ[1,0]*dR_du + invJ[1,1]*dR_dv
                    # collect element dofs
                    u_e = np.zeros(3*nn)
                    for k,(ii,jj) in enumerate(elem_ctrl_idx):
                        gidx = idx(ii,jj)
                        u_e[3*k+0] = u_global[3*gidx+0]
                        u_e[3*k+1] = u_global[3*gidx+1]
                        u_e[3*k+2] = u_global[3*gidx+2]
                    # compute curvature vector
                    curv = np.zeros(3)
                    # curvature = [d(phx)/dx; d(phy)/dy; d(phx)/dy + d(phy)/dx]
                    for a in range(nn):
                        curv[0] += dR_dx[a] * u_e[3*a+1]
                        curv[1] += dR_dy[a] * u_e[3*a+2]
                        curv[2] += dR_dy[a] * u_e[3*a+1] + dR_dx[a] * u_e[3*a+2]
                    # shear strains = [phi_x + dw/dx; phi_y + dw/dy]
                    gamma = np.zeros(2)
                    for a in range(nn):
                        gamma[0] += R[a]*u_e[3*a+1] + dR_dx[a]*u_e[3*a+0]
                        gamma[1] += R[a]*u_e[3*a+2] + dR_dy[a]*u_e[3*a+0]
                    # bending moments M = D_b * curv
                    M = D_b @ curv
                    # bending stresses at top/bottom z = +- t/2: sigma = (12*z/t^3) * M? Simpler: M_x = ∫ z*sigma_xx dz -> for isotropic sigma_xx(z)= (E/(1-nu^2))*(kappa_x * z + nu*kappa_y*z) etc.
                    # We'll compute classical linear bending stress at z: sigma = z * Cb * curv, with Cb = E/(1-nu^2) * [[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]]
                    Cb = E/(1-nu**2) * np.array([[1, nu, 0],[nu,1,0],[0,0,(1-nu)/2]])
                    z_top = +patch.thickness/2.0
                    z_bot = -patch.thickness/2.0
                    sigma_top = z_top * (Cb @ curv)
                    sigma_bot = z_bot * (Cb @ curv)
                    # shear stresses (FSDT treats shear as constant through thickness): tau = G * gamma (with correction)
                    tau = G * gamma * patch.shear_correction
                    stresses.append({
                        'x': x[0], 'y': x[1], 'curv': curv.copy(), 'M': M.copy(),
                        'sigma_top': sigma_top.copy(), 'sigma_bot': sigma_bot.copy(),
                        'tau_xz': tau[0], 'tau_yz': tau[1]
                    })
    return stresses

# ------------------ Example run ------------------
# User-configurable parameters
# nxe = 4
# nxe = 16
nxe = 32
# nxe = 64

patch = Patch(Lx=1.0, Ly=1.0, nxe=nxe, nye=nxe, p=2, q=2, thickness=0.02,
              E=70e9, nu=0.33, shear_correction=5/6,
              scale_coords_by_thickness=True, scale_bending_by_1_over_th2=True)
K, F, patch = assemble_FSDT_patch(patch, q_load=1.0)
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

# compute stresses at quadrature points
stresses = compute_stresses_at_quadrature(patch, u)
print("Computed stresses at", len(stresses), "quad points. Sample entry:")
print(stresses[len(stresses)//2])

# Display the vertical displacement array (control points)
import pandas as pd
df = pd.DataFrame(w_mat, index=[f'i={i}' for i in range(patch.n_ctrl_x)], columns=[f'j={j}' for j in range(patch.n_ctrl_y)])
# import caas_jupyter_tools as jt; jt.display_dataframe_to_user("Vertical deflection (w) at control points", df)
print(F"{df=}")

# # Save results to /mnt/data for download if desired
# import json
# out = {"w_control_pts": w.tolist(), "patch": {"n_ctrl_x": patch.n_ctrl_x, "n_ctrl_y": patch.n_ctrl_y, "thickness": patch.thickness}}
# with open("/mnt/data/fsdt_plate_results.json","w") as f:
#     json.dump(out, f)

# print("[Download the results JSON file] (sandbox:/mnt/data/fsdt_plate_results.json)")

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