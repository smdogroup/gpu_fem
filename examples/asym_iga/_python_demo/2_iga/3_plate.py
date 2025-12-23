import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--nxe", type=int, default=4)
parser.add_argument("--L", type=float, default=1.0)
parser.add_argument("--thick", type=float, default=0.01)
parser.add_argument("--load", type=float, default=1e4)
args = parser.parse_args()


# ===========================
# PROBLEM SETUP
# ===========================

# assume square grid
nxe = nye = args.nxe
p = 2 # order of basis
nnx = nny = nxe + p
Lx = Ly = args.L

# material
E = 70e9; nu = 0.33; ks = 5.0/6.0
thick = args.thick

# contruct knots U in x dir and V in y direction
x_knots = y_knots = [0.0]*(p+1) + [i/nxe for i in range(1,nxe)] + [1.0]*(p+1)

# construct the x and y control points
x_coords = np.linspace(0.0, Lx, nnx)
y_coords = np.linspace(0.0, Ly, nny)

# just doing B-splines not full NURBS first.. (just plate not cylinder)
# so no weights just x and y coords

nelems = nxe * nye
nnodes = nnx * nny # aka num control points (not really a node)
NDOF = 3 * nnodes 

# stiffnesses
G = E/(2*(1+nu))
t = thick
D_b = E * t**3 / (12*(1-nu**2)) * np.array([[1, nu, 0],[nu,1,0],[0,0,(1-nu)/2]])
D_s = ks * G * t * np.eye(2)

# global stiffness matrix
K = np.zeros((NDOF, NDOF))
F = np.zeros(NDOF)

# helper to map local (i,j) ctrl indices to global index
def idx(i,j):
    return j * nnx + i

# Gauss points and weights (1D)
def gauss(n):
    from numpy.polynomial.legendre import leggauss
    x,w = leggauss(n)
    # map from [-1,1] to [0,1]
    pts = 0.5*(x+1)
    w = 0.5*w
    return pts, w
quad_pts, quad_wts = gauss(p+1)
_pts = quad_pts*2-1.0
print(f"{_pts=}")

def quad_bernstein(xi):
    N = np.array([(1-xi)**2, 2*xi*(1-xi), xi**2])
    dN = np.array([-2*(1-xi), 2*(1-2*xi), 2*xi])
    return N, dN

def get_1d_basis_and_deriv(xi, bndry_span):
    assert len(bndry_span) == 4
    B, dB = quad_bernstein(xi)
    left_bndry = abs(bndry_span[0] - bndry_span[1]) < 1e-12
    right_bndry = abs(bndry_span[2] - bndry_span[3]) < 1e-12

    # print(f"{left_bndry=} {right_bndry=}")

    # bndry adjustment and regular basis
    # on GPU can code it up with ? ternary operators probably
    N = 0.5 * np.array([B[0], np.sum(B) + B[1], B[2]])
    N += 0.5 * left_bndry * np.array([B[0], -B[0], 0.0])
    N += 0.5 * right_bndry * np.array([0.0, -B[2], B[2]])

    # bndry adjustment and regular derivs
    dN = 0.5 * np.array([dB[0], np.sum(dB) + dB[1], dB[2]])
    dN += 0.5 * left_bndry * np.array([dB[0], -dB[0], 0.0])
    dN += 0.5 * right_bndry * np.array([0.0, -dB[2], dB[2]])
    return N, dN

# loop over each element
for ielem in range(nelems):

    ixe = ielem % nxe; iye = ielem // nxe
    # even though 2nd order elements have 2 knots in each dir
    # it has 3 control points each direction
    # in fact higher order (also have only 2 knots each direction), but extra basis functions
    
    # get the element control points and global control indices
    elem_ctrl_idx = []; elem_xpts = []
    for i in range((p+1)**2):
        ii = i % (p+1); jj = i // (p+1)
        ix = ixe + ii; iy = iye + jj        
        elem_ctrl_idx += [idx(ix, iy)]
        elem_xpts += [(x_coords[ix], y_coords[iy])]
    # print(f"{elem_ctrl_idx=}\n{elem_xpts=}")
    # exit()

    # get the current knot spans including one more each way
    x_bndry_span = x_knots[(p+ixe-1):(p+ixe+3)] # one more each side (to check bndry)
    y_bndry_span = y_knots[(p+iye-1):(p+iye+3)] # one more each side (to check bndry)
    x_span = x_bndry_span[1:-1] # two knots only
    y_span = y_bndry_span[1:-1] # two knots only
    # print(f"{x_span=} {y_span=}")
    # exit()

    # now construct kelem and felem
    nelem_nodes = (p+1)**2
    nelem_DOF = 3 * nelem_nodes # = 27 for p=2
    Ke = np.zeros((nelem_DOF,nelem_DOF))
    Fe = np.zeros(nelem_DOF)

    # Gauss quadrature
    for iquad in range(nelem_nodes):
        # get quad pts and weights
        ixi, ieta = iquad % (p+1), iquad // (p+1)
        xi, xiw = quad_pts[ixi], quad_wts[ixi]
        eta, etaw = quad_pts[ieta], quad_wts[ieta]

        # get current patch coords (u,v)
        u = x_span[0] + xi * np.diff(x_span)[0]
        v = y_span[0] + eta * np.diff(y_span)[0]

        # get 1d basis functions and derivs
        print(f"{x_bndry_span=} {y_bndry_span=}")
        na, dna = get_1d_basis_and_deriv(xi, x_bndry_span)
        nb, dnb = get_1d_basis_and_deriv(eta, y_bndry_span)
        # print(f"{na=} {nb=} {dna=}")

        R, dR_da, dR_db = np.zeros(nelem_nodes), np.zeros(nelem_nodes), np.zeros(nelem_nodes)
        for i in range(nelem_nodes):
            ia = i % (p+1); ib = i // (p+1)
            # print(f"{ia=} {ib=}")
            R[i] = na[ia] * nb[ib]
            dR_da[i] = dna[ia] * nb[ib]
            dR_db[i] = na[ia] * dnb[ib]

        print(f"{R=}\n{dR_da=}\n{dR_db=}")
        # plt.imshow(R.reshape((3,3)))
        # plt.show()

        # compute d(xy)/d(ab) derivs at the quadpt
        xy_ab = np.zeros((2,2))
        for loc_node in range(nelem_nodes):
            xy = elem_xpts[loc_node]
            xy_ab[0,0] += xy[0] * dR_da[loc_node]
            xy_ab[0,1] += xy[0] * dR_db[loc_node]
            xy_ab[1,0] += xy[1] * dR_da[loc_node]
            xy_ab[1,1] += xy[1] * dR_db[loc_node]
        # xy_ab = xy_ab.T

        # print(f"{xy_ab=}")
        # exit()
            
        # now compute the jacobian and xy derivatives
        invJ = np.linalg.inv(xy_ab)
        detJ = np.linalg.det(xy_ab)
        dR_dx = invJ[0,0] * dR_da + invJ[0,1] * dR_db
        dR_dy = invJ[1,0] * dR_da + invJ[1,1] * dR_db

        # now build the strain-disp matrices for plate theory
        nn = len(R)
        Bb = np.zeros((3, 3*nn))  # curvature from rotations: [d(phx)/dx; d(phy)/dy; d(phx)/dy + d(phy)/dx]
        Bs = np.zeros((2, 3*nn))  # shear: [phi_x + dw/dx; phi_y + dw/dy]
        for a in range(nn):
            # w DOF index = 3*a, thx = 3*a+1, thy = 3*a+2
            # curvature contributions from rotation DOFs derivatives
            Bb[0, 3*a+2] = dR_dx[a]    # d(phy)/dx
            Bb[1, 3*a+1] = -dR_dy[a]    # -d(phx)/dy
            Bb[2, 3*a+2] = dR_dy[a]    # d(phy)/dy
            Bb[2, 3*a+1] = -dR_dx[a]    # -d(phx)/dx
            # shear
            Bs[0, 3*a+0] = dR_dx[a]    # dw/dx coefficient
            Bs[0, 3*a+2] = R[a]       # phi_y coefficient
            Bs[1, 3*a+0] = dR_dy[a]    # dw/dy
            Bs[1, 3*a+1] = -R[a]       # -phi_x

        # print(f"{Bb=}")
        # B = np.concatenate([Bb, Bs], axis=0)
        # plt.imshow(B)
        # plt.show()

        # element stiffness contribution
        weight = xiw * etaw * detJ
        Ke += (Bb.T @ D_b @ Bb + Bs.T @ D_s @ Bs) * weight
        # load vector (transverse q applied to w DOF), note physical area scale = detJ
        # if user wants load scaling by thickness or mu they can multiply q_load externally; here q_load is provided as argument
        # contribution to w DOFs only:
        for a in range(nn):
            Fe[3*a+0] += R[a] * args.load * weight

        # Assemble into global K,F
            # Map local dofs to global
            dof_map = []
            # print(f"{elem_ctrl_idx=}")
            for inode in elem_ctrl_idx:
                # gidx = idx(ii,jj)
                dof_map += [3*inode+0, 3*inode+1, 3*inode+2]
            for ii_local, I in enumerate(dof_map):
                F[I] += Fe[ii_local]
                for jj_local, Jidx in enumerate(dof_map):
                    K[I,Jidx] += Ke[ii_local, jj_local]

print(f"{K[:3,:3]=}")

# ========================================
# SOLVE THE LINEAR SYSTEM
# ========================================

# enforce clamped boundary conditions on linear system
# clamp all control points on outer boundary (w=0, thx=0, thy=0)
fixed = []
for i in range(nnx):
    for j in range(nny):
        if i==0 or i==nnx-1 or j==0 or j==nny-1:
            gid = idx(i,j)
            fixed += [3*gid+0, 3*gid+1, 3*gid+2]
free = np.setdiff1d(np.arange(K.shape[0]), np.array(fixed, dtype=int))
K_red = K[np.ix_(free, free)]
F_red = F[free]


# solve the linear system using LU solve in python
u_red = np.linalg.solve(K_red, F_red)
u = np.zeros(K.shape[0])
u[free] = u_red


# ========================================
# PLOT
# ========================================

w = u[0::3]
w_mat = w.reshape((nnx, nny))

# Create meshgrid of control point coordinates
X, Y = np.meshgrid(x_coords, y_coords)

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
