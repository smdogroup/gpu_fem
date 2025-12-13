# general method to figure out whether a fine node is in the coarse element for general unstructured meshes..
# probably do need some sort of proximity detection (like these are the nearest coarse nodes too.. otherwise could confuse it..)
# then need some sort of normal comp remover so you can compute xi, eta for it..

import numpy as np
import matplotlib.pyplot as plt


# is the cquad4 element actually just a plane? or is it slightly quadratic?
# ok it's technically slightly quadratic with (1-xi)*(1-eta) for example is quadratic due to the xy term..
# so actually not quite a plane (but we can approx as plane with centroid and normal at centroid..)

#  maybe a simpler way I can do it is compute the nearest neighbors and then add up for each element, a score of up to 5 nearest elems 
# and a freq, highest freq is the coarse element it's in..

# but then still how to get an approx xi, eta for the point?


# plan : do nearest neighbors with LocatePoint on CPU (6 nearest neighbors so can hit both elems for struct mesh case)
# then compute list of for each coarse node, which coarse elems it belongs to
# then need method to compute xi, eta values for the point on the coarse element.. (technically X(xi,eta) is quadratic surface.. can solve it with newton iteration?, after normal vec removed too?)
# then get up to 2 elems it belongs too.. ties would be ignored.. and store xi,eta values for each fine node also for each nearest elem.. but only do prolong/restrict on xi,eta in [-1.01, 1.01] some tolerance..

# for the xi,eta solve.. three xyz coords we match.. let's just compute one shell normal at (xi,eta) = (0,0) and fix that for simplicity..
# then we can say xyz = N(xi,eta) * nodal_pts + zeta * normal and that gives us three DOFs to solve.. let's solve it with newton itreation since it's quadratic.. should solve quickly.. let's do an example here


# perfectly flat element here for example
np.random.seed(1234)
normal = np.random.rand(3)
normal /= np.linalg.norm(normal)

norm_sum = 1.0 # rhs 
xvals, yvals = np.array([1.0, 2.0]), np.array([1.5, 2.5])
xpts = np.zeros(12)

# fillin xvals and yvals
xpts[0] = xvals[0]
xpts[3] = xvals[1]
xpts[6] = xvals[1]
xpts[9] = xvals[0]

xpts[1] = yvals[0]
xpts[4] = yvals[0]
xpts[7] = yvals[1]
xpts[10] = yvals[1]

for i in range(4):
    x, y = xpts[3*i], xpts[3*i+1]
    z = (norm_sum - normal[0] * x - normal[1] * y) / normal[2]
    xpts[3 * i + 2] = z

print(f"{xpts=}")

# define shape funcs eqn..
def xyz_interp(xis):
    xi, eta, zeta = xis[0], xis[1], xis[2]
    N = np.array([
        0.25 * (1 - xi) * (1 - eta),
        0.25 * (1 + xi) * (1 - eta),
        0.25 * (1 + xi) * (1 + eta),
        0.25 * (1 - xi) * (1 + eta),
    ])

    x = np.dot(N, xpts[0::3]) + zeta * normal[0]
    y = np.dot(N, xpts[1::3]) + zeta * normal[1]
    z = np.dot(N, xpts[2::3]) + zeta * normal[2]

    return np.array([x, y, z])

def xyz_dxi(xis):
    xi, eta, zeta = xis[0], xis[1], xis[2]
    N_xi = np.array([
        0.25 * -(1 - eta),
        0.25 * (1 - eta),
        0.25 * (1 + eta),
        0.25 * -(1 + eta),
    ])

    x_xi = np.dot(N_xi, xpts[0::3])
    y_xi = np.dot(N_xi, xpts[1::3])
    z_xi = np.dot(N_xi, xpts[2::3])

    return np.array([x_xi, y_xi, z_xi])

def xyz_deta(xis):
    xi, eta, zeta = xis[0], xis[1], xis[2]
    N_eta = np.array([
        -0.25 * (1 - xi),
        -0.25 * (1 + xi),
        0.25 * (1 + xi),
        0.25 * (1 - xi),
    ])

    x_eta = np.dot(N_eta, xpts[0::3])
    y_eta = np.dot(N_eta, xpts[1::3])
    z_eta = np.dot(N_eta, xpts[2::3])

    return np.array([x_eta, y_eta, z_eta])

def xyz_dxi_deta(xis):
    xi, eta, zeta = xis[0], xis[1], xis[2]
    N_eta = np.array([
        0.25,
        -0.25,
        0.25,
        -0.25,
    ])

    x_eta = np.dot(N_eta, xpts[0::3]) 
    y_eta = np.dot(N_eta, xpts[1::3])
    z_eta = np.dot(N_eta, xpts[2::3])

    return np.array([x_eta, y_eta, z_eta])

def xyz_dzeta(xis):
    xi, eta, zeta = xis[0], xis[1], xis[2]
    return normal


def solve_comp_coords(xyz_star):
    xis = np.zeros(3)

    for ct in range(4): # don't need many iterations

        xyz = xyz_interp(xis)
        delta_xyz = xyz - xyz_star

        dxi = xyz_dxi(xis)
        deta = xyz_deta(xis)
        dzeta = xyz_dzeta(xis)

        grad_0 = 2.0 * np.array([
            np.dot(delta_xyz, dxi),
            np.dot(delta_xyz, deta),
            np.dot(delta_xyz, dzeta),
        ])

        hess = 2.0 * np.array([
            [np.dot(dxi, dxi), np.dot(dxi, deta), 0.0],
            [np.dot(dxi, deta), np.dot(deta, deta), 0.0],
            [0.0, 0.0, np.dot(dzeta, dzeta)],
        ])

        dxi_deta = xyz_dxi_deta(xis) # second order term
        hess[0,1] += 2.0 * np.dot(delta_xyz, dxi_deta)
        hess[1,0] += 2.0 * np.dot(delta_xyz, dxi_deta)

        xis_update = -np.linalg.solve(hess, grad_0)

        xis += xis_update

        resid = np.linalg.norm(delta_xyz)
        print(f"{resid=:.2e}")

    # double check that xyz = xyz_star now..
    xyz_pred = xyz_interp(xis)
    resid = np.linalg.norm(xyz_pred - xyz_star)
    print(f"final {resid=:2e}")

    return xis


if __name__ == "__main__":

    xyz_star = np.array([1.3, 1.8, -1.0])
    xis_star = solve_comp_coords(xyz_star)

    print(f"{xis_star=}")