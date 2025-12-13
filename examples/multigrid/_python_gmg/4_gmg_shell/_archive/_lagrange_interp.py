# consider lagrange basis and some measured derivs on it
import numpy as np
import matplotlib.pyplot as plt
import niceplots

N_f = 50
# N_f = 10
# N_f = 6

# xmax = 1.0
xmax = 0.5

N_c = N_f // 2
x_c = np.linspace(0.0, xmax, N_c + 1)
y_c = np.sin(np.pi * x_c)
x_f = np.linspace(0.0, xmax, N_f + 1)
y_f = np.sin(np.pi * x_f)

def plot_coarse_fine(_y_f_hat):
    plt.style.use(niceplots.get_style())
    fig, ax = plt.subplots(1, 3)
    ax[0].plot(x_f, y_f, 'g-', label='fine')
    ax[0].plot(x_f, _y_f_hat, 'k-', label='fine-hat')
    ax[0].plot(x_c, y_c, '--', color='tab:gray', label='coarse')
    ax[0].legend()
    ax[0].set_title("y(x)")
    # plt.show()

    # now get the 1st derivs for stresses
    dy_c = 0.5 * 0.5 * (y_c[1:] - y_c[:-1])
    xd_c = 0.5 * (x_c[1:] + x_c[:-1])
    dy_f_hat = 0.5 * (_y_f_hat[1:] - _y_f_hat[:-1])
    xd_f = 0.5 * (x_f[1:] + x_f[:-1])
    dy_f = 0.5 * (y_f[1:] - y_f[:-1])

    ax[1].plot(xd_f, dy_f, 'g-', label='fine')
    ax[1].plot(xd_f, dy_f_hat, 'k-', label='fine-hat')
    ax[1].plot(xd_c, dy_c, '--', color='tab:gray', label='coarse')
    ax[1].legend()
    ax[1].set_title("dy/dx")
    # plt.show()

    # now get 2nd derivs for load computations
    dy2_c = 0.25 * 0.25 * (y_c[2:] - 2 * y_c[1:-1] + y_c[:-2])
    xd2_c = x_c[1:-1]
    dy2_f_hat = 0.25 * (_y_f_hat[2:] - 2 * _y_f_hat[1:-1] + _y_f_hat[:-2])
    xd2_f = x_f[1:-1]
    dy2_f = 0.25 * (y_f[2:] - 2 * y_f[1:-1] + y_f[:-2])

    ax[2].plot(xd2_f, dy2_f, 'g-', label='fine')
    ax[2].plot(xd2_f, dy2_f_hat, 'k-', label='fine-hat')
    ax[2].plot(xd2_c, dy2_c, '--', color='tab:gray', label='coarse')
    ax[2].legend()
    ax[2].set_title("d2y/dx2")
    plt.show()


# now interp to fine using lagrange basis per say
y_f_hat = np.zeros_like(x_f)
y_f_hat[0::2] = y_c[:]
y_f_hat[1::2] = 0.5 * (y_c[:-1] + y_c[1:])

# plot_coarse_fine(y_f_hat) # plots it for lagrange basis prolongation


# now try min L2-norm prolongation operator (from p-multigrid paper)
# see if that has smoother derivatives (and if you need to you can just do D^-1 not full M^-1 Gram matrix inverse for computational efficiency)
# need to compute the area integrals and gram matrix entries..

# first let's construct the M matrix using 2nd order gauss quadrature (it is sparse, but going to store as dense for now)
nnodes = N_f + 1
nelems = N_f
M = np.zeros((nnodes, nnodes))
h = xmax / nelems

irt3 = 1.0 / np.sqrt(3)

# M_ij = int(phi_i * phi_j, dx)
for inode in range(nnodes):
    for jnode in range(nnodes):
        # need to be adjacent nodes to get gauss quadrature
        if inode == jnode: # same basis function (spans two elements)
            elems = [inode-1, inode]
            
            for i, elem in enumerate(elems):
                if elem < 0: continue
                if elem > nelems-1: continue

                M[inode, inode] += 1.0 * (irt3 * 0.5 + 0.5)**2 * h
                M[inode, inode] += 1.0 * (-irt3 * 0.5 + 0.5)**2 * h

        elif np.abs(inode - jnode) <= 1: # they are one node apart
            M[inode, jnode] += 2.0 * (0.5 - 0.5 * irt3) * (0.5 + 0.5 * irt3) * h

# show M matrix
# plt.imshow(M)
# plt.show()

# now construct the coarse-fine version of gram matrix P_ij = int(phi_i * psi_j, dx)
nnodes_c = N_c + 1
nelems_c = N_c
P = np.zeros((nnodes, nnodes_c))
for jnode_c in range(nnodes_c):
    jnode_f = 2 * jnode_c
    ct = 0

    for inode_f in range(nnodes):
        if np.abs(inode_f - jnode_f) <= 2: # otherwise too far apart and no overlap

            elems = [inode_f - 1, inode_f]
            for i, elem in enumerate(elems):
                if elem < 0: continue
                if elem > nelems - 1: continue
                if elem - jnode_f >= 2: continue
                if jnode_f - elem > 2: continue

                ct += 1

                for xi in [-irt3, irt3]:
                    # eval at each quadpt
                    loc = elem + 0.5 * (1 + xi)
                    phi_i = 1.0 - np.abs(inode_f - loc)
                    psi_j = 1.0 - np.abs(0.5 * (jnode_f - loc))

                    # print(f"{inode_f=} {jnode_c=} {elem=} {loc=} {phi_i=:.2e} {psi_j=:.2e}") 
                    if psi_j < 0: 
                        raise AssertionError(f"{psi_j=} at {inode_f=} {jnode_f=} {elem=}")

                    P[inode_f, jnode_c] += 1.0 * phi_i * psi_j * h

    # print(f"{jnode_c=} hits {ct=} fine elems")
# plt.imshow(P)
# plt.show()

# now apply min L2 norm prolongation operator (it comes from minimizing L2 norm^2 of integrated fine vs coarse basis error, which may lead to smoother solution)
# and this comes from p-multigrid book

# v1 - using the full mass matrix
# TODO : need to eliminate bcs here so don't include that in interp?
# y_f_hat2 = np.linalg.solve(M, np.dot(P, y_c))

# v2 - using only diag of mass matrix (more comp efficient and suggested from the book)
m_cons_diag = np.sum(M, axis=1) # row sums of the mass matrix
y_f_hat2 = np.dot(np.diag(1.0 / m_cons_diag), np.dot(P, y_c))
# we also just force bcs to zero (left point) instead of removing from prolongation (but you can do that too)
y_f_hat2[0] = 0.0

# I = np.linalg.solve(M, P)
# plt.imshow(I)
# plt.show()
# y_f_hat_diff = y_f_hat - y_f_hat2
# print(f"{y_f_hat_diff}")

plot_coarse_fine(y_f_hat2) # plots it for L2 norm prolongation