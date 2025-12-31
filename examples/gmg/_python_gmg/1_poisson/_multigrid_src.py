"""
some basic multigrid routines for the Poisson plate case
"""

from _pde_src import *

def _get_LDU_parts(A):
    """get lower, diagonal and upper parts"""
    L, D = np.tril(A, k=-1), np.diag(np.diag(A))
    U = A - L - D
    return L, D, U

def gauss_seidel(A, b, x0, omega:float=1.0, n_iter:int=3):
    """iterative process: (L + D) * x_{k+1} = b - U * x_k, lower omega can be better (relaxation)"""
    x = x0.copy()
    L, D, U = _get_LDU_parts(A)
    for i in range(n_iter):
        temp = U @ x
        xnew = np.linalg.solve(L + D, b - temp)
        x = x + omega * (xnew - x) # relaxed update, omega = 1 is full update
    return x


def gauss_seidel_SOR(A, b, x0, omega:float=1.3, n_iter:int=3):
    """gauss seidel with over relaxation (better error smoothing) :
       (D + omega * L) * x_{k+1} = omega * b - (omega * U + (omega - 1) * D ) * x_k
       where omega in [0,2] usually and no need to exactly invert, can use triangular solve on LHS"""
    # TODO : what values of omega to choose?
    # NOTE : I don't use efficient method here, I'm just demoing whether the solve works in this tool
    # for omega = 1, same as regular Gauss-seidel
    # NOTE : also lexigraphic vs red-black ordering has much different result in conv performance here..
    x = x0.copy()
    L, D, U = _get_LDU_parts(A)
    # plt.imshow(A) # plt.imshow(L)
    # plt.show()
    
    for i in range(n_iter):
        rhs = omega * b - (omega * U + (omega - 1) * D) @ x
        x = np.linalg.solve(D + omega * L, rhs)
    return x

def _get_dof(ix, iy, nx):
    return nx * iy + ix

def coarse_fine_operators(nxe_fine, nxe_factor:int=2, remove_bcs:bool=True):
    """make the maps I_h^H and I_H^h = (I_h^H)^T from coarse to fine and vice versa
    NOTE : it's important the stencil of the interpolation operators are at least as accurate as the PDE 
    otherwise it breaks conv i.e. 2nd order PDE => 2nd order interp, 
    4th order PDE => 4th order interp (see book on this)"""

    nxe_coarse = nxe_fine // nxe_factor
    assert((nxe_coarse * 2) == nxe_fine)
    nxc = nxe_coarse + 1
    nxf = nxe_fine + 1
    N_coarse = nxc**2
    N_fine = nxf**2

    I_h_to_H = np.zeros((N_coarse, N_fine)) # coarse to fine

    # we'll remove bcs from it in a sec.. (some formulations like Neumann don't remove yet..)

    # from eqn 2.3.3. in "multigrid" book
    for i_c in range(N_coarse):
        iyc = i_c // nxc
        ixc = i_c % nxc

        # get nearby fine points?
        ixf = 2 * ixc # center point
        iyf = 2 * iyc
        i_f = _get_dof(ixf, iyf, nxf)

        # print(f"{i_c=} {i_f=}")

        # NOTE : since restriction and interp operators only apply to error (and no error on BCs)
        # it's ok that we miss the partition of unity for error terms.. the interp / restrict would just be 0 there anyways
        I_h_to_H[i_c, i_f] = 4.0 / 16.0 # middle, middle

        if iyf > 0:
            I_h_to_H[i_c, i_f-nxf] = 2.0 / 16.0 # middle, bottom

        if iyf < nxf-1:
            # print(f"{ixc=} {iyc=} {ixf=} {iyf=} {i_c=} {i_f=} {i_f+nxf=} {N_fine=}")
            I_h_to_H[i_c, i_f+nxf] = 2.0 / 16.0 # middle, top

        if ixf > 0:
            I_h_to_H[i_c, i_f-1] = 2.0 / 16.0 # left, middle

            if iyf > 0:
                I_h_to_H[i_c, i_f-1-nxf] = 1.0 / 16.0 # left, bottom

            if iyf < nxf-1:
                I_h_to_H[i_c, i_f-1+nxf] = 1.0 / 16.0 # left, top
        
        if ixf < nxf-1:
            I_h_to_H[i_c, i_f+1] = 2.0 / 16.0 # right, middle

            if iyf > 0:
                I_h_to_H[i_c, i_f+1-nxf] = 1.0 / 16.0 # right, bottom

            if iyf < nxf-1:
                I_h_to_H[i_c, i_f+1+nxf] = 1.0 / 16.0 # right, top

    # now remove bcs from each, here for dirichlet
    if remove_bcs:
        _, coarse_free_dof = get_bcs_and_free_dof(nxc)
        _, fine_free_dof = get_bcs_and_free_dof(nxf)

        I_h_to_H = I_h_to_H[coarse_free_dof,:][:,fine_free_dof]

    I_H_to_h = I_h_to_H.T * 4 # why the *4 though? for 2^d?

    # plt.imshow(I_h_to_H)
    # plt.show()

    # plt.imshow(I_H_to_h)
    # plt.show()

    # short hand notation, i.e. I_cf is coarse to fine
    I_cf = I_H_to_h
    I_fc = I_h_to_H

    return I_cf, I_fc

def damped_jacobi_defect(n_iters, A, _defect):
    defect = _defect.copy()
    dx = np.zeros_like(defect)
    Dinv = 1.0 / np.diag(A)
    omega = 2.0 / 3.0
    for i in range(n_iters):
        dx_update = omega * Dinv * defect

        dx += dx_update
        defect -= np.dot(A, dx_update)
        d_norm = np.linalg.norm(defect)
        print(f"{i=} : {d_norm=:.3e}")
    return 
