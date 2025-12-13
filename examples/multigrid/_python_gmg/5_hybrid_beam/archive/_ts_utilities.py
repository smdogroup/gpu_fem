import numpy as np
from numba import jit, njit

class Material:
    def __init__(self, E, rho, nu, G, k_s, ys):
        self.E = E
        self.rho = rho
        self.nu = nu
        self.G = G
        self.k_s = k_s  # shear correction factor
        self.ys = ys # yield stress
    
    @classmethod
    def aluminum(cls):
        E = 70e9; nu = 0.3; rho = 2.7e3
        G = E / 2.0 / (1+nu)

        # Shear correction factor:
        k_s = 5/6
        ys = 276e6
        #ys = 11e6 #350e6
        return cls(E, rho, nu, G, k_s, ys)

# constitutive
# ------------
def get_CK(E, A, G, J, Iy, Iz, ky=5.0/6.0, kz=5.0/6.0):
    # the C input for get_strain_energy and get_stiffness_matrix
    return np.array([
        E * A,
        G * J,
        E * Iz,
        E * Iy,
        ky * G * A,
        kz * G * A,
    ])

def get_CM(rho, A, Iy, Iz):
    # the CM input for get_kinetic_energy and get_mass_matrix
    return np.array([
        rho * A,
        rho * Iz,
        rho * Iy,
    ])

def get_stress_constitutive(material:Material):
    # get four constitutive values needed for VM stress
    kyG = material.k_s * material.G
    kzG = material.k_s * material.G
    return np.array([material.E, kyG, kzG, material.ys])

def get_constitutive_data(material:Material, t1, t2):
    A = t1 * t2
    # t1 for y and t2 for z in local coords
    Iy = t2**3 * t1 / 12.0 # Iy is for bending in xz plane
    Iz = t1**3 * t2 / 12.0 # Iz is for bending in xy plane plane
    J = Iy + Iz

    CK = get_CK(material.E, A, material.G, J, Iy, Iz)
    CM = get_CM(material.rho, A, Iy, Iz)
    return np.concatenate([CK, CM], axis=0)

# basis
# -----

def basis_fcn(ind, xi):
    if ind == 0:
        return 0.5 * (1 - xi)
    else:
        return 0.5 * (1 + xi)

def norm3(x):
    # more friendly for a complex-step over the length
    return np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)

def get_beam_node_normals(N_BASIS, xpts, axis):
    # get beam normal directions
    # ref axis is basically the second direction

    fn1 = np.zeros((3*N_BASIS,), dtype=np.complex128)
    fn2 = np.zeros((3*N_BASIS,), dtype=np.complex128)

    for ibasis in range(N_BASIS):
        # pt = -1.0 + 2.0 * ibasis # -1 or 1

        # get fields grad X0xi
        X0xi = xpts[0:3] * -0.5 + xpts[3:6] * 0.5
        t1 = X0xi / norm3(X0xi)

        # get second direction
        t2 = axis - np.dot(t1, axis) * t1
        # print(f"pre {axis=} {X0xi=} {xpts=} {t1=} {t2=}")
        t2 /= norm3(t2)

        # get remaining direction
        t3 = np.cross(t1, t2)

        # print(f"{t1=} {t2=}")
        # store node normals
        fn1[3*ibasis:3*(ibasis+1)] = t2[:]
        fn2[3*ibasis:3*(ibasis+1)] = t3[:]
        
    return fn1, fn2

def compute_director(qvars, fn):
    d = np.zeros((6,), dtype=np.complex128)
    # uses rotational DOF here so last 3 of each node
    d[:3] = np.cross(qvars[3:6], fn[:3])
    d[3:] = np.cross(qvars[9:12], fn[3:])
    return d

def compute_tying_strains(xpts, fn1, fn2, qvars, d1, d2):
    # only considers the transverse shear stresses here (2 strains)
    ety = np.zeros((2,), dtype=np.complex128)
    # xi = 1.0 # weird that it's not just 0 then
    # would just interp to the second point Xxi, Uxi etc.
    # double check this?

    # prelim
    Xxi = 0.5 * (xpts[3:] - xpts[:3])
    Udisp_xi = 0.5 * (qvars[6:9] - qvars[0:3]) # only u,v,w considered here not rotations

    # first the g12 strain (linear model)
    xi = 0.0 # which means basis functions are just 0.5 each
    d0 = 0.5 * (d1[:3] + d1[3:])
    n0 = 0.5 * (fn1[:3] + fn1[3:])
    ety[0] = 0.5 * (np.dot(Xxi, d0) + np.dot(n0, Udisp_xi))
    
    # then the g13 strain (linear model)
    d0 = 0.5 * (d2[:3] + d2[3:])
    n0 = 0.5 * (fn2[:3] + fn2[3:])
    ety[1] = 0.5 * (np.dot(Xxi, d0) + np.dot(n0, Udisp_xi))

    return ety