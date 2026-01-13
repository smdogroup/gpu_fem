__all__ = ["IsotropicMaterial", "PlateFemGeom", "PlateLoads", "plot_init"]

import math, numpy as np
import matplotlib.pyplot as plt

def plot_init():
    plt.rcParams.update({
        # 'font.family': 'Courier New',  # monospace font
        'font.family' : 'monospace', # since Courier new not showing up?
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'figure.titlesize': 20
    }) 

    # also good starting fig size is (10, 10) for example, adjust as needed

class IsotropicMaterial:
    def __init__(self, E:float, nu:float, rho:float, ys:float):
        self.E = E
        self.nu = nu
        self.rho = rho
        self.ys = ys

    @classmethod
    def aluminum(cls):
        # aluminum in metric units
        return cls(E=20e6, nu=0.3, rho=2700, ys=4e5)

class PlateFemGeom:
    def __init__(self, nxe:int, nye:int, nxh:int, nyh:int, a:float, b:float):
        self.nxe = nxe
        self.nye = nye
        self.nxh = nxh
        self.nyh = nyh
        self.a = a
        self.b = b

    @classmethod
    def unit_square(cls, num_elements:int, num_components:int):
        ne = int(num_elements**0.5)
        nc = int(num_components**0.5)
        assert(ne % nc == 0) # num components divides into num_elements
        assert(ne**2 == num_elements) # num_elements is perfect square
        assert(nc**2 == num_components) # num_components is perfect square

        return cls(ne, ne, nc, nc, a=1.0, b=1.0)
    
class PlateLoads:
    def __init__(self, qmag:float, load_fcn):
        self.qmag = qmag
        self.load_fcn = load_fcn

    @classmethod
    def game_of_life_trig_load(cls, qmag=2e-2):
        def load_fcn(x,y):
            theta = math.atan2(y, x)
            r = np.sqrt(x**2 + y**2)
            return 100.0 * np.sin(5.0  * np.pi * r) * np.cos(4*theta)
        return cls(qmag, load_fcn)
    
    @classmethod
    def mode1_load(cls, qmag=2e-2):
        def load_fcn(x,y):
            return 1.0 * np.sin(np.pi * x) * np.sin(np.pi * y)
        return cls(qmag, load_fcn)