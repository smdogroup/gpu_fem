import numpy as np

class ScaleTracker:
    def __init__(self, name:str, ndof, assembly, mem, solve):
        self.name = name
        self.ndof = ndof
        self.assembly = assembly
        self.mem = mem
        self.solve = solve
        
    def get_bigO_scale(self, x, y) -> float:
        logx, logy = np.log(x), np.log(y)
        # slope, intercept = np.polyfit(logx, logy, 1) # overall..
        slope = np.diff(logy)[-1] / np.diff(logx)[-1] # slope near high end..
        return slope

    @property
    def assembly_bigO(self):
        return self.get_bigO_scale(self.ndof, self.assembly)

    @property
    def mem_bigO(self):
        return self.get_bigO_scale(self.ndof, self.mem)

    @property
    def solve_bigO(self):
        return self.get_bigO_scale(self.ndof, self.solve)

    def __str__(self) -> str:
        p1, p2, p3 = self.assembly_bigO, self.mem_bigO, self.solve_bigO
        return self.name + f" : bigO exponents O(N^p) for assembly {p1:.2e}, mem {p2:.2e}, solve {p3:.2e}"


# plate problem big O scalings for direct vs GMG
plate_direct = ScaleTracker(
    name="plate-direct",
    ndof=6 * np.array([32, 64, 128, 256, 512])**2,
    assembly=np.array([2.36e-1, 3.04e-1, 8.07e-1, 5.25e0, 52.6e1]),
    mem=np.array([1.22e1, 6.80e1, 3.65e2, 1.85e3, 8.99e3]),
    solve=np.array([5.99e-2, 1.97e-1, 8.72e-1, 3.19e0, 1.74e1]),
)
print(plate_direct)

# plate multigrid now
plate_gmg = ScaleTracker(
    name="plate-gmg",
    ndof=6 * np.array([32, 64, 128, 256, 512])**2,
    assembly=np.array([4.90e-2, 1.03e-1, 1.81e-1, 5.05e-1, 1.90e0]),
    mem=np.array([3.48e0, 2.25e1, 6.32e1, 2.26e2, 8.74e2]),
    solve=np.array([8.70e-2, 6.39e-2, 6.88e-2, 8.87e-2, 2.38e-1]),
)
print(plate_gmg)

# NACA wingbox problem big O scalings for direct vs GMG
# wingbox direct
wing_direct = ScaleTracker(
    name="wing-direct",
    ndof=np.array([1.67e5, 6.15e5]),
    assembly=np.array([0.3e-1, 1.795e0]),
    mem=np.array([5.64e2, 2.47e3]),
    solve=np.array([1.222e0, 4.981e0]),
)
print(wing_direct)

wing_gmg = ScaleTracker(
    name="wing-gmg",
    ndof=np.array([1.67e5, 6.15e5]),
    assembly=np.array([9.19e-1, 2.75e0]),
    mem=np.array([1.17e2, 3.74e2]),
    solve=np.array([1.04e0, 2.27e0]),
)
print(wing_gmg)
