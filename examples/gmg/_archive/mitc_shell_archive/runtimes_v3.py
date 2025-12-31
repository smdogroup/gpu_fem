import numpy as np
import matplotlib.pyplot as plt


class RuntimeGroup:
    def __init__(self, gpu:str, case:str, solver:str, SR:float, nxe:list, times:list):
        assert gpu in ['3060Ti', 'A100']
        assert case in ['plate', 'cylinder']
        assert solver in ['direct', 'gmg']

        self.gpu = gpu
        self.case = case
        self.solver = solver
        self.SR = SR
        self.nxe = nxe
        self.times = times

        self.ndof = 6 * np.array(self.nxe)**2

# nelems = np.array([32, 64, 128, 256, 512, 1024, 2048])
# ndof = nelems**2

runtime_groups = []

# reported here are just solve times (not startup / assembly, though assembly + startup slower for direct solves)
# runtime_groups += [
#     RuntimeGroup(
#         gpu='3060Ti',
#         case='plate',
#         solver='direct',
#         SR=50.0,
#         ndof=[]
#         times=[5.10e-2, 1.82e-1, 7.95e-1, 3.33e0, 1.75e1] + [np.nan] * 2,
#     )
# ]

runtime_groups += [
    RuntimeGroup(
        gpu='A100',
        case='plate',
        solver='direct',
        SR=50.0,
        nxe=[32, 64, 128, 256, 512],
        times=[5.10e-2, 1.82e-1, 7.95e-1, 3.33e0, 1.75e1],
    )
]

# first with double_smooth = true on plate
runtime_groups += [
    RuntimeGroup(
        gpu='A100',
        case='plate',
        solver='gmg',
        SR=50.0,
        nxe=[256, 512, 1024, 2048],
        # times=[1.79e-1, 3.75e-1, 1.45e0, 5.39e0] # before color speedup
        times=[1.38e-1, 2.31e-1, 9.23e-1, 3.39e0] # with color speedup and double_smooth = true
        # times=[1.17e-1, 2.71e-1, 9.94e-1, 4.04e0] # with color speedup and double_smooth = false
    )
]

runtime_groups += [
    RuntimeGroup(
        gpu='A100',
        case='cylinder',
        solver='direct',
        SR=50.0,
        nxe=[128, 256, 512],
        times=[6.35e-1, 2.655e0, 1.2489e1],
    )
]

runtime_groups += [
    RuntimeGroup(
        gpu='A100',
        case='cylinder',
        solver='gmg',
        SR=50.0,
        nxe=[256, 512, 1024, 2048],
        # times=[5.74e-1, 1.44e0, 4.82e0, 1.82e1], # before color speedup
        times=[3.99e-1, 1.06e0, 3.46e0, 1.36e1]
    )
]

# compute speedups
plate_direct_512 = runtime_groups[0].times[-1]
plate_gmg_512 = runtime_groups[1].times[1]
plate_speedup = plate_direct_512 / plate_gmg_512
print(f"plate 512 GMG/direct speedup = {plate_speedup:.4e}")

cyl_direct_512 = runtime_groups[2].times[-1]
cyl_gmg_512 = runtime_groups[3].times[1]
cyl_speedup = cyl_direct_512 / cyl_gmg_512
print(f"cylinder 512 GMG/direct speedup = {cyl_speedup:.4e}")

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

colors = ["#264653", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"]

fig, ax = plt.subplots(1, 2, figsize=(15, 8))

for i, geom in enumerate(['plate', 'cylinder']):
    ct = -1

    for group in runtime_groups:
        if group.case == geom:
            ct += 1
            ax[i].plot(group.ndof, group.times, 'o-', color=colors[ct], label=group.gpu + "-" + group.solver)
    ax[i].set_title(geom)

    ax[i].set_xscale('log')
    ax[i].set_yscale('log')
    ax[i].set_xlabel("NDOF")
    ax[i].set_ylabel("Time (sec)")
    ax[i].legend()

plt.tight_layout()
# plt.show()
plt.savefig("plate_cyl_gmg_direct.svg")