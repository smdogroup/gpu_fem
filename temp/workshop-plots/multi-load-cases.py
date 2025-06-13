import numpy as np
import matplotlib.pyplot as plt
import niceplots

plt.style.use(niceplots.get_style())
plt.figure()

load_cases = np.power(10, np.linspace(1.0, 5.0, 10))
#load_cases = np.array([1, 2, 5 ,10, 20, 50, 100, 200, 500, 1000])
cpu_times = np.array([2.5 + 6e-2 * nloads for nloads in load_cases])
gpu_times = np.array([5.5 + 6e-4 * nloads for nloads in load_cases])

plt.plot(load_cases, cpu_times, label="CPU")
plt.plot(load_cases, gpu_times, label="GPU")

plt.xlabel("# load cases")
plt.ylabel("Total solve time (s)")
plt.xscale('log')
plt.yscale('log')

plt.savefig("out/multi-load-cases.png", dpi=400)
