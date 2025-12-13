echo "nprocs, nxe, ndof, nz_time(s), assembly_time(s), solve_time(s), tot_time(s)"
export NPROCS=4
mpirun -n ${NPROCS} python _cpu_baseline.py --nxe 10
mpirun -n ${NPROCS} python _cpu_baseline.py --nxe 20
mpirun -n ${NPROCS} python _cpu_baseline.py --nxe 40
mpirun -n ${NPROCS} python _cpu_baseline.py --nxe 80
mpirun -n ${NPROCS} python _cpu_baseline.py --nxe 160
mpirun -n ${NPROCS} python _cpu_baseline.py --nxe 320
