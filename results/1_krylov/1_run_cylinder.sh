#!/bin/bash
# first just the 100K DOF
# for NXE in 128
for NXE in 64
do
    for SR in 1e0 3e0 1e1 3e1 1e2 3e2 1e3
    # for SR in 1e1 1e2
    do
        for solver in direct ilu0 ilu1 ilu2 jacobi gsmc chebyshev asw
        do
        echo "nxe = $NXE, SR = $SR"
        mpiexec -n 1 ./1_cylinder.out --solver $solver --nxe $NXE --SR $SR
        # mpiexec -n 1 ./1_cylinder.out --solver direct --nxe $NXE --SR $SR
        # mpiexec -n 1 ./1_cylinder.out --solver ilu0 --nxe $NXE --SR $SR
        # mpiexec -n 1 ./1_cylinder.out --solver ilu1 --nxe $NXE --SR $SR
        # mpiexec -n 1 ./1_cylinder.out --solver ilu2 --nxe $NXE --SR $SR
        # mpiexec -n 1 ./1_cylinder.out --solver jacobi --nxe $NXE --SR $SR
        # mpiexec -n 1 ./1_cylinder.out --solver gsmc --nxe $NXE --SR $SR
        # mpiexec -n 1 ./1_cylinder.out --solver chebyshev --nxe $NXE --SR $SR
        # mpiexec -n 1 ./1_cylinder.out --solver asw --nxe $NXE --SR $SR
        done
    done
done

# then scalability study
# for NXE in 32 128 512 2048
for NXE in 16 32 64 128 256
do
    for SR in 1e1 1e2 1e3
    do
        for solver in direct ilu0 ilu1 ilu2 jacobi gsmc chebyshev asw
        do
        echo "nxe = $NXE, SR = $SR"
        mpiexec -n 1 ./1_cylinder.out --solver $solver --nxe $NXE --SR $SR
        done
    done
done