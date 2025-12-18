# for HPC Milan A100s

mpiexec -n 1 ./0_plate.out --SR 500 --elem CFI4 --nxe 16 --nsmooth 1
mpiexec -n 1 ./0_plate.out --SR 500 --elem CFI4 --nxe 32 --nsmooth 1
mpiexec -n 1 ./0_plate.out --SR 500 --elem CFI4 --nxe 64 --nsmooth 2 # helps smooth out plot.. looks weird otherwise
mpiexec -n 1 ./0_plate.out --SR 500 --elem CFI4 --nxe 128 --nsmooth 1
mpiexec -n 1 ./0_plate.out --SR 500 --elem CFI4 --nxe 256 --nsmooth 1
mpiexec -n 1 ./0_plate.out --SR 500 --elem CFI4 --nxe 512 --nsmooth 1
mpiexec -n 1 ./0_plate.out --SR 500 --elem CFI4 --nxe 1024 --nsmooth 1
mpiexec -n 1 ./0_plate.out --SR 500 --elem CFI4 --nxe 2048 --nsmooth 1

mpiexec -n 1 ./0_plate.out direct --SR 500 --elem MITC4 --nxe 16 # coarser than coarsest grid (direct solve)
mpiexec -n 1 ./0_plate.out direct --SR 500 --elem MITC4 --nxe 32
mpiexec -n 1 ./0_plate.out --SR 500 --elem MITC4 --nxe 64 --nsmooth 4 
mpiexec -n 1 ./0_plate.out --SR 500 --elem MITC4 --nxe 128 --nsmooth 2
mpiexec -n 1 ./0_plate.out --SR 500 --elem MITC4 --nxe 256 --nsmooth 2
mpiexec -n 1 ./0_plate.out --SR 500 --elem MITC4 --nxe 512 --nsmooth 2
mpiexec -n 1 ./0_plate.out --SR 500 --elem MITC4 --nxe 1024 --nsmooth 2
mpiexec -n 1 ./0_plate.out --SR 500 --elem MITC4 --nxe 2048 --nsmooth 2

# direct solve for first two since coarser than the coarsest grid
mpiexec -n 1 ./0_plate.out direct --SR 500 --elem CFI9 --nxe 8 
mpiexec -n 1 ./0_plate.out direct --SR 500 --elem CFI9 --nxe 16
mpiexec -n 1 ./0_plate.out --SR 500 --elem CFI9 --nxe 32 --nsmooth 4 --omega 0.7
mpiexec -n 1 ./0_plate.out --SR 500 --elem CFI9 --nxe 64 --nsmooth 4 --omega 0.7
mpiexec -n 1 ./0_plate.out --SR 500 --elem CFI9 --nxe 128 --nsmooth 8 --omega 0.7
mpiexec -n 1 ./0_plate.out --SR 500 --elem CFI9 --nxe 256 --nsmooth 16 --omega 0.7
mpiexec -n 1 ./0_plate.out --SR 500 --elem CFI9 --nxe 512 --nsmooth 16 --omega 0.7

mpiexec -n 1 ./0_plate.out direct --SR 500 --elem MITC9 --nxe 8 
mpiexec -n 1 ./0_plate.out direct --SR 500 --elem MITC9 --nxe 16
mpiexec -n 1 ./0_plate.out --SR 500 --elem MITC9 --nxe 32 --nsmooth 4 --omega 0.7
mpiexec -n 1 ./0_plate.out --SR 500 --elem MITC9 --nxe 64 --nsmooth 8 --omega 0.7
mpiexec -n 1 ./0_plate.out --SR 500 --elem MITC9 --nxe 128 --nsmooth 8 --omega 0.7
mpiexec -n 1 ./0_plate.out --SR 500 --elem MITC9 --nxe 256 --nsmooth 8 --omega 0.7
mpiexec -n 1 ./0_plate.out --SR 500 --elem MITC9 --nxe 512 --nsmooth 8 --omega 0.7

mpiexec -n 1 ./0_plate.out direct --SR 500 --elem AIG9 --nxe 8 
mpiexec -n 1 ./0_plate.out direct --SR 500 --elem AIG9 --nxe 16
mpiexec -n 1 ./0_plate.out --SR 500 --elem AIG9 --nxe 32 --nsmooth 2
mpiexec -n 1 ./0_plate.out --SR 500 --elem AIG9 --nxe 64 --nsmooth 2
mpiexec -n 1 ./0_plate.out --SR 500 --elem AIG9 --nxe 128 --nsmooth 2
mpiexec -n 1 ./0_plate.out --SR 500 --elem AIG9 --nxe 256 --nsmooth 2
mpiexec -n 1 ./0_plate.out --SR 500 --elem AIG9 --nxe 512 --nsmooth 2

mpiexec -n 1 ./0_plate.out direct --SR 500 --elem CFI16 --nxe 4
mpiexec -n 1 ./0_plate.out direct --SR 500 --elem CFI16 --nxe 8 
mpiexec -n 1 ./0_plate.out direct --SR 500 --elem CFI16 --nxe 16
mpiexec -n 1 ./0_plate.out direct --SR 500 --elem CFI16 --nxe 32
mpiexec -n 1 ./0_plate.out direct --SR 500 --elem CFI16 --nxe 64
mpiexec -n 1 ./0_plate.out direct --SR 500 --elem CFI16 --nxe 128

mpiexec -n 1 ./0_plate.out direct --SR 500 --elem HRA4 --nxe 16 
mpiexec -n 1 ./0_plate.out direct --SR 500 --elem HRA4 --nxe 32 
mpiexec -n 1 ./0_plate.out direct --SR 500 --elem HRA4 --nxe 64 
mpiexec -n 1 ./0_plate.out direct --SR 500 --elem HRA4 --nxe 128
mpiexec -n 1 ./0_plate.out direct --SR 500 --elem HRA4 --nxe 256

# fix these if possible to multigrid (in order to be consistent)
mpiexec -n 1 ./0_plate.out direct --SR 500 --elem MITC16 --nxe 4
mpiexec -n 1 ./0_plate.out direct --SR 500 --elem MITC16 --nxe 8 
mpiexec -n 1 ./0_plate.out direct --SR 500 --elem MITC16 --nxe 16
mpiexec -n 1 ./0_plate.out direct --SR 500 --elem MITC16 --nxe 32
mpiexec -n 1 ./0_plate.out direct --SR 500 --elem MITC16 --nxe 64
mpiexec -n 1 ./0_plate.out direct --SR 500 --elem MITC16 --nxe 128

mpiexec -n 1 ./0_plate.out direct --SR 500 --elem LFI16 --nxe 4
mpiexec -n 1 ./0_plate.out direct --SR 500 --elem LFI16 --nxe 8 
mpiexec -n 1 ./0_plate.out direct --SR 500 --elem LFI16 --nxe 16
mpiexec -n 1 ./0_plate.out direct --SR 500 --elem LFI16 --nxe 32
mpiexec -n 1 ./0_plate.out direct --SR 500 --elem LFI16 --nxe 64
mpiexec -n 1 ./0_plate.out direct --SR 500 --elem LFI16 --nxe 128