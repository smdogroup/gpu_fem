# for HPC Milan A100s

mpiexec -n 1 ./0_plate.out --SR 10.0 --elem CFI4 --nxe 16 --nsmooth 1
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem CFI4 --nxe 32 --nsmooth 1
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem CFI4 --nxe 64 --nsmooth 1
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem CFI4 --nxe 128 --nsmooth 1
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem CFI4 --nxe 256 --nsmooth 1
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem CFI4 --nxe 512 --nsmooth 1
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem CFI4 --nxe 1024 --nsmooth 1
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem CFI4 --nxe 2048 --nsmooth 1

mpiexec -n 1 ./0_plate.out --SR 10.0 --elem MITC4 --nxe 16 --nsmooth 1
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem MITC4 --nxe 32 --nsmooth 1
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem MITC4 --nxe 64 --nsmooth 1
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem MITC4 --nxe 128 --nsmooth 1
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem MITC4 --nxe 256 --nsmooth 1
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem MITC4 --nxe 512 --nsmooth 1
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem MITC4 --nxe 1024 --nsmooth 1
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem MITC4 --nxe 2048 --nsmooth 1

mpiexec -n 1 ./0_plate.out direct --SR 10.0 --elem CFI9 --nxe 8 
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem CFI9 --nxe 16 --nsmooth 2
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem CFI9 --nxe 32 --nsmooth 2
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem CFI9 --nxe 64 --nsmooth 2
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem CFI9 --nxe 128 --nsmooth 2
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem CFI9 --nxe 256 --nsmooth 2
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem CFI9 --nxe 512 --nsmooth 2

mpiexec -n 1 ./0_plate.out direct --SR 10.0 --elem MITC9 --nxe 8 
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem MITC9 --nxe 16 --nsmooth 2
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem MITC9 --nxe 32 --nsmooth 2
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem MITC9 --nxe 64 --nsmooth 2
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem MITC9 --nxe 128 --nsmooth 2 #--omega 0.95
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem MITC9 --nxe 256 --nsmooth 2 #--omega 0.95
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem MITC9 --nxe 512 --nsmooth 2 #--omega 0.95

mpiexec -n 1 ./0_plate.out direct --SR 10.0 --elem AIG9 --nxe 8 
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem AIG9 --nxe 16 --nsmooth 2
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem AIG9 --nxe 32 --nsmooth 2
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem AIG9 --nxe 64 --nsmooth 2
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem AIG9 --nxe 128 --nsmooth 2
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem AIG9 --nxe 256 --nsmooth 2
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem AIG9 --nxe 512 --nsmooth 2

mpiexec -n 1 ./0_plate.out direct --SR 10.0 --elem CFI16 --nxe 4
mpiexec -n 1 ./0_plate.out direct --SR 10.0 --elem CFI16 --nxe 8 
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem CFI16 --nxe 16 --nsmooth 2
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem CFI16 --nxe 32 --nsmooth 2
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem CFI16 --nxe 64 --nsmooth 2
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem CFI16 --nxe 128 --nsmooth 2
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem CFI16 --nxe 256 --nsmooth 2

mpiexec -n 1 ./0_plate.out direct --SR 10.0 --elem MITC16 --nxe 4
mpiexec -n 1 ./0_plate.out direct --SR 10.0 --elem MITC16 --nxe 8 
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem MITC16 --nxe 16 --nsmooth 4
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem MITC16 --nxe 32 --nsmooth 4
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem MITC16 --nxe 64 --nsmooth 4
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem MITC16 --nxe 128 --nsmooth 4

mpiexec -n 1 ./0_plate.out direct --SR 10.0 --elem LFI16 --nxe 4
mpiexec -n 1 ./0_plate.out direct --SR 10.0 --elem LFI16 --nxe 8 
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem LFI16 --nxe 16 --nsmooth 4
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem LFI16 --nxe 32 --nsmooth 4
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem LFI16 --nxe 64 --nsmooth 4
mpiexec -n 1 ./0_plate.out --SR 10.0 --elem LFI16 --nxe 128 --nsmooth 4

mpiexec -n 1 ./0_plate.out direct --SR 10.0 --elem HRA4 --nxe 8
mpiexec -n 1 ./0_plate.out direct --SR 10.0 --elem HRA4 --nxe 16 
mpiexec -n 1 ./0_plate.out direct --SR 10.0 --elem HRA4 --nxe 32 
mpiexec -n 1 ./0_plate.out direct --SR 10.0 --elem HRA4 --nxe 64 
mpiexec -n 1 ./0_plate.out direct --SR 10.0 --elem HRA4 --nxe 128
mpiexec -n 1 ./0_plate.out direct --SR 10.0 --elem HRA4 --nxe 256
