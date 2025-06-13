# Profiling Tips with Nsight compute

## Profiling commands
Generate a profile with (and sudo sometimes needed with full path to ncu executable for permissions):
<<<<<<< HEAD
* note you may not always need sudo, can just do `ncu **` rest of commands in that case.
=======
>>>>>>> 42213dc9efa552f5caa9f39073da390944f4589e
```
sudo /usr/local/cuda-12.6/bin/ncu --export myrep.ncu-rep ./4_lin_ucrm.out
```

If you need to view registers at each point of the line, need to give permission to see source code
```
sudo /usr/local/cuda-12.6/bin/ncu --set full     --export add_residual_report     ./4_lin_ucrm.out
```

To view the nsight compute report
```
ncu-ui add_residual_report.ncu-rep 
```