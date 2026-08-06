## Multi CPU and GPU

- [ ] multi CPU and multi GPU to allow >4 GPUs on mat-vec product, precond, etc.
    * does it require CPU-GPU communication to go among >4 GPUs? GPU to CPU then CPU-CPU then CPU to GPU? Is that really slow?
- [ ] OpenMPI for multi CPU? and do multithreading also to decrease host setup time?