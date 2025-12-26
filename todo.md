# Current Devel tasks

## Qordering ILU(0) NASA
- [x] read through qordering ILU(0) libraries in NASA FUN3D (SFE section is where Kyle's code is)
   * do they do pivoting in each block and how do they do pivoting (YES)
   * do they use cusparse on GPU? (NO)
- [ ] code up qorder "iluk_sse" block size 6 pivoting (slow version in C++)
- [ ] demo iluk_sse qorder on wing case (like in Kyle's paper) first CPU parallel
- [ ] write my own pivoted ILU factorization on GPU with level sets (to replace CuSparse no pivot ILU factor), if possible. will have to see if it is possible.


## done tasks
