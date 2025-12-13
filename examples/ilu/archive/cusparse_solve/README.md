# Demo & Verify Cusparse solves
* goal here is to demo & verify cusparse solutions
* consider small matrices I make up
* check residual for each after solve on GPU

- [ ] determine if CHOLMOD does fillin well (seemed to get it wrong before)
- [ ] try small 4x4 block matrix with block_dim = 1, creating BsrMat datatype