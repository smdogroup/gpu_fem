## Multi-GPU GMG-ASW Demo Tasks

### 1. Partitioning / Domain Decomposition
- [x] Implement structured multi-GPU element-connectivity partitioner
- [x] Assign owned nodes per GPU
- [x] Build owned-node lists per GPU
- [x] Build local node sets per GPU from element connectivity
- [x] Build owned-to-local node maps
- [x] Build ghost/interface node maps
  - [x] Source owned-reduced maps
  - [x] Destination local-placement maps
- [x] Move partition + ghost maps to device
- [ ] Extend partitioner to unstructured meshes later

### 2. Multi-GPU Vector / Matrix Infrastructure
- [x] Implement `GPUvec` from multi-GPU partition
  - [x] Store owned-node data per GPU
  - [x] Allocate local expanded vectors
  - [x] Allocate reduced ghost buffers on source GPUs
  - [x] Allocate reduced ghost buffers on destination GPUs
  - [x] Pack owned → reduced ghost data
  - [x] Peer-copy reduced ghost data
  - [x] Place ghost values into local expanded vectors
- [ ] Implement `GPUmat` from multi-GPU partition
  - [x] Design reduced element connectivity
  - [x] Build local BSR sparsity pattern from element connectivity
  - [x] Allocate local BSR matrix storage
  - [x] Use partition ghost maps through `GPUvec::expandToLocal()`
  - [x] Support matvec using owned rows + local expanded columns
  - [x] Add efficient element-to-BSR-entry maps for GPU assembly
<!-- - [ ] allow matrix permutations.. and perm operations -->

### 3. Multi-GPU Assembly
- [x] Create base multi-GPU assembler class
  - [x] Support different local element connectivity on each GPU
  - [x] Support local element loops and local residual/Jacobian assembly
- [x] Implement MITC shell assembler subclass

### 4. Multi-GPU ASW Smoother
- [x] Implement element-based ASW preconditioner
  - [x] Build ASW patches from owned elements
  - [x] Use ghost values via partition maps in `smoothDefect`
  - [x] Accumulate corrections only to owned nodes
- [x] Validate ASW smoothing against single-GPU behavior

### 5. Multi-GPU GMG Transfer Operators
- [ ] Implement multi-GPU prolongation/restriction
  - [ ] Partition by owned fine-grid rows
  - [ ] Handle ghost-node input/output sizing
  - [ ] Sync ghost values before/after transfer
- [ ] Create multi-GPU GMG prolongation class
  - [ ] Reuse existing templating where possible
  - [ ] Add multi-GPU state and ownership maps

### 6. Coarse-Grid Solve
- [x] Gather coarse problem onto one GPU
- [x] Implement direct coarse solve on single GPU
- [x] Scatter coarse correction back to multi-GPU vectors

### 7. Full Demo / Validation
- [x] Run K-cycle GMG-ASW on cylinder benchmark
- [x] Compare multi-GPU results against single-GPU baseline
- [x] Verify residual histories, displacements, and analysis outputs match
- [x] Measure speedup and scaling across GPUs

### 8. mult-CPU + multi-GPU
- [ ] might be limited to only 4 GPUs if only using CPU serial => so need to code multi-CPU + multi-GPU case next for even higher DOF problems..