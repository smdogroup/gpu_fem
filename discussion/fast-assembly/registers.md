# How to Optimize Register usage

* need scalar addresses, it appears vector declarations like:
T pt[2]
results in allocation in local meomry with global address space (high number of cycles to read memory). So I would need to do something like this:
T xi, eta;
* how do I handle clearly array data like:
T etn[Basis::num_nodes]
Maybe I put this array in shared memory?
And then I can write and read to it?
Want really low number of array usage with local / global memory apparently.

## Motivation for looking at register usage very carefully
* commenting out methods that do almost nothing but use a global memory array result in huge increase by 104 registers
    * example 1: for drill strain kernel: with method ShellComputeDrillStrainSensV3, if I comment this out it results in about 100 fewer registers and 8x speedup approx in the code
    * example 2 : for tying strain kernel: if I comment out the method addInterpTyingStrainTransposeLight, then I reduce registers by 104 and runtime speeds up by 8x. Don't really know why
    * still I don't see huge speedup I would expect versus CPU probably because I'm using lots more local device memory than I want (which stored globally, this is weird why it's called that). I think this is because I use a bunch of arrays that become global memory and are slow to access. I could make some of these shared memory.. and then make pointers to them for local operations (temporary), but still best to rely on registers as much as possible I think..
* I thought I was using all shared and registers, but that is not the case, generating ptx files helped me realize I need to rethink the way the code is written a lot to speed it up more.
* I probably want to speedup my apply_bcs vec and mat too if possible..

## TODO : best practices
* use nsight compute with the ptx file side by side so you can see registers accumulating at certain lines

## Speeding up drill strain kernel
* tried adding shared memory array: this didn't work well if you need to set the values in multiple times.. it spills into shared memory very easily and is very buggy..
* maybe I won't use shared memory very much then, will still use local arrays.. but try to minimize this as much as possible, can use pointers to the local array if you want to rename it..
* thus switching back to local memory array.