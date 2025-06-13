# Notes on Optimizing the Assemble Residual code

## How to Optimize Register usage

* need scalar addresses, it appears vector declarations like:
T pt[2]
results in allocation in local meomry with global address space (high number of cycles to read memory). So I would need to do something like this:
T xi, eta;
* how do I handle clearly array data like:
T etn[Basis::num_nodes]
Maybe I put this array in shared memory?
And then I can write and read to it?
Want really low number of array usage with local / global memory apparently.

## Issue with shared memory for local matrix-vec computations?

// how to compute shell node normal light, should I put n0 in shared memory arrays? I certainly could

        // // get shell transform and Xdn frame scope
        // T Tmat[9], Xd[9];
        // {
        //     T n0[3];
        //     ShellComputeNodeNormalLight(pt, xpts, n0);

        //     // assemble Xd frame
        //     Basis::assembleFrameLight<3>(pt, xpts, n0, Xd);

        //     // compute the shell transform based on the ref axis in Data object
        //     ShellComputeTransformLight<T, Basis, Data>(refAxis, pt, xpts, n0, Tmat);
        // }  // end of Xd and shell transform scope