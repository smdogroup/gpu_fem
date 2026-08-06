// dense matrix for debugging
#pragma once
#include "../cuda_utils.h"

// square N x N dense matrix
template <class Vec>
class DenseMat {
   public:
    using T = typename Vec::type;

    __HOST__ DenseMat(int N) : N(N) {
        N2 = N * N;
        data = Vec(N2);
    }

    __HOST_DEVICE__ int getSize() { return N2; }
    __HOST__ void zeroValues() { data.zeroValues(); }
    __HOST_DEVICE__ Vec getVec() { return data; }
    __HOST__ HostVec<T> createHostVec() { return data.createHostVec(); }
    __HOST_DEVICE__ T *getPtr() { return data.getPtr(); }

    __HOST__ void applyBCs(HostVec<int> bcs) {
        int nbcs = bcs.getSize();
        for (int ibc = 0; ibc < nbcs; ibc++) {
            int idof = bcs[ibc];
            for (int i = 0; i < N; i++) {
                data[N * ibc + i] = 0.0;  // sets row to zero
                data[N * i + ibc] = 0.0;  // sets col to zero
            }
            data[N * ibc + ibc] = 1.0;  // set (ibc,ibc) entry to 1.0
            // if non-dirichlet bcs handle later.. in TODO
        }
    }

    __HOST__ void applyBCs(DeviceVec<int> bcs) {
        // TODO : make a kernel for this (see BsrMat)
        // not huge priority though since I don't plan on computing large dense
        // matrices on device anyways DenseMat just for debugging small matrices
        // on host.
    }

    __HOST_DEVICE__ void addElementMatrixValues(const T scale, const int ielem,
                                                const int dof_per_node, const int nodes_per_elem,
                                                const int32_t *elem_conn, const T *elem_mat) {
        // similar method to vec.h or Vec.addElementValues but here for matrix
        int dof_per_elem = dof_per_node * nodes_per_elem;
        for (int idof = 0; idof < dof_per_elem; idof++) {
            int local_inode = idof / dof_per_node;
            int iglobal = elem_conn[local_inode] * dof_per_node + (idof % dof_per_node);

            for (int jdof = 0; jdof < dof_per_elem; jdof++) {
                int local_jnode = jdof / dof_per_node;
                int jglobal = elem_conn[local_jnode] * dof_per_node + (jdof % dof_per_node);
                data[N * iglobal + jglobal] += scale * elem_mat[dof_per_elem * idof + jdof];
            }
        }
    }

#ifdef USE_GPU

    __DEVICE__ void addElementMatrixValuesFromShared(const bool active_thread, const int start,
                                                     const int stride, const T scale,
                                                     const int ielem, const int dof_per_node,
                                                     const int nodes_per_elem,
                                                     const int32_t *elem_conn,
                                                     const T *shared_elem_mat) {
        // copies values to the shared element array on GPU (shared memory)
        if (!active_thread) return;
        int dof_per_elem = dof_per_node * nodes_per_elem;
        for (int idof = start; idof < dof_per_elem; idof += stride) {
            int local_inode = idof / dof_per_node;
            int iglobal = elem_conn[local_inode] * dof_per_node + (idof % dof_per_node);

            for (int jdof = 0; jdof < dof_per_elem; jdof++) {
                int local_jnode = jdof / dof_per_node;
                int jglobal = elem_conn[local_jnode] * dof_per_node + (jdof % dof_per_node);
                atomicAdd(&data[N * iglobal + jglobal],
                          shared_elem_mat[dof_per_elem * idof + jdof]);
            }
        }
    }
#endif  // USE_GPU

    template <typename I>
    __HOST_DEVICE__ T &operator[](const I i) {
        return data[i];
    }
    template <typename I>
    __HOST_DEVICE__ const T &operator[](const I i) const {
        return data[i];
    }

    // __HOST__ void ~DenseMat() { delete data; }
    void free() {
        data.free();
    }

   private:
    int N, N2;
    Vec data;
};

template <class Assembler, class Vec>
__HOST_DEVICE__ DenseMat<Vec> createDenseMat(Assembler &assembler) {
    return DenseMat(assembler.num_nodes * assembler.vars_per_node);
}