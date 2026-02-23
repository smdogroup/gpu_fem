// class types for different coarse-fine or prolongation operators
// the restriction is often the transpose or row-normalized transpose for geom multigrid
#pragma once
#include "_structured_iga.cuh"

template <class Assembler>
class StructuredIGAProlongation {
   public:
    using T = double;
    using Basis = typename Assembler::Basis;
    static constexpr int32_t order = Basis::order;
    static constexpr bool structured = true;
    static constexpr bool assembly = false;
    static constexpr bool smoothed = false;

    static constexpr int32_t vars_per_node = Assembler::Phys::vars_per_node;

    StructuredIGAProlongation(Assembler &fine_assembler) {
        // extract some relevant data from the coarse and fine assemblers
        nelems_fine = fine_assembler.get_num_elements();
        nnodes_fine = fine_assembler.get_num_nodes();
        N_fine = fine_assembler.get_num_vars();
        d_fine_iperm = fine_assembler.getBsrData().iperm;
        d_weights = DeviceVec<T>(N_fine).getPtr();
        nxe_fine = sqrt((float)nelems_fine);
    }

    // no update required for structured prolongs
    void update_after_assembly(DeviceVec<T> &vars) {}

    void init_coarse_data(Assembler &coarse_assembler) {
        // has to be called separately (since coarse grid isn't made at same time as fine grid)
        d_coarse_iperm = coarse_assembler.getBsrData().iperm;
        N_coarse = coarse_assembler.get_num_vars();
        nelems_coarse = coarse_assembler.get_num_elements();
        nxe_coarse = sqrt((float)nelems_coarse);
        assert(nxe_fine == (2 * nxe_coarse));
    }

    void prolongate(DeviceVec<T> coarse_soln_in, DeviceVec<T> dx_fine) {
        // zero weights to ensure partition of unity
        cudaMemset(d_weights, 0.0, N_fine * sizeof(T));

        /* launch the struct prolong kernel */
        dim3 block(32);
        int nblocks_x = (nnodes_fine + 31) / 32;
        dim3 grid(nblocks_x);
        k_plate_iga_prolongate<T, Basis><<<grid, block>>>(
            order, vars_per_node, nxe_coarse, nxe_fine, nnodes_fine, d_coarse_iperm, d_fine_iperm,
            coarse_soln_in.getPtr(), dx_fine.getPtr(), d_weights);

        /* ensure partition of unity by weight normalization */
        int nblock2 = (N_fine + 31) / 32;
        dim3 grid2(nblock2);
        k_vec_normalize<T><<<grid2, block>>>(N_fine, dx_fine.getPtr(), d_weights);
    }

    template <bool normalize = false>
    void restrict_vec(DeviceVec<T> fine_vec_in, DeviceVec<T> coarse_vec_out) {
        // zero weights to ensure partition of unity
        if constexpr (normalize) {
            cudaMemset(d_weights, 0.0, N_fine * sizeof(T));
        }

        /* launch struct restrict kernel */
        dim3 block(32);
        int nblocks_x = (nnodes_fine + 31) / 32;
        dim3 grid(nblocks_x);
        k_plate_iga_restrict<T, Basis><<<grid, block>>>(
            order, vars_per_node, nxe_coarse, nxe_fine, nnodes_fine, d_coarse_iperm, d_fine_iperm,
            fine_vec_in.getPtr(), coarse_vec_out.getPtr(), d_weights);

        // now normalize by the weights so partition of unity remains (only for restricting soln in
        // NL problems, not loads))
        if constexpr (normalize) {
            int nblock2 = (N_coarse + 31) / 32;
            dim3 grid2(nblock2);
            k_vec_normalize<T><<<grid2, block>>>(N_coarse, coarse_vec_out.getPtr(), d_weights);
        }
    }

   private:
    int N_coarse, N_fine, nelems_coarse, nelems_fine, nnodes_fine;
    int nxe_coarse, nxe_fine;
    int *d_coarse_iperm, *d_fine_iperm;
    T *d_weights;
};
