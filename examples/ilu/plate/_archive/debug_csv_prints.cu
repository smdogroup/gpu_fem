// debug
    // auto h_loads = loads.createHostVec();
    // printToVTK<Assembler,HostVec<T>>(assembler, h_loads, "plate.vtk");
    // return 1;


    // printout kmat before solve (since cusparse solve changes kmat lower triangular)
    // also printout stiffness matrix
    // bool debug_num_elements = nxe <= 10;
    // if (debug_num_elements) {
    //     auto h_kmat = kmat.createHostVec();
    //     write_to_csv<double>(h_kmat.getPtr(), h_kmat.getSize(), "csv/plate_kmat.csv");
    // }

    // compute total direc derivative of analytic residual

    // DEBUGGING
    // // print some of the data of host residual
    // auto h_soln = soln.createHostVec();
    // // auto h_loads = loads.createHostVec();

    // // printf("took %d microseconds to run add jacobian\n", (int)duration.count());
    // // printf("took %d microseconds to run cusparse solve\n",
    // //        (int)duration2.count());

    // // write the solution to binary file so I can read it in in python
    // // always write this one out regardless of size
    // write_to_csv<double>(h_loads.getPtr(), h_loads.getSize(), "csv/plate_loads.csv");
    // write_to_csv<double>(h_soln.getPtr(), h_soln.getSize(), "csv/plate_soln.csv");
    // auto d_perm = DeviceVec<int32_t>((nxe+1)*(nxe+1), kmat.getBsrData().perm);
    // auto h_perm = d_perm.createHostVec();
    // write_to_csv<int>(h_perm.getPtr(), h_perm.getSize(), "csv/perm.csv");

    // auto bsrData = kmat.getBsrData();
    // DeviceVec<int> d_rowPtr(bsrData.nnodes + 1, bsrData.rowPtr);
    // auto h_rowPtr = d_rowPtr.createHostVec();
    // // printf("h_rowPtr: ");
    // // printVec<int>(bsrData.nnodes, h_rowPtr.getPtr());
    // // printf("\n");
    // DeviceVec<int> d_colPtr(bsrData.nnzb, bsrData.colPtr);
    // auto h_colPtr = d_colPtr.createHostVec();

    // if (debug_num_elements) {
    //     write_to_csv<int>(h_rowPtr.getPtr(), h_rowPtr.getSize(), "csv/plate_rowPtr.csv");
    //     write_to_csv<int>(h_colPtr.getPtr(), h_colPtr.getSize(), "csv/plate_colPtr.csv");
    // }

    // return;