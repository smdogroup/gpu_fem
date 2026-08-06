// debug print
auto h_ua = ua.createHostVec();
printf("ua:");
printVec<T>(10, h_ua.getPtr());
auto h_fa = fa.createHostVec();
printf("fa:");
printVec<T>(h_fa.getSize(), h_fa.getPtr());
auto h_fs = fs.createHostVec();
printf("fs:");
printVec<T>(h_fs.getSize(), h_fs.getPtr());
auto h_fs_ext = fs_ext.createHostVec();
printf("h_fs_ext:");
printVec<T>(h_fs_ext.getSize(), h_fs_ext.getPtr());
auto h_us = us.createHostVec();
printf("us:");
printVec<T>(10, h_us.getPtr());