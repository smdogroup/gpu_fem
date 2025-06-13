include ../Makefile.in

default: assembly-profile

# the following code uses the -lineinfo to ensure you can see the C++ code with live registers in nsight compute
%: %.cu
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE -lineinfo ${INCLUDE_FLAGS} -std=c++17 $< -o $@.out ${LINK}

# then here we run the full generation of an nsight compute report, you'll then run 'ncu-ui assembly_report.ncu-rep' to profile the kernel and scroll down
# in details, click on source counters in order to view the live registers (switch to live registers in the tab)
time-profile:
	rm -rf time_report.ncu-rep
	make time
	sudo /usr/local/cuda-12.6/bin/ncu --set full --export time_report.ncu-rep  ./time.out

time-jac-profile:
	rm -rf time_jac_report.ncu-rep
	make time_jac
	sudo /usr/local/cuda-12.6/bin/ncu --set full --export time_jac_report.ncu-rep  ./time_jac.out

# %.ptx: %.cu
# 	${GPU_CXX} -ptx --source-in-ptx -g -G ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 $< -o $@ ${LINK} # --maxrregcount=32