include ../../../Makefile.in

default: assembly

# the following code uses the -lineinfo to ensure you can see the C++ code with live registers in nsight compute
%: %.cu
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE -Xptxas=-v -lineinfo ${INCLUDE_FLAGS} -std=c++17 $< -o $@.out ${LINK}
	 # --maxrregcount=32

# then here we run the full generation of an nsight compute report, you'll then run 'ncu-ui assembly_report.ncu-rep' to profile the kernel and scroll down
# in details, click on source counters in order to view the live registers (switch to live registers in the tab)
assembly-profile:
	rm -rf assembly_report.ncu-rep
	make assembly
	sudo /usr/local/cuda-12.6/bin/ncu --set full --export assembly_report.ncu-rep     ./assembly.out

math:
	nvcc -O3 --use_fast_math -arch=sm_86 -I/home/seanfireball1/git/_archive/a2d/include -lineinfo -std=c++17 math.cu -o math.out
#  -Xptxas=-v

math-profile:
	rm -rf math.ncu-rep || echo "no math.ncu_rep file to remove"
	make math
	sudo /usr/local/cuda-12.6/bin/ncu --set full --export math.ncu-rep     ./math.out