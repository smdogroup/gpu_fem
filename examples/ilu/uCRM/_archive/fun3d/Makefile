include ../../Makefile.in

FILES := 1_linear_static 2_nl_static 3_meld 4_coupled

default: 1_linear_static

nonlinear: 2_nl_static

meld: 3_meld

aeroelastic: 4_coupled

clean:
	rm *.out || echo "no *.out files to remove"
	rm *.vtk || echo "no *.vtk files to remove"

%: %.cu
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 $< -o $@.out ${LINK}