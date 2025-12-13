include ../../../Makefile.in

PY_EXT = $(shell python3-config --extension-suffix)

default: 1_static

# prepares the *.so file for pybind to run optimizations
optim:
	cd _src && ${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -shared -Xcompiler "-fPIC -Wno-deprecated-declarations" -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} ${PYBIND11_INCLUDE} -std=c++17 binding.cu -o gpusolver$(PY_EXT) ${LINK}	

clean:
	rm *.out || echo "no *.out files to remove"
	rm *.vtk || echo "no *.vtk files to remove"

%: %.cu
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 $< -o $@.out ${LINK}