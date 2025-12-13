include ../../../Makefile.in

PY_EXT = $(shell python3-config --extension-suffix)

default: partial_derivs

# prepares the *.so file for pybind to run optimizations
optim:
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -shared -Xcompiler "-fPIC -Wno-deprecated-declarations" -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} ${PYBIND11_INCLUDE} -std=c++17 binding.cu -o gpusolver$(PY_EXT) ${LINK}	


%: %.cu
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 $< -o $@.out ${LINK}