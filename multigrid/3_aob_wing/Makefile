include ../../Makefile.in

PY_EXT = $(shell python3-config --extension-suffix)
CUDA_LINK := -lcublas -lcusparse -lcudart -lcusolver

clean:
	rm *out || echo "no *.out files"
	
clean-out:
	rm out/*png || echo "no *png files to remove"
	rm out/*svg || echo "no *svg files to remove"
	rm out/*vtk || echo "no *vtk files to remove"

# prepares the *.so file for pybind to run optimizations
optim:
	cd _src && ${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -shared -Xcompiler "-fPIC -Wno-deprecated-declarations" -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} ${PYBIND11_INCLUDE} -std=c++17 binding.cu -o gpusolver$(PY_EXT) ${LINK}	


%: %.cu
	${GPU_CXX} -w ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 $< -o $@.out ${LINK}