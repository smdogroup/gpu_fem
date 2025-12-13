include ../../../../Makefile.in

CUDA_LINK := -lcublas -lcusparse -lcudart -lcusolver

clean:
	rm *out || echo "no *.out files"
	
clean-out:
	rm out/*png || echo "no *png files to remove"
	rm out/*svg || echo "no *svg files to remove"
	rm out/*vtk || echo "no *vtk files to remove"

%: %.cu
	${GPU_CXX} -w ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 $< -o $@.out ${LINK}