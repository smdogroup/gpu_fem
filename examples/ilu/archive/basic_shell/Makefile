include ../../Makefile.in

default:
	@if [ "$(USE_GPU)" = "1" ]; then \
	    $(MAKE) gpu; \
	else \
	    $(MAKE) cpu; \
	fi

res-cpu:
	@echo "Building with CPU support"
	${CXX} -x c++ ${CC_FLAGS} ${INCLUDE_FLAGS} -std=c++17 res.cu

res-gpu:
	@echo "Building with GPU support"
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${INCLUDE_FLAGS} -std=c++17 res.cu


jac-cpu:
	@echo "Building with CPU support"
	${CXX} -x c++ ${CC_FLAGS} ${INCLUDE_FLAGS} -std=c++17 jac.cu

fillin:
	@echo "Building with CPU support"
	${CXX} -x c++ ${CC_FLAGS} ${SSPARSE_INCLUDE} ${INCLUDE_FLAGS} -DSUITE_SPARSE -std=c++17 test_fill_in.cu ${SSPARSE_LINK}

jac-gpu:
	@echo "Building with GPU support"
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${INCLUDE_FLAGS} -std=c++17 jac.cu

test-gpu:
	@echo "Building with GPU support"
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${INCLUDE_FLAGS} -std=c++17 test_gpu.cu

solve-gpu:
	@echo "Building with GPU support"
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${INCLUDE_FLAGS} -std=c++17 solve_bsr_gpu.cu ${CUSPARSE_GPU_LIBS}

clean:
	rm *.out || echo "no .out files"
	rm *.txt || echo "no .txt files"

run:
	${MAKE} clean
	${MAKE} 2> make.txt
	./a.out