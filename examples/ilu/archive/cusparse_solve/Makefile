include ../../Makefile.in

default: solve-gpu

test-sparse:
	@echo "building sparse utils"
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler -DUSE_CUSPARSE ${SSPARSE_INCLUDE} ${INCLUDE_FLAGS} -std=c++17 sparse_util_test.cu ${LINK}

example1:
	@echo "Building example1 with GPU support"
	${GPU_CXX} ${GPU_CC_FLAGS} -g -DUSE_GPU -Xcompiler -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 1_example.cu ${LINK}

run1: 	
	$(MAKE) clean
	$(MAKE) example1
	./a.out

example2:
	@echo "Building example2 with GPU support"
	${GPU_CXX} ${GPU_CC_FLAGS} -g -DUSE_GPU -Xcompiler -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 2_example.cu ${LINK}
#    ${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler -DUSE_CUSPARSE ${SSPARSE_INCLUDE} ${INCLUDE_FLAGS} -DSUITE_SPARSE -std=c++17 2_example.cu ${CUSPARSE_GPU_LIBS} ${SSPARSE_LINK}

run2: 	
	$(MAKE) clean
	$(MAKE) example2
	./a.out

example3:
	@echo "Building with GPU support"
	${GPU_CXX} ${GPU_CC_FLAGS} -g -DUSE_GPU -Xcompiler -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 3_example.cu ${LINK}
#	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler -DUSE_CUSPARSE ${SSPARSE_INCLUDE} ${INCLUDE_FLAGS} -DSUITE_SPARSE -std=c++17 3_example.cu ${CUSPARSE_GPU_LIBS} ${SSPARSE_LINK}

run3: 	
	$(MAKE) clean
	$(MAKE) example3
	./a.out

plate:
	@echo "Building with GPU support"
	${GPU_CXX} ${GPU_CC_FLAGS} -g -DUSE_GPU -Xcompiler -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 4_plate.cu ${LINK}
#	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler -DUSE_CUSPARSE ${SSPARSE_INCLUDE} ${INCLUDE_FLAGS} -DSUITE_SPARSE -std=c++17 4_plate.cu ${CUSPARSE_GPU_LIBS} ${SSPARSE_LINK}

run-plate: 	
	$(MAKE) clean
	$(MAKE) plate
	./a.out
# ./a.out >> out.txt

clean:
	rm *.out || echo "no .out files"
	rm *.txt || echo "no .txt files"
