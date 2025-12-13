include ../../Makefile.in

default: gpu

debug:
	rm out.txt || echo "can't remote out.txt"
	rm a.out || echo "can't remote a.out"
	make gpu
	./a.out >> out.txt

gpu:
	@echo "Building plate example with GPU support"
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 1_linear_static.cu -o 1_linear_static.out ${LINK}
#   ${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler -DUSE_CUSPARSE ${SSPARSE_INCLUDE} ${INCLUDE_FLAGS} -DSUITE_SPARSE -std=c++17 linear_static.cu ${CUSPARSE_GPU_LIBS} ${SSPARSE_LINK}

assembler:
	@echo "Building plate example with GPU support"
	${CXX} -g -x c++ ${CC_FLAGS} ${INCLUDE_FLAGS} -std=c++17 make_assembler.cpp

reordering:
	${CXX} -g -x c++ ${CC_FLAGS} ${INCLUDE_FLAGS} -std=c++17 reordering.cpp

# gpu-pcg:
# 	@echo "Building plate example for 10x10 mesh with GPU support for PCG solver"
# 	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 linear_static_PCG.cu ${LINK}

one-elem:
	@echo "Building with GPU support"
	${GPU_CXX} -g ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 _one_elem.cu ${LINK}

one-elem-debug:
	rm out.txt || echo "can't remote out.txt"
	rm a.out || echo "can't remote a.out"
	make one-elem
	./a.out >> out.txt

# test16: 
# # tests the kmat cpp vs python for 16 elements (smallest size where you can see profile)
# # if change nxe / num_nodes => run one-elem first to update kelem from dx,dy effects
# 	${MAKE} clean
# 	${MAKE} solve-gpu
# 	rm elems16/cpp-kmat-bsr.csv || echo "can't remove elems16/cpp-kmat-bsr.csv"
# 	echo "ielem,gblock,elem_block,erow,ecol,value" >> elems16/cpp-kmat-bsr.csv || echo "cannot echo into csv file"
# 	./a.out >> elems16/cpp-kmat-bsr.csv
# 	python3 plot_soln.py
# 	python3 _conv_kmat_csv.py

# cpu:
# 	@echo "Building plate example for 10x10 mesh with CPU support"
# 	${CXX} -g ${CC_FLAGS} -DUSE_EIGEN ${INCLUDE_FLAGS} -I/usr/include/eigen3 -std=c++17 linear_static.cpp

# can also make test100 or test4 here

clean:
	rm *.out || echo "no .out files"
	rm *.txt || echo "no .txt files"

# run:
# 	${MAKE} clean
# 	${MAKE} 2> make.txt
# 	./a.out

# solve-gpu-debug:
# 	@echo "Building with GPU support"
# 	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${SSPARSE_INCLUDE} ${INCLUDE_FLAGS} -DSUITE_SPARSE -std=c++17 linear_static_debug.cu ${CUSPARSE_GPU_LIBS} ${SSPARSE_LINK}
# one-elem-dense:
# 	@echo "Building with GPU support"
# 	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${SSPARSE_INCLUDE} ${INCLUDE_FLAGS} -DSUITE_SPARSE -std=c++17 _one_elem_dense.cu ${CUSPARSE_GPU_LIBS} ${SSPARSE_LINK}
