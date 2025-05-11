include ../../Makefile.in

FILES := 1_linear_static 2_nl_static

default: 1_linear_static

%: %.cpp
	g++ ${INCLUDE_FLAGS} $< -o $@.out

%: %.cu
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 $< -o $@.out ${LINK}