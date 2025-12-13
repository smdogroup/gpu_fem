include ../../../Makefile.in

default: 1_plate

debug:
	rm out.txt || echo "can't remote out.txt"
	rm a.out || echo "can't remote a.out"
	make gpu
	./a.out >> out.txt
%: %.cu
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 $< -o $@.out ${LINK}