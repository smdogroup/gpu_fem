include ../../Makefile.in

default: solve-gpu

example1:
	@echo "Building example1 with CPU support"
	${CXX} ${CC_FLAGS} -DUSE_EIGEN ${INCLUDE_FLAGS} -I/usr/include/eigen3 -std=c++17 1_example.cpp
	
run1: 	
	$(MAKE) clean
	$(MAKE) example1
	./a.out

plate2:
	@echo "Building plate example for 2x2 mesh with CPU support"
	${CXX} ${CC_FLAGS} -DUSE_EIGEN ${INCLUDE_FLAGS} -I/usr/include/eigen3 -std=c++17 2_plate2x2.cpp
	
plate10:
	@echo "Building plate example for 10x10 mesh with CPU support"
	${CXX} ${CC_FLAGS} -DUSE_EIGEN ${INCLUDE_FLAGS} -I/usr/include/eigen3 -std=c++17 2_plate2x2.cpp

clean:
	rm *.out || echo "no .out files"
	rm *.txt || echo "no .txt files"
