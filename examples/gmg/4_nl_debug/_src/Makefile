include ../../../../Makefile.in

PY_EXT = $(shell python3-config --extension-suffix)

default: plate

# prepares the *.so file for pybind to run python nonlinear plate MG solve
plate:
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -shared -Xcompiler "-fPIC -Wno-deprecated-declarations" -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE \
	${INCLUDE_FLAGS} ${PYBIND11_INCLUDE} -std=c++17 _plate_bind.cu -o nlplategpu$(PY_EXT) ${LINK}	

wing:
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -shared -Xcompiler "-fPIC -Wno-deprecated-declarations" -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE \
	${INCLUDE_FLAGS} ${PYBIND11_INCLUDE} -std=c++17 _wing_bind.cu -o nlwinggpu$(PY_EXT) ${LINK}	

%: %.cu
	${GPU_CXX} ${GPU_CC_FLAGS} -DUSE_GPU -Xcompiler ${CC_FLAGS} -DUSE_CUSPARSE ${INCLUDE_FLAGS} -std=c++17 $< -o $@.out ${LINK}