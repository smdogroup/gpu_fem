
CUDA_LINK := -lcublas -lcusparse -lcudart -lcusolver

%: %.cu
	nvcc $< -o $@.out ${CUDA_LINK}