
CUDA_LINK := -lcublas -lcusparse -lcudart -lcusolver

Dcsr: 1_gmres_Dcsr

Sbsr: 2_gmres_Sbsr

Dbsr: 3_gmres_Dbsr

test:
	rm *.txt || echo "no *.txt files"
	make Dcsr
	make Sbsr
	make Dbsr
	./1_gmres_Dcsr.out >> out1.txt
	./2_gmres_Sbsr.out >> out2.txt
	./3_gmres_Dbsr.out >> out3.txt

%: %.cu
	nvcc -Xcompiler "-Wno-deprecated-declarations" $< -o $@.out ${CUDA_LINK}