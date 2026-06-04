rm *.txt || echo "can't remove *.txt files"
rm *.out || echo "can't remove *.out files"

export NX=6
export SD=2

make 1_scyl_struct
echo "run 1_scyl_struct.out"
./1_scyl_struct.out --nxe ${NX} --subdomain ${SD} >> out1.txt

make 2_mcyl_struct
echo "run 2_mcyl_struct.out"
./2_mcyl_struct.out >> out2.txt