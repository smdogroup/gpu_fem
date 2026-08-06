rm *.txt || echo "can't remove *.txt files"
rm *.out || echo "can't remove *.out files"

# export NXE=4
export NXE=8
# export NXE=10

# make 1_single_asw
# echo "run 1_single_asw.out"
# ./1_single_asw.out --nxe ${NXE} >> out1.txt

# make 2_multi_asw
# echo "run 2_multi_asw.out"
# ./2_multi_asw.out ${NXE} >> out2.txt

# make 3_single_gmg_asw
# echo "run 3_single_gmg_asw.out"
# ./3_single_gmg_asw.out --nxe ${NXE} --SR 1e1 >> out3.txt

# make 4_multi_gmg_asw
# echo "run 4_multi_gmg_asw.out"
# ./4_multi_gmg_asw.out ${NXE} >> out4.txt

# export LEVEL=1
# export LEVEL=2
export LEVEL=3

make 5_swing_gmg_asw
echo "run 5_swing_gmg_asw.out"
./5_swing_gmg_asw.out --level ${LEVEL} --SR 300 >> out5.txt

make 6_mwing_gmg_asw
echo "run 6_mwing_gmg_asw.out"
./6_mwing_gmg_asw.out ${LEVEL} >> out6.txt