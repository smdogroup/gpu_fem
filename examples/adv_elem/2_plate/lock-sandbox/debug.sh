rm out_py.txt
rm out_cu.txt
python 2_debug.py >> out_py.txt
./3_debug.out >> out_cu.txt
# ./3_debug.out --nsmooth_mat 0 >> out_cu.txt