#!/bin/bash

declare -a p_len=("5" "10" "50" "100" "200")
n_pl=5

for (( idx_pl=0; idx_pl<${n_pl}; idx_pl++))
do
python mem_train.py --point_len=${p_len[$idx_pl]}
python raw_train.py --point_len=${p_len[$idx_pl]}
#python fft_train.py --point_len=${p_len[$idx_pl]}
#python mfcc_train.py --point_len=${p_len[$idx_pl]}
done
