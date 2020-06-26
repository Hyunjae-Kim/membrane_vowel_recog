#!/bin/bash

declare -a p_len=("200" "100" "50" "10" "5")
n_pl=5

for (( idx_pl=0; idx_pl<${n_pl}; idx_pl++))
do
python mem_train.py --point_len=${p_len[$idx_pl]}
python raw_train.py --point_len=${p_len[$idx_pl]}
python fft_train.py --point_len=${p_len[$idx_pl]}
python mfcc_train.py --point_len=${p_len[$idx_pl]}
done
