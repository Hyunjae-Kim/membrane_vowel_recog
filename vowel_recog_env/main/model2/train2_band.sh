#!/bin/bash

declare -a p_len=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10")
n_pl=11

for (( idx_pl=0; idx_pl<${n_pl}; idx_pl++))
do
python mem_band_env_train.py --n_band=${p_len[$idx_pl]}
done
