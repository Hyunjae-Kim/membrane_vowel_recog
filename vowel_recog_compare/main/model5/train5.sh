#!/bin/bash

declare -a p_len=("1" "2" "3")
n_pl=3

for (( idx_pl=0; idx_pl<${n_pl}; idx_pl++))
do
python mfcc_train.py --trial_num=${p_len[$idx_pl]}
done
