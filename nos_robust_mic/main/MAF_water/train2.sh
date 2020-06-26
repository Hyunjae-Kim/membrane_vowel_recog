#!/bin/bash

declare -a p_len=("15" "10" "5" "0" "-5" "-10" "-15" "-20")
n_pl=8

for (( idx_pl=0; idx_pl<${n_pl}; idx_pl++))
do
python raw_train.py --SNR=${p_len[$idx_pl]}   
done


