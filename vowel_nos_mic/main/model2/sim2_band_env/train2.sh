#!/bin/bash

declare -a p_len=("15" "10" "5" "0" "-5" "-10" "-15" "-20")
declare -a p_band=("8" "9" "10")
n_pl=8
n_b=3

for (( idx_b=0; idx_b<${n_b}; idx_b++))
do
for (( idx_pl=0; idx_pl<${n_pl}; idx_pl++))
do
python mic_train.py --SNR=${p_len[$idx_pl]} --n_band=${p_band[$idx_b]}
done
done

