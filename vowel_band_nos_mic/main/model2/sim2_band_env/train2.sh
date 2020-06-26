#!/bin/bash

declare -a p_len=("10" "5" "0" "-5" "-10" "-15")
declare -a p_band=("2" "6" "10" "14")
n_pl=6
n_b=4

for (( idx_b=0; idx_b<${n_b}; idx_b++))
do
for (( idx_pl=0; idx_pl<${n_pl}; idx_pl++))
do
python mic_train.py --SNR=${p_len[$idx_pl]} --n_band=${p_band[$idx_b]}
done
done

