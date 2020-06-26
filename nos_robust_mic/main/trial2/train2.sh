#!/bin/bash

declare -a p_len=("15" "10" "5" "0" "-5" "-10" "-15" "-20")
declare -a c_len=("1" "2" "3" "5" "7" "9" "12" "15" "19")
n_pl=8
n_ch=9

for (( idx_ch=0; idx_ch<${n_ch}; idx_ch++))
do
for (( idx_pl=0; idx_pl<${n_pl}; idx_pl++))
do
python mic_train.py --SNR=${p_len[$idx_pl]} --ch_num=${c_len[$idx_ch]}
done
done


