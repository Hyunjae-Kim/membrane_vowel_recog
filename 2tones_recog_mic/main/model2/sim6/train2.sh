#!/bin/bash

declare -a p_len=("5" "10" "50")
declare -a c_num=("3")
n_pl=3
n_ch=1

for (( idx_pl=0; idx_pl<${n_pl}; idx_pl++))
do
python raw_train.py --point_len=${p_len[$idx_pl]}  
done



for (( idx_pl=0; idx_pl<${n_pl}; idx_pl++))
do
for (( idx_ch=0; idx_ch<${n_ch}; idx_ch++))
do
python mic_train.py --point_len=${p_len[$idx_pl]} --ch_num=${c_num[$idx_ch]}
done
done



