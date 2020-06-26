#!/bin/bash

declare -a p_len=("5" "10" "50" "100" "200")
declare -a c_num=("1")
n_pl=5
n_ch=1

for (( idx_pl=0; idx_pl<${n_pl}; idx_pl++))
do
for (( idx_ch=0; idx_ch<${n_ch}; idx_ch++))
do
python pure_train.py --point_len=${p_len[$idx_pl]} --ch_num=${c_num[$idx_ch]}
done
done


for (( idx_pl=0; idx_pl<${n_pl}; idx_pl++))
do
python pure_train_raw.py --point_len=${p_len[$idx_pl]}   
done

for (( idx_pl=0; idx_pl<${n_pl}; idx_pl++))
do
python pure_train_w_raw.py --point_len=${p_len[$idx_pl]}   
done

for (( idx_pl=0; idx_pl<${n_pl}; idx_pl++))
do
python pure_train_raw_clean.py --point_len=${p_len[$idx_pl]}   
done

for (( idx_pl=0; idx_pl<${n_pl}; idx_pl++))
do
python pure_train_w_raw_clean.py --point_len=${p_len[$idx_pl]}   
done