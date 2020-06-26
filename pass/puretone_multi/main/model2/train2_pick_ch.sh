#!/bin/bash

declare -a p_len=("5" "10" "50" "100" "200")
declare -a c_num=("1" "3" "5" "7" "9")
n_pl=5
n_ch=5

for (( idx_pl=0; idx_pl<${n_pl}; idx_pl++))
do
for (( idx_ch=0; idx_ch<${n_ch}; idx_ch++))
do
python pure_train_1ch_pick.py --pick_ch=${c_num[$idx_ch]} --point_len=${p_len[$idx_pl]}
done
done