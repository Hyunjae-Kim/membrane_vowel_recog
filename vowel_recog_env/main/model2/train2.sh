#!/bin/bash

declare -a p_len=("200")
n_pl=1

for (( idx_pl=0; idx_pl<${n_pl}; idx_pl++))
do
# python mem_train.py --point_len=${p_len[$idx_pl]}
# python raw_train.py --point_len=${p_len[$idx_pl]}
# python mem_env_train.py --point_len=${p_len[$idx_pl]}
# python raw_env_train.py --point_len=${p_len[$idx_pl]}
python raw_band_env_train.py --point_len=${p_len[$idx_pl]}
done
