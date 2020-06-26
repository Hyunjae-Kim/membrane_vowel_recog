mic array만들어서, pure tone쏴줬을때 recognition. transient할때랑 stationary할때 비교.

pure_data1 : 650 ~ 750 Hz,  5Hz 간격. 
pure_data2 : raw데이터에 대해, resonance 700Hz인 mic의 frequency response를 따르도록 freq별로 amplitude다르게한것.
pure_data3 : 1000~ 1100 Hz에 대해서 1ch_pick해서 single_mic 성능 보기위한것.
pure_data4 : single mic와 raw데이터 processing에 따른 결과 비교하려고. w_raw는 pure_data2랑 같은거. raw_clean은 random 노이즈 없앤것.