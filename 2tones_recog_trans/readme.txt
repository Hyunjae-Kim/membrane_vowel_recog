sim_data2 : 2tones_recog_mic와 동일, transient부분(처음부터 데이터 사용)
    
sim_data2_rep : - sim_data2에서 gamma값을 200-> 5000 (짧은 transient 구간)
                - sound amplitude는 1000 -> 10000
                - raw데이터에서 time_len만큼의 데이터를 2번 반복해서 쏴줌.
                   > 처음 쏴줄때는 data 수집 x ( 초기값을 0이 아닌 time_len만큼 지난 후의 값으로 만들어줌 )
                   > 두번째 쏴줄때 data수집. ( sim_data2와 동일하게 수집 )
                -> 결국 사용한 raw데이터는 time_len만큼의 짧은 window. 이렇게 두번 반복함으로 steady로 고려하게하는
                   효과 있는지 확인. 이걸로 정답률 향상되면, multi dimension&channel 효과.