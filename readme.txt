2tones_couple : 커플링된 heterogeneous osc 시뮬레이션 , 2 pure tones 
2tones_recog_compare : 실제 멤브레인 실험데이터, & MFCC, FFT 비교
2tones_recog_memsim : 멤브레인 시뮬레이션으로 2tones,,,
2tones_recog_mic : heterogeneous osc 시뮬레이션, 2 pure tones

vowel_band_nos_mic : 쏴주는 소리의 band별 TFS정보 없앴을때. TFS대신에 white noise섞어준것. 노이즈환경에서 아에이오우 분류
vowel_nos_mic : 쏴주는 소리는 noise만 섞인 소리, 그거의 출력값으로 얻은 hetero osc들을 각 osc별로 raw데이터를 ENV로 대체했을때 노이즈 환경에서 어떻게되는지
vowel_recog_compare : 실제 멤브레인 실험데이터 & MFCC, FFT 비교 // 아에이오우
vowel_recog_env : 실제 membrane 데이터에서 출력값들의 Raw를 ENV로 대체했을때 어떻게 되는지.
vowel_recog_mic : heterogeneous osc시뮬레이션 & 그 데이터로 band env 확인// 아에이오우