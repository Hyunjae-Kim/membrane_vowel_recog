AEIOU recognition (stationary data) for 0.5, 1, 5, 10, 20 ms 

model1 - CNN  7 layers, 2 FC layers,  BN , LeakyReLU

model2 - CNN 5 layers, 2 FC layers, BN, LeakyReLU   ### main model

model3 - FCN 5 layers, BN, LeakyReLU

model4 - CNN 3 layers, 2 FC layers, BN, LeakyReLU

model5 - for MFCC comparison  using model4 structure with mfcc trial data




data - raw data. upsampliing한것, mic array or membrane simulation에 사용할 수 있음.