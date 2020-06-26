import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

a =time.time()
data_num = 2
ch_list = [9]

for c in ch_list:
    np.random.seed(37)

    ch_num = c
    time_len = 20     ## ms 
    dt = 1e-6         
    itmax = int((time_len/1000)/dt)
    print(itmax)


    w0 = np.linspace(200, 1200,ch_num+2)[1:-1]
    w0 = 2*np.pi*w0
    gm = 200*2*np.pi  ## /s
    mass = 1/w0
    print('check :', gm/(2*np.pi), w0/(2*np.pi))
    ##print('check :', (gm**2)*(w0**2))

    amp = 1000
    freq_weight = [0.94500197, 0.9580766 , 0.96976758, 0.97984662, 0.98809453,
        0.99431086, 0.99832389, 1.        , 0.99925172, 0.99604344,
        0.99039413, 0.98237689, 0.97211504, 0.95977553, 0.94556009,
        0.92969519, 0.91242172, 0.89398509, 0.87462663, 0.8545765 ]
    count = -1
    for freq in range(650,750,5):
        print(freq)
        count += 1
        amp_w = freq_weight[count]
        for try_num in range(1000):
            sound = np.sin(2*np.pi*(freq*np.linspace(0,time_len/1000,itmax)+np.random.random()))*amp_w
            sound = sound + 0.1*np.random.random(len(sound))

            if try_num==0:
                if ch_num==9: whole_raw = sound[::20].reshape(1,-1)
            else:
                if ch_num==9: whole_raw = np.concatenate((whole_raw, sound[::20].reshape(1,-1)))
                    
            if try_num%10==0: print(try_num/1000)
                
        if ch_num==9:
            np.save('data/pure_data%d/raw/%d.npy'%(data_num, freq), whole_raw)
            print(np.shape(whole_raw))

