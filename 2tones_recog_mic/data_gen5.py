import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

a =time.time()
data_num = 5

np.random.seed(37)

ch_num = 19
time_len = 30     ## ms 
dt = 1e-6         
itmax = int((time_len/1000)/dt)
print('iteration :', itmax)

w0 = np.linspace(200, 1200,ch_num+2)[1:-1]
w0 = 2*np.pi*w0
gm = 200*2*np.pi  ## /s
mass = 1/w0
amp = 1000
print('check :', gm/(2*np.pi), w0/(2*np.pi))

count = 0
for freq1 in range(600, 700, 5):
    for freq2 in range(freq1+5, 700, 5):
        if freq1==freq2: print('same'); continue
        f1 = 2*np.pi*freq1
        f2 = 2*np.pi*freq2
        amp_cf1 = (amp*w0)/(np.sqrt( (w0**2-f1**2)**2 + (gm**2)*(f1**2) ))
        amp_cf2 = (amp*w0)/(np.sqrt( (w0**2-f2**2)**2 + (gm**2)*(f2**2) ))
        for try_num in range(500):
            amp1 = np.random.random()*0.2 + 0.8
            amp2 = np.random.random()*0.2 + 0.8
            sound1 = np.tile(amp1*np.sin(2*np.pi*(freq1*np.linspace(0,time_len/1000,itmax)+np.random.random())), (ch_num, 1))
            sound2 = np.tile(amp2*np.sin(2*np.pi*(freq2*np.linspace(0,time_len/1000,itmax)+np.random.random())), (ch_num, 1))
            sound1 = amp_cf1.reshape(-1,1)*sound1
            sound2 = amp_cf2.reshape(-1,1)*sound2
            pos_list = sound1 + sound2
            pos_list = pos_list[:,::20]
            
            if try_num==0:
                whole_pos = pos_list.reshape(1,np.shape(pos_list)[0], np.shape(pos_list)[1])
            else:
                whole_pos = np.concatenate((whole_pos, pos_list.reshape(1,np.shape(pos_list)[0], 
                                                                        np.shape(pos_list)[1])))

            if try_num%50==0: print(try_num/500)

        np.save('data/sim_data%d/mic/%d_%d.npy'%(data_num, freq1, freq2), whole_pos)
        print('%d_%d data shape :'%(freq1, freq2), np.shape(whole_pos))

