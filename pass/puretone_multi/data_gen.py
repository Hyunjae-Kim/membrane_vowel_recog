import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

a =time.time()
data_num = 1
ch_list = [5, 3, 1]

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
    for freq in range(650,750,5):
        print(freq)
        for try_num in range(1000):
            sound = np.sin(2*np.pi*(freq*np.linspace(0,time_len/1000,itmax)+np.random.random()))
            sound = sound + 0.1*np.random.random(len(sound))

            u = np.zeros(ch_num)
            v = np.zeros(ch_num)
            dudt = np.zeros(ch_num)
            dvdt = np.zeros(ch_num)

            for it in range(0,itmax):             
                t = it*dt
                dudt = v
                dvdt = -gm*v -(w0**2)*u + amp*sound[it]/mass

                u = u + dudt*dt
                v = v + dvdt*dt

                if it%20==0:  ### 50k sampling rate
                    if it==0:
                        pos_list = u.reshape(1,-1)
                    else:
                        pos_list = np.concatenate((pos_list, u.reshape(1,-1)))

            pos_list = np.transpose(pos_list)

            if try_num==0:
                whole_pos = pos_list.reshape(1,np.shape(pos_list)[0], np.shape(pos_list)[1])
                if ch_num==19: whole_raw = sound[::20].reshape(1,-1)
            else:
                whole_pos = np.concatenate((whole_pos, pos_list.reshape(1,np.shape(pos_list)[0], 
                                                                        np.shape(pos_list)[1])))
                if ch_num==19: whole_raw = np.concatenate((whole_raw, sound[::20].reshape(1,-1)))
                    
            if try_num%10==0: print(try_num/1000)
                
        np.save('pure_data%d/%dch/%d.npy'%(data_num, ch_num, freq), whole_pos)
        if ch_num==19:
            np.save('pure_data%d/raw/%d.npy'%(data_num, freq), whole_raw)

        print('data shape :', np.shape(whole_pos))

