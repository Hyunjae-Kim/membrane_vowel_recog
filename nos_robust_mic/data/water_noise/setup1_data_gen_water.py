import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

np.random.seed(37)
a =time.time()

ch_num = 19

time_len = 440     ## ms 
dt = 1e-6         
itmax = int((time_len/1000)/dt)
print(itmax)

w0 = np.linspace(200, 1200,ch_num+2)[1:-1]
w0 = 2*np.pi*w0
gm = 200*2*np.pi
mass = 1/w0
print('check :', gm/(2*np.pi), w0/(2*np.pi))

for SNR in [-5,-10,-15,-20]:
    for p_num in range(1,66):
        for v_num in range(1,6):
            print('snr :', SNR, 'p_num :', p_num, 'vowel :', v_num)
            sr, sound_ = wavfile.read('wav_1000k/OS1_%d_%d.wav'%(p_num,v_num))
            sr, noise_ = wavfile.read('water_noise2.wav')
            rand_idx = np.random.randint(len(noise_)-len(sound_)-100)
            noise_ = noise_[rand_idx:rand_idx+len(sound_),0]
            sound = ((2./65535.) * (sound_.astype(np.float32) - 32767) + 1.)
            noise = ((2./65535.) * (noise_.astype(np.float32) - 32767) + 1.)

            Vs = np.sqrt(np.mean(np.square(sound)))
            Vni = Vs/(10**(SNR/20))
            Vn = np.sqrt(np.mean(np.square(noise)))
            noise = noise*(Vni/Vn)
            sound += noise

            ##               plt.plot(sound[:10000])
            ##               plt.show()
            ##
            ##               wavfile.write('%ddB_sound_water.wav'%SNR, sr, sound)

            amp = 1000

            u = np.zeros(ch_num)
            v = np.zeros(ch_num)
            dudt = np.zeros(ch_num)
            dvdt = np.zeros(ch_num)

            time_list = []
            for it in range(0,itmax+1):
                t = it*dt
                dudt = v
                dvdt = -gm*v -(w0**2)*u + amp*sound[it]/mass

                u = u + dudt*dt
                v = v + dvdt*dt


                if it%20==0:
                    time_list.append(t*1000)
                    if it==0:
                        pos_list = u.reshape(1,-1)
                    else:
                        pos_list = np.concatenate((pos_list, u.reshape(1,-1)))

            pos_list = np.transpose(pos_list)

            print(np.shape(pos_list), np.shape(sound[:itmax+1][::20]))
            np.save('mic_water/%ddb_%d_%d.npy'%(SNR, p_num, v_num), pos_list)
            np.save('raw_water/%ddb_%d_%d.npy'%(SNR, p_num,v_num), sound[:itmax+1][::20])


