import numpy as np
from numpy.fft import rfft, rfftfreq
from numpy.random import normal
import matplotlib.pyplot as plt

PJ_freq = 10e6 # 1MHz
PJ_amp = 10 # 10ps
RJ_rms = 1 # 1ps
sample_rate = 1e9 #1Gbps
sample_interval = 1./sample_rate
N_cycle =1000
pts = sample_rate/PJ_freq*N_cycle


t = sample_interval*np.arange(pts)*1E6

# sin jitter
plt.figure()
tie_sin = PJ_amp*np.sin(np.linspace(0,2*np.pi*N_cycle,sample_rate/PJ_freq*N_cycle))
plt.subplot(311)
plt.plot(t,tie_sin)
plt.xlabel("T(us)")
plt.ylabel("Time Trend(ps)")
plt.xlim([0,1]) # only plot 1us
plt.subplot(312)
plt.hist(tie_sin,bins=1000,normed=False)
plt.xlabel("Jitter(ps)")
plt.ylabel("Hist(population)")
plt.xlim([-10.2,10.2])
plt.subplot(313)
plt.plot(rfftfreq(len(tie_sin),1/sample_rate)/1e6,np.abs(rfft(tie_sin)))
plt.xlabel("Freq(MHz)")
plt.ylabel("Spectrum(ps)")
plt.xlim([0,50])
plt.tight_layout()

### random jitter
plt.figure()
tie_normal = normal(loc=0,scale=RJ_rms,size=pts)
plt.subplot(311)
plt.plot(t,tie_normal)
plt.xlabel("T(us)")
plt.ylabel("Time Trend(ps)")
plt.xlim([0,1]) # only plot 1us
plt.subplot(312)
plt.hist(tie_normal,bins=1000,normed=False)
plt.xlabel("Jitter(ps)")
plt.ylabel("Hist(population)")
plt.subplot(313)
plt.plot(rfftfreq(len(tie_normal),1/sample_rate)/1e6,20*np.log10(np.abs(rfft(tie_normal))))
plt.xlabel("Freq(MHz)")
plt.ylabel("Spectrum(dB)")
plt.tight_layout()

#Combined jitter
plt.figure()
tie_combine = tie_sin+tie_normal
plt.subplot(311)
plt.plot(t,tie_combine)
plt.xlabel("T(us)")
plt.ylabel("Time Trend(ps)")
plt.xlim([0,1]) # only plot 1us
plt.subplot(312)
plt.hist(tie_combine,bins=1000,normed=False)
plt.xlabel("Jitter(ps)")
plt.ylabel("Hist(population)")
plt.subplot(313)
plt.plot(rfftfreq(len(tie_combine),1/sample_rate)/1e6,20*np.log10(np.abs(rfft(tie_combine))))
plt.xlabel("Freq(MHz)")
plt.ylabel("Spectrum(dB)")
plt.xlim([0,50])
plt.tight_layout()


plt.show()
