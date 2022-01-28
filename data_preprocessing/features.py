import math
import numpy as np
from scipy.signal import butter,lfilter
import warnings
import torch
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def butter_bandpass(order=1):
    fs = 100
    freq0 = [0.5,49.5]
    freq1 = [0.5,4]
    freq2 = [4,8]
    freq3 = [7,13]
    freq4 = [11,16]
    freq5 = [16,49.5]
    freq = [freq0,freq1,freq2,freq3,freq4,freq5]
    bas = []
    for (lowcut,highcut) in freq:
        nyq=0.5*fs
        low = lowcut/nyq
        high = highcut/nyq
        b,a = butter(order,[low,high],btype='band')
        bas.append([b,a])
    return bas 

def compute_DE(signal):
    variance = np.var(signal,ddof=1)
    return math.log(2*math.pi*math.e*variance)/2

def compute_energy(signal):
    return sum(signal**2)/len(signal)

def get_features(x,config,proc=None,return_dict=None):
    bas = butter_bandpass()
    fin = []
    step = config.nperseg-config.noverlap
    for (b,a) in bas:
        i=0
        y = lfilter(b,a,x)
        fin_y = []
        y = y[0]
        while i<len(y):
            fin_y.append(compute_DE(y[i:i+config.nperseg]))
            i+=step
        fin.append(fin_y)
    return torch.from_numpy(np.array(fin))
