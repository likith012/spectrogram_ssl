import torch
import numpy as np
import os
from .features import get_features
from scipy.signal import butter,lfilter
from scipy.interpolate import interp1d

def denoise_channel(ts, bandpass, signal_freq, bound):
    """
    bandpass: (low, high)
    """
    nyquist_freq = 0.5 * signal_freq
    filter_order = 1
    
    low = bandpass[0] / nyquist_freq
    high = bandpass[1] / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    ts_out = lfilter(b, a, ts)

    ts_out[ts_out > bound] = bound
    ts_out[ts_out < -bound] = - bound

    return np.array(ts_out)

def noise_channel(ts, mode, degree, bound):
    """
    Add noise to ts
    
    mode: high, low, both
    degree: degree of noise, compared with range of ts    
    
    Input:
        ts: (n_length)
    Output:
        out_ts: (n_length)
        
    """
    len_ts = len(ts)
    num_range = np.ptp(ts)+1e-4 # add a small number for flat signal
    
    ### high frequency noise
    if mode == 'high':
        noise = degree * num_range * (2*np.random.rand(len_ts)-1)
        out_ts = ts + noise
        
    ### low frequency noise
    elif mode == 'low':
        noise = degree * num_range * (2*np.random.rand(len_ts//100)-1)
        x_old = np.linspace(0, 1, num=len_ts//100, endpoint=True)
        x_new = np.linspace(0, 1, num=len_ts, endpoint=True)
        f = interp1d(x_old, noise, kind='linear')
        noise = f(x_new)
        out_ts = ts + noise
        
    ### both high frequency noise and low frequency noise
    elif mode == 'both':
        noise1 = degree * num_range * (2*np.random.rand(len_ts)-1)
        noise2 = degree * num_range * (2*np.random.rand(len_ts//100)-1)
        x_old = np.linspace(0, 1, num=len_ts//100, endpoint=True)
        x_new = np.linspace(0, 1, num=len_ts, endpoint=True)
        f = interp1d(x_old, noise2, kind='linear')
        noise2 = f(x_new)
        out_ts = ts + noise1 + noise2

    else:
        out_ts = ts

    out_ts[out_ts > bound] = bound
    out_ts[out_ts < -bound] = - bound
        
    return out_ts

class SLEEPCALoader(torch.utils.data.Dataset):

    def __init__(self, list_IDs, config,dir, SS=True):

        self.list_IDs = list_IDs
        self.dir = dir
        self.SS = SS

        self.label_list = ['W', 'R', 1, 2, 3]
        self.bandpass1 = (1, 5)
        self.bandpass2 = (30, 49)
        self.n_length = 100 * 30
        self.n_channels = 1
        self.n_classes = 5
        self.signal_freq = 100
        self.bound = 0.00025
        self.config = config
        self.features = []
        #if self.SS == False:
        #    for idx,f_name in enumerate(self.list_IDs):
        #        self.features.append(get_features(np.load(f_name)['X'],self.config))
    def __len__(self):
        return len(self.list_IDs)

    def add_noise(self, x, ratio):
        """
        Add noise to multiple ts
        Input: 
            x: (n_channel, n_length)
        Output: 
            x: (n_channel, n_length)
        """
        for i in range(self.n_channels):
            if np.random.rand() > ratio:
                mode = np.random.choice(['high', 'low', 'both', 'no'])
                x[i,:] = noise_channel(x[i,:], mode=mode, degree=0.05, bound=self.bound)
        return x
    
    def remove_noise(self, x, ratio):
        """
        Remove noise from multiple ts
        Input: 
            x: (n_channel, n_length)
        Output: 
            x: (n_channel, n_length)

        Three bandpass filtering done independently to each channel
        sig1 + sig2
        sig1
        sig2
        """
        for i in range(self.n_channels):
            rand = np.random.rand()
            if rand > 0.75:
                x[i, :] = denoise_channel(x[i, :], self.bandpass1, self.signal_freq, bound=self.bound) +                        denoise_channel(x[i, :], self.bandpass2, self.signal_freq, bound=self.bound)
            elif rand > 0.5:
                x[i, :] = denoise_channel(x[i, :], self.bandpass1, self.signal_freq, bound=self.bound)
            elif rand > 0.25:
                x[i, :] = denoise_channel(x[i, :], self.bandpass2, self.signal_freq, bound=self.bound)
            else:
                pass
        return x
    
    def crop(self, x):
        l = np.random.randint(1, self.n_length - 1)
        x[:, :l], x[:, l:] = x[:, -l:], x[:, :-l]

        return x
    
    def augment(self, x):
        t = np.random.rand()
        if t > 0.65:
            x = self.add_noise(x, ratio=0.5)
        elif t > 0.33:
            x = self.remove_noise(x, ratio=0.5)
        else :
            x = self.crop(x)
        return x
    def __getitem__(self, index):
        if self.SS:
            path = os.path.join(self.dir , self.list_IDs[index])
        else:
            path = self.list_IDs[index]
        sample = np.load(path)
        X, y = sample['X'], sample['y'] # (2, 3000), (1,)
       
        if self.SS:
            aug1 = self.augment(X.copy())
            aug2 = self.augment(X.copy())
            return torch.from_numpy(aug1), torch.from_numpy(aug2),get_features(aug1,self.config),get_features(aug2,self.config)
        else:
            return torch.from_numpy(X),get_features(X,self.config),y


def data_generator(data_path,list_IDs,configs):

    train_dataset = SLEEPCALoader(list_IDs,configs,data_path)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=6,pin_memory=True,persistent_workers=True)

    return train_loader


def cross_data_generator(data_path,train_list_IDs,val_list_IDs,configs):

    train_ds = SLEEPCALoader(train_list_IDs,configs,data_path+"/train",SS=False)
    val_ds = SLEEPCALoader(val_list_IDs,configs,data_path+"/test",SS=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=6,pin_memory=True,persistent_workers=True)

    valid_loader = torch.utils.data.DataLoader(dataset=val_ds, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=configs.drop_last,
                                               num_workers=6,pin_memory=True,persistent_workers=True)

    return train_loader,valid_loader
