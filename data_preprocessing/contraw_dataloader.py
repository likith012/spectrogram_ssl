#%%
import torch
import numpy as np
import os
from scipy.signal import butter, lfilter
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
        
    return out_ts
    
def standardize(x):
    
    if not isinstance(x, np.ndarray):
        x = x.numpy()

    for ch in range(x.shape[0]):
        x[ch, :] = (x[ch, :] - np.mean(x[ch, :])) / np.std(x[ch, :])
    return x

    
#############################################################

# Data loader
class SLEEPCALoader(torch.utils.data.Dataset):

    def __init__(self, path, SS=True):

        dataset = torch.load(path)
        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]

        self.SS = SS

        self.bandpass1 = (1, 5)
        self.bandpass2 = (30, 49)
        self.n_length = 100 * 30
        self.n_channels = 2
        self.n_classes = 5
        self.signal_freq = 100
        self.bound = 0.00025

    def __len__(self):
        return self.len

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
                x[i, :] = denoise_channel(x[i, :], self.bandpass1, self.signal_freq, bound=self.bound) +   denoise_channel(x[i, :], self.bandpass2, self.signal_freq, bound=self.bound)
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
        if t > 0.75:
            x = self.add_noise(x, ratio=0.5)
        elif t > 0.5:
            x = self.remove_noise(x, ratio=0.5)
        elif t > 0.25:
            x = self.crop(x)
        else:
            x = x[[1,0],:] # channel flipping
        return x

   
    def __getitem__(self, index):
       
        X, y = self.x_data[index], self.y_data[index]

        if self.SS:
            aug1 = self.augment(X.numpy().copy())
            aug2 = self.augment(X.numpy().copy())
            return aug1, aug2
        else:
            return X.float(), y



def data_generator(data_path, configs):

    #train_dataset = torch.load(os.path.join(data_path, "pretext.pt"))
    #train_dataset = Load_Dataset(train_dataset, configs)
    train_dataset = SLEEPCALoader(os.path.join(data_path,"pretext.pt"))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=10,pin_memory=True,persistent_workers=True)

    return train_loader

def ft_data_generator(data_path,configs):
    train_ds = torch.load(os.path.join(data_path, "train.pt"))
    valid_ds = torch.load(os.path.join(data_path, "val.pt"))
    
    #selected_train = torch.randperm(train_ds['samples'].shape[0],generator=torch.manual_seed(123))
    #train_ds['samples'] = train_ds['samples'][selected_train]
    #train_ds['labels'] = train_ds['labels'][selected_train]
    #n = int(train_ds['samples'].shape[0])//100
    #train_ds['samples'] = train_ds['samples'][:n]
    #train_ds['labels'] = train_ds['labels'][:n]

    train_ds = SLEEPCALoader(os.path.join(data_path, "train.pt"), SS=False)
    valid_ds = SLEEPCALoader(os.path.join(data_path, "val.pt"), SS=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=10,pin_memory=True,persistent_workers=True)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_ds, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=configs.drop_last,
                                               num_workers=10,pin_memory=True,persistent_workers=True)

    return train_loader,valid_loader
    #return valid_loader,train_loader

def cross_data_generator(data_path,train_idxs,val_idxs,configs):
    mod = configs.ft_mod
    mod_dict = {'eeg':0,'emg':1,'eog':2}
    train_ds = torch.load(os.path.join(data_path, "train.pt"))
    train_ds['samples'] = train_ds['samples'][:,mod_dict[mod],:].unsqueeze(1)
    valid_ds = torch.load(os.path.join(data_path, "val.pt"))
    valid_ds['samples'] = valid_ds['samples'][:,mod_dict[mod],:].unsqueeze(1)
    
    if train_idxs !=[]:

        dataset = {}
        train_dataset = {}
        valid_dataset = {}
        dataset['samples'] = np.vstack((train_ds['samples'],valid_ds['samples']))
        dataset['labels'] = np.hstack((train_ds['labels'],valid_ds['labels']))

        train_dataset['samples'] = dataset['samples'][train_idxs]
        train_dataset['labels'] = dataset['labels'][train_idxs]
        train_dataset = Load_Dataset(train_dataset, configs,training_mode='ft')

        valid_dataset['samples'] = dataset['samples'][val_idxs]
        valid_dataset['labels'] = dataset['labels'][val_idxs]
        valid_dataset = Load_Dataset(valid_dataset,configs,training_mode='ft')

        #test_dataset = Load_Dataset(test_dataset, configs,training_mode='ft')

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                                   shuffle=True, drop_last=configs.drop_last,
                                                   num_workers=10,pin_memory=True,persistent_workers=True)

        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=configs.batch_size,
                                                   shuffle=False, drop_last=configs.drop_last,
                                                   num_workers=10,pin_memory=True,persistent_workers=True)

        #test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
        #                                          shuffle=False, drop_last=False,
        #                                          num_workers=10,pin_memory=True,persistent_workers=True)
        del dataset
        del train_dataset
        del valid_dataset

        return train_loader,valid_loader
    ret = train_ds['samples'].shape[0]+valid_ds['samples'].shape[0]
    del train_ds
    del valid_ds
    return ret
