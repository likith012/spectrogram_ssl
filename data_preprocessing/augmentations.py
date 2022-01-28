#%%
import numpy as np
import torch
from scipy.interpolate import interp1d

def noise_channel(ts, mode, degree):
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

def jitter(x, config,degree_scale=1):

    #mode = np.random.choice(['high', 'low', 'both', 'no'])
    mode = 'both' 

    ret = []
    for chan in range(x.shape[0]):
        ret.append(noise_channel(x[chan],mode,config.degree*degree_scale))
    ret = np.vstack(ret)
    ret = torch.from_numpy(ret)
    return ret 

def scaling(x,config,degree_scale=2):
    #eprint(x.shape)
    ret = np.zeros_like(x)
    for i in range(x.shape[0]):
        if i!=1: 
            degree = config.degree*(degree_scale+np.random.rand())
        else:#for emg
            degree = 0.006*(degree_scale+np.random.rand())
        factor = 2*np.random.normal(size=x.shape[1])-1
        ret[i]=x[i]*(1.5+(2*np.random.rand())+degree*factor)
    ret = torch.from_numpy(ret)
    return ret 

def masking(x,config):
    # for each modality we are using different masking
    segments = config.mask_min_points + int(np.random.rand()*(config.mask_max_points-config.mask_min_points))
    points = np.random.randint(0,3000-segments)
    ret = x.detach().clone()
    for i,k in enumerate(x):
        ret[i,points:points+segments] = 0

    return ret

def multi_masking(x, mask_min = 40, mask_max = 0,min_seg= 8, max_seg= 14):
    # Applied idependently to each channel
    fin_masks = []
    segments = min_seg + int(np.random.rand()*(max_seg-min_seg))
    for seg in range(segments):
        fin_masks.append(mask_min + int(np.random.rand() * (mask_max-mask_min)))
    points = np.random.randint(0, 3000-segments,size=segments)
    ret = x.clone()
    for i,k in enumerate(x):
        for seg in range(segments):
            ret[i, points[seg]:points[seg]+fin_masks[seg]] = 0
    return ret

def permutation(x,config):
    # for all modalities same permutation
    orig_steps = np.arange(x.shape[1])
    num_segs = np.random.randint(2,config.permutation_segments+1)
    ret = np.zeros_like(x)
    if config.permutation_mode=='random':
        split_points = np.random.choice(x.shape[1]-2,num_segs-1,replace=False)
        split_points.sort()
        splits = np.split(orig_steps,split_points)
    else:
        splits = np.array_split(orig_steps,num_segs)
    np.random.shuffle(splits) # inplace 
    warp = np.concatenate(splits).ravel()
    
    for i in range(x.shape[0]):
        ret[i] = x[i,warp]

    return ret

def flip(x,config):
    # horizontal flip
    if np.random.rand() >0.5:
        return torch.tensor(np.flip(x.numpy(),1).copy())
    else:
        return x

def augment(x,config):
    ''' use jitter in every aug to get two different eeg signals '''
    choice = np.random.rand()
    if choice<0.99:
        weak_ret = masking(jitter(x,config),config)
        strong_ret = scaling(flip(x,config),config,degree_scale=3)
    elif choice<0.7:
        weak_ret = permutation(jitter(x,config),config)
        strong_ret = scaling(permutation(x,config),config)
    else:
        weak_ret = scaling(flip(masking(x,config),config),config)
        strong_ret = flip(masking(jitter(x,config,degree_scale=1.5*np.random.rand()+1),config),config)
    return weak_ret,strong_ret
#%%
#import numpy as np
#from scipy.interpolate import interp1d
#import copy
#
##Augmentations
#def noise_channel(ts, mode, degree):
#    """
#    Add noise to ts
#    
#    mode: high, low, both
#    degree: degree of noise, compared with range of ts    
#    
#    Input:
#        ts: (n_length)
#    Output:
#        out_ts: (n_length)
#        
#    """
#    len_ts = len(ts)
#    num_range = np.ptp(ts)+1e-4 # add a small number for flat signal
#    ### high frequency noise
#    if mode == 'high':
#        noise = degree * num_range * (2*np.random.rand(len_ts)-1)
#        out_ts = ts + noise
#    ### low frequency noise
#    elif mode == 'low':
#        noise = degree * num_range * (2*np.random.rand(len_ts//100)-1)
#        x_old = np.linspace(0, 1, num=len_ts//100, endpoint=True)
#        x_new = np.linspace(0, 1, num=len_ts, endpoint=True)
#        f = interp1d(x_old, noise, kind='linear')
#        noise = f(x_new)
#        out_ts = ts + noise
#    ### both high frequency noise and low frequency noise
#    elif mode == 'both':
#        noise1 = degree * num_range * (2*np.random.rand(len_ts)-1)
#        noise2 = degree * num_range * (2*np.random.rand(len_ts//100)-1)
#        x_old = np.linspace(0, 1, num=len_ts//100, endpoint=True)
#        x_new = np.linspace(0, 1, num=len_ts, endpoint=True)
#        f = interp1d(x_old, noise2, kind='linear')
#        noise2 = f(x_new)
#        out_ts = ts + noise1 + noise2
#    return out_ts
#def jitter(x, degree = 0.07, degree_scale=1):
#    # Applied idependently to each channel
#    ret = []
#    for ch in range(x.shape[0]):
#        mode = np.random.choice(['high', 'low', 'both'])
#        ret.append(noise_channel(x[ch], mode, degree * degree_scale))
#    ret = np.vstack(ret)
#    ret = torch.from_numpy(ret)
#    return ret
#def scaling(x, sigma=0.3):
#    # Not applied independently to each channel
#    factor = np.random.normal(loc=1.0, scale=sigma, size= x.shape[1])
#    ai = []
#    for ch in range(x.shape[0]):
#        ai.append(np.multiply(x[ch], factor))
#    return torch.from_numpy(np.stack((ai), axis=0))
#def crop(x, n_length = 100*30):
#    # Not applied independently to each channel
#    l = np.random.randint(1, n_length - 1)
#    ret = x.clone()
#    ret[:, :l], ret[:, l:] = ret[:, -l:], ret[:, :-l]
#    return ret
#def masking(x, mask_min = 40, mask_max = 0,min_seg= 8, max_seg= 14):
#    # Applied idependently to each channel
#    fin_masks = []
#    segments = min_seg + int(np.random.rand()*(max_seg-min_seg))
#    for seg in range(segments):
#        fin_masks.append(mask_min + int(np.random.rand() * (mask_max-mask_min)))
#    points = np.random.randint(0, 3000-segments,size=segments)
#    ret = x.clone()
#    for i,k in enumerate(x):
#        for seg in range(segments):
#            ret[i, points[seg]:points[seg]+fin_masks[seg]] = 0
#    return ret
#def augment(x,config):
#    mode  = np.random.choice(['jitter', 'scale', 'mask'])
#    #mode='jitter'
#    #if mode == 'jitter':
#    #    x = jitter(x)
#    #elif mode == 'scale':
#    #    x = scaling(x)
#    #elif mode == 'mask':
#    #    x = masking(x)
#    #elif mode == 'crop':
#    #    x = crop(x)
#    weak_ret = jitter(x)
#    strong_ret = scaling(masking(x))
#    return weak_ret,strong_ret
