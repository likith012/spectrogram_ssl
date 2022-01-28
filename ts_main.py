#%%
import wandb
import numpy as np
import pytorch_lightning as pl
import os
import torch
from sklearn.model_selection import KFold
from pytorch_lightning.callbacks import LearningRateMonitor
from data_preprocessing.dataloader import data_generator,cross_data_generator,ft_data_generator
from trainer import sleep_ft,sleep_pretrain
from config import Config

#path = "/scratch/SLEEP_data/data_multi/sleepEDF/"
path = "/scratch/SLEEP_data/"


training_mode = 'ss'
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

#%%
# for self supervised training
if training_mode == 'ss':
    name = 'spect_det_kfold'
    ss_wandb = wandb.init(project='pretrain_kfold',name=name,notes='have used spectrogram and time with 3 contrastive loss',save_code=True,entity='sleep-staging')
    config = Config(ss_wandb)
    ss_wandb.save('/home2/vivek.talwar/exps/spect//config.py')
    ss_wandb.save('/home2/vivek.talwar/exps/spect//trainer.py')
    ss_wandb.save('/home2/vivek.talwar/exps/spect//data_preprocessing/*')
    ss_wandb.save('/home2/vivek.talwar/exps/spect//models/*')
    print("Loading")
    dataloader = data_generator(path,config)
    print("Done")
    #%%
    model = sleep_pretrain(config,name,dataloader,ss_wandb)
    print('Model Loaded')
    #ss_wandb.watch([model],log='all',log_freq=500)
    model.fit()
    ss_wandb.finish()
#%%

# cross validation linear evaluation
elif training_mode == 'cross':
    config = Config()
    file_name = 'resnet50_test.pt'
    name = os.path.join(config.exp_path, file_name) 
    src_path = "/scratch/SLEEP_data/data_multi/sleepEDF/"
    n = cross_data_generator(src_path,[],[],config)
    kfold = KFold(n_splits=5,shuffle=False)
    idxs = np.arange(0,n,1)
    #%%
    for split,(train_idx,val_idx) in enumerate(kfold.split(idxs)):
        wandb.init(project='1d_resnet_test',notes='',save_code=True,entity='sleep-staging',group="modified resnet",job_type='split: '+str(split))
        train_dl,valid_dl= cross_data_generator(src_path,train_idx,val_idx,config)
        le_model = sleep_ft(name,config,train_dl,valid_dl,wandb)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        le_trainer = pl.Trainer(callbacks=[lr_monitor],profiler='simple',enable_checkpointing=False,max_epochs=config.num_ft_epoch,gpus=1)
        le_trainer.fit(le_model)
        wandb.watch([le_model],log='all',log_freq=200)
        wandb.finish()

