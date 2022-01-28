
#%%
import argparse
import wandb
import numpy as np
import pytorch_lightning as pl
import os
from sklearn.model_selection import KFold
from pytorch_lightning.callbacks import LearningRateMonitor
from data_preprocessing.dataloader import data_generator,cross_data_generator,ft_data_generator
from trainer import sleep_ft,sleep_pretrain
from config import Config

#path = "/scratch/SLEEP_data/data_multi/sleepEDF/"
path = "/scratch/SLEEP_data/"


training_mode = 'ss'
for i in range(1,2):
    inp_epoch = 20*(8-i)
    config = Config()
    def ft_fun(file_name,epoch):
        file_name = file_name+'_epoch'+str(inp_epoch)+'.pt'
        name = os.path.join(config.exp_path,file_name)
        print(name)
        src_path = '/scratch/SLEEP_data/'
        wandb.init(project='finv1_fusion_intra_new_sch',notes='',name=file_name+"_epoch"+str(epoch),save_code=True,entity='sleep-staging')
        train_dl,valid_dl= ft_data_generator(src_path,config)
        le_model = sleep_ft(name,config,train_dl,valid_dl,epoch,wandb)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        le_trainer = pl.Trainer(callbacks=[lr_monitor],enable_checkpointing=False,max_epochs=config.num_ft_epoch,gpus=1)
        wandb.watch([le_model],log='all',log_freq=200)
        le_trainer.fit(le_model)
        wandb.finish(quiet=True)


    parser = argparse.ArgumentParser()
    
    parser.add_argument("--file_name", type=str, default="finv1_fusion_intra_new_sch",
                        help="weights file name")
    
    parser.add_argument("--epoch", type= int, default=0,
                        help="current epoch")
    
    args = parser.parse_args()
    
    ft_fun(args.file_name,int(inp_epoch))
