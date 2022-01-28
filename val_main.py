
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
config = Config()
def ft_fun(file_name,epoch):
    try:
        #file_name = file_name+"_epoch"+str(epoch)+'.pt'
        name = os.path.join(config.exp_path,file_name)
        src_path = '/scratch/SLEEP_data/'
        wandb.init(project='finv1_fusion_intra',notes='',name=file_name,save_code=True,entity='sleep-staging')
        train_dl,valid_dl= ft_data_generator(src_path,config)
        le_model = sleep_ft(name,config,train_dl,valid_dl,'hello',wandb)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        le_trainer = pl.Trainer(callbacks=[lr_monitor],profiler='simple',enable_checkpointing=False,max_epochs=config.num_ft_epoch,gpus=1)
        wandb.watch([le_model],log='all',log_freq=200)
        le_trainer.fit(le_model)
        wandb.finish(quiet=True)
    except:
        print("Error in Linear Evaluation")
        pass


parser = argparse.ArgumentParser()

parser.add_argument("--file_name", type=str, default="delete.pt",
                    help="weights file name")

parser.add_argument("--epoch", type= int, default=160,
                    help="current epoch")

args = parser.parse_args()

ft_fun(args.file_name,int(args.epoch))
