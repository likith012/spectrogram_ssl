import torch
import os
import numpy as np
import wandb
from data_preprocessing.dataloader import ft_data_generator
from models.model import encoder
from config import Config
from datetime import datetime
from tqdm import tqdm

name = str(datetime.now())
wandb.init(project='finv1_fusion_intra',notes='clustering',save_code=True,entity='sleep-staging',name='fusion cluster')
wandb_config = wandb.config
config = Config(wandb_config)
device = config.device
print(device)

SEED = 23
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

exp_path = '/home2/vivek.talwar/sleepedf/experiment_logs/experiment/saved_models/'
model_name = 'finv1_fusion_intra_epoch160.pt'
data_path = '/scratch/SLEEP_data'
#print(ss_model_name)
chkpoint = torch.load(os.path.join(exp_path,model_name),map_location=device)
pretrained_dict = chkpoint['eeg_model_state_dict']

# Logging
train_dl,_ = ft_data_generator(data_path,config)

model = encoder(config).to(device)
model.load_state_dict(pretrained_dict)
model.eval()

fin=[]
targets = []
for x,y in tqdm(train_dl):
    x = x.float().to(device)
    feat2,feat1 = model(x)
    feat1 = torch.cat((feat1,feat2),dim=-1)
    print("Hello")
    if fin==[]:
        fin=feat1.cpu().detach().numpy()
        targets = y.cpu().detach().numpy()
    else:
        fin=np.append(fin,feat1.cpu().detach().numpy(),axis=0)
        targets = np.append(targets,y.cpu().detach().numpy(),axis=0)


import matplotlib.pyplot as plt
import umap
reducer = umap.UMAP()
embedding = reducer.fit_transform(fin)
colors = ['r', 'g', 'b', 'y', 'm']
plt.scatter(embedding[:,0],embedding[:,1],c=[colors[int(col)] for col in targets])
plt.title("Red:Wake Green:1 Blue:2 Yellow:3 Magenta:REM")
wandb.log({'cluster chart':plt})
