import torch
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import io
import PIL
import math

from cirtorch.datasets.genericdataset import ImagesFromList
from cirtorch.datasets.genericdataset import ImagesFromList
from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from cirtorch.datasets.traindataset import TuplesDataset

torch.manual_seed(1)

"""
PARAMS
"""
BATCH_SIZE = 1000
EPOCH = 1000

INPUT_DIM = 2048
HIDDEN_DIM1 = 1024
HIDDEN_DIM2 = 512
HIDDEN_DIM3 = 256
OUTPUT_DIM = 128 #TODO: Is this right?

LR = 0.01
WD = 4e-3

network_path = 'data/exp_outputs1/mapillary_resnet50_gem_contrastive_m0.70_adam_lr1.0e-06_wd1.0e-06_nnum5_qsize2000_psize20000_bsize5_uevery5_imsize1024/model_epoch38.pth.tar'
multiscale = '[1]'
imsize = 320

posDistThr = 25
negDistThr = 25
workers = 8
query_size = 2000
pool_size = 20000

t = time.strftime("%Y-%d-%m_%H:%M:%S", time.localtime())
tensorboard = SummaryWriter(f'data/correlation_runs/{INPUT_DIM}_{OUTPUT_DIM}_{t}')

"""
Dataset
"""
# Data loading code
print('MEAN: ' + str(model.meta['mean']))
print('STD: ' + str(model.meta['std']))

normalize = transforms.Normalize(mean=model.meta['mean'], std=model.meta['std'])
resize = transforms.Resize((int(imsize * 3/4), imsize), interpolation=2)

transform = transforms.Compose([
        resize,
        transforms.ToTensor(),
        normalize,
])

train_dataset = TuplesDataset(
        name='mapillary',
        mode='train',
        imsize=imsize,
        nnum=5,
        qsize=query_size,
        poolsize=pool_size,
        transform=transform,
        posDistThr=posDistThr,
        negDistThr=negDistThr, 
        root_dir = 'data',
        cities=''
)

val_dataset = TuplesDataset(
        name='mapillary',
        mode='val',
        imsize=imsize,
        nnum=5,
        qsize=float('Inf'),
        poolsize=float('Inf'),
        transform=transform,
        posDistThr=negDistThr, # Use 25 meters for both pos and neg
        negDistThr=negDistThr,
        root_dir = 'data',
        cities=''
)

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=workers, pin_memory=True, sampler=None,
        drop_last=True, collate_fn=collate_tuples
)


val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=workers, pin_memory=True,
        drop_last=True, collate_fn=collate_tuples
)

"""
NETWORK
"""
class CorrelationNet(torch.nn.Module):
    def __init__(self):
        super(CorrelationNet, self).__init__()
        self.input = torch.nn.Linear(INPUT_DIM, HIDDEN_DIM1)
        self.hidden1 = torch.nn.Linear(HIDDEN_DIM1, HIDDEN_DIM2)
        self.hidden2 = torch.nn.Linear(HIDDEN_DIM2, HIDDEN_DIM3)
        self.output = torch.nn.Linear(HIDDEN_DIM3, OUTPUT_DIM)

    def forward(self, x):
        x = F.leaky_relu(self.input(x))
        x = F.leaky_relu(self.hidden1(x))
        x = F.leaky_relu(self.hidden2(x))
        x = self.output(x)
        return x

"""
TRAINING
"""
# Network
net = CorrelationNet()
optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=math.exp(-0.01))
loss_func = torch.nn.MSELoss()

# Move to GPU
net = net.cuda()
loss_func = loss_func.cuda()

# Train loop
losses = np.zeros(EPOCH)
for epoch in range(EPOCH):    
    epoch_loss = 0
    for i, (input, target, gps_info) in enumerate(train_loader):
        
        nq = len(input) # number of training tuples
        ni = len(input[0]) # number of images per tuple
        gps_info = torch.tensor(gps_info)

        for q in range(nq):
            output = torch.zeros(OUTPUT_DIM, ni).cuda()
            for imi in range(ni):
                # compute output vector for image imi
                output[:, imi] = model(input[q][imi].cuda()).squeeze()
        
        loss = mse_loss(output, target[q].cuda(), gps_info[q])
        epoch_loss += loss
 
        loss.backward()         
 
    tensorboard.add_scalar('Loss/train', epoch_loss, epoch)

    if (epoch % (EPOCH // 100) == 0 or (epoch == (EPOCH-1))):
        test(net, val_loader)

    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()


def mse_loss(output, target, gps_info):
    return 1.0