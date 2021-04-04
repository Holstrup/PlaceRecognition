import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

torch.manual_seed(1)    # reproducible





"""
PARAMS
"""
root_path = 'correlation_data'
pool_path = f'{root_path}/pool_raw.csv'
df_path = f'{root_path}/postprocessed.csv'
utm_path = f'{root_path}/pool_utm.txt'


BATCH_SIZE = 500
EPOCH = 10000

INPUT_DIM = 2048
HIDDEN_DIM1 = 1024
HIDDEN_DIM2 = 1024
OUTPUT_DIM = 2

LR = 0.02
WD = 0.10 #4e-3


def TrainDataset(poolpath, dataframe_path, utm_path):
    # X - Image embeddings
    poolvecs = np.loadtxt(poolpath)
    poolvecs = poolvecs.T

    # Y - Coordinates
    df = pd.read_csv(dataframe_path)
    utm_coors = np.zeros(2)
    with open(utm_path, 'r') as filehandle:
        for line in filehandle:
            key = line[:-1][-26:-4]
            utm = df.loc[df['key'] == key].values[0][2:4]
            utm_coors = np.vstack((utm_coors, utm))
    utm_coors = utm_coors[1:, :].astype(float)
    
    # To tensor
    poolvecs = torch.Tensor(poolvecs) 
    utm_coors = torch.Tensor(utm_coors)
    
    # Dataset
    torch_dataset = Data.TensorDataset(poolvecs, utm_coors) 
    return Data.DataLoader(
        dataset=torch_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=2,)


"""
NETWORK
"""
# another way to define a network
net = torch.nn.Sequential(
        torch.nn.Linear(INPUT_DIM, HIDDEN_DIM1),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(HIDDEN_DIM1, HIDDEN_DIM2),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(HIDDEN_DIM2, OUTPUT_DIM),
    )

optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss


"""
TRAINING
"""


loader = TrainDataset(pool_path, df_path, utm_path)
losses = np.zeros(EPOCH)
for epoch in range(EPOCH):
    if epoch == EPOCH // 2:
        for g in optimizer.param_groups:
            g['lr'] = 0.005
            print('New lr')
    epoch_loss = 0
    for step, (batch_x, batch_y) in enumerate(loader): # for each training step
        
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        prediction = net(b_x)    

        loss = loss_func(prediction, b_y) 
        epoch_loss += loss

        #optimizer.zero_grad()   
        loss.backward()         
        #optimizer.step()        

        if step == 1 and (epoch % (EPOCH // 10) == 0 or (epoch == (EPOCH-1))):
            plt.scatter(b_y.data[:, 0].numpy(), b_y.data[:, 1].numpy(), color = "blue", alpha=0.2)
            plt.scatter(prediction.data[:, 0].numpy(), prediction.data[:, 1].numpy(), color = "red", alpha=0.2)
            
            #plt.show()
            plt.savefig(f'correlation_plots/prediction_{epoch}.png')
            plt.clf()
    print(f'{epoch}/{EPOCH} => {epoch_loss}')
    losses[epoch] = epoch_loss
    optimizer.step()
    optimizer.zero_grad()

plt.plot(torch.linspace(0, EPOCH, EPOCH), losses, color = "blue", alpha=0.2)
plt.savefig(f'correlation_plots/loss.png')
plt.clf()

plt.plot(torch.linspace(EPOCH // 2, EPOCH, EPOCH // 2), losses[EPOCH // 2:], color = "blue", alpha=0.2)
plt.savefig(f'correlation_plots/loss1.png')
plt.clf()
