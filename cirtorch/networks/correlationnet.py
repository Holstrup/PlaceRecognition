import torch
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

from cirtorch.datasets.genericdataset import ImagesFromList
from cirtorch.datasets.genericdataset import ImagesFromList
from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from cirtorch.datasets.traindataset import TuplesDataset

torch.manual_seed(1)    # reproducible


"""
PARAMS
"""
root_path = '/Users/alexanderholstrup/Desktop/correlation_data'
pool_path = f'{root_path}/pool_raw.csv'
df_path = f'{root_path}/postprocessed.csv'
utm_path = f'{root_path}/pool_utm.txt'


BATCH_SIZE = 500
EPOCH = 10000

INPUT_DIM = 2048
HIDDEN_DIM1 = 1024
HIDDEN_DIM2 = 512
HIDDEN_DIM3 = 256
OUTPUT_DIM = 2

LR = 0.02
WD = 0.10 #4e-3

network_path = 'data/exp_outputs1/mapillary_resnet50_gem_contrastive_m0.70_adam_lr1.0e-06_wd1.0e-06_nnum5_qsize2000_psize20000_bsize5_uevery5_imsize1024/model_epoch38.pth.tar'
multiscale = 1
imsize = 1024

"""
Dataset
"""
def main():
    # loading network from path
    if network_path is not None:
        state = torch.load(network_path)

        # parsing net params from meta
        # architecture, pooling, mean, std required
        # the rest has default values, in case that is doesnt exist
        net_params = {}
        net_params['architecture'] = state['meta']['architecture']
        net_params['pooling'] = state['meta']['pooling']
        net_params['local_whitening'] = state['meta'].get(
            'local_whitening', False)
        net_params['regional'] = state['meta'].get('regional', False)
        net_params['whitening'] = state['meta'].get('whitening', False)
        net_params['mean'] = state['meta']['mean']
        net_params['std'] = state['meta']['std']
        net_params['pretrained'] = False

        # load network
        net = init_network(net_params)
        net.load_state_dict(state['state_dict'])

        # if whitening is precomputed
        if 'Lw' in state['meta']:
            net.meta['Lw'] = state['meta']['Lw']

        print(">>>> loaded network: ")
        print(net.meta_repr())

        # setting up the multi-scale parameters
    ms = list(eval(multiscale))
    if len(ms) > 1 and net.meta['pooling'] == 'gem' and not net.meta['regional'] and not net.meta['whitening']:
        msp = net.pool.p.item()
        print(">> Set-up multiscale:")
        print(">>>> ms: {}".format(ms))
        print(">>>> msp: {}".format(msp))
    else:
        msp = 1

    # moving network to gpu and eval mode
    net.cuda()
    net.eval()
    # set up the transform
    resize = transforms.Resize((240, 320), interpolation=2)
    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    transform = transforms.Compose([
        resize,
        transforms.ToTensor(),
        normalize
    ])
    posDistThr = 25
    negDistThr = 25
    test_dataset = TuplesDataset(
        name='mapillary',
        # mode='test',
        mode='train',
        imsize=imsize,
        transform=transform,
        posDistThr=posDistThr,
        negDistThr=negDistThr,
        tuple_mining='gps'
    )
    qidxs, pidxs = test_dataset.get_loaders()

    opt = {'batch_size': 1, 'shuffle': False,
           'num_workers': 8, 'pin_memory': True}

    # Step 1: Extract Database Images - dbLoader
    dbLoader = torch.utils.data.DataLoader(
          ImagesFromList(root='', images=[test_dataset.dbImages[i] for i in range(
               len(test_dataset.dbImages))], imsize=imsize, transform=transform),
          **opt)
    poolvecs = torch.zeros(net.meta['outputdim'], len(test_dataset.dbImages)).cuda()
    for i, input in enumerate(dbLoader):
            poolvecs[:, i] = net(input.cuda()).data.squeeze()

    # Step 2: Extract Query Images - qLoader
    qLoader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=[
                           test_dataset.qImages[i] for i in qidxs], imsize=imsize, transform=transform), **opt)

    qvecs = torch.zeros(net.meta['outputdim'], len(qidxs)).cuda()
    for i, input in enumerate(qLoader):
            qvecs[:, i] = net(input.cuda()).data.squeeze()

    # GPS: get query and pool coordinates
    querycoordinates = torch.tensor(
            [test_dataset.gpsInfo[test_dataset.qImages[i][-26:-4]] for i in qidxs], dtype=torch.float)
    poolcoordinates = torch.tensor([test_dataset.gpsInfo[test_dataset.dbImages[i][-26:-4]]
                                    for i in range(len(test_dataset.dbImages))], dtype=torch.float)

    # Step 3: Ranks
    scores = torch.mm(poolvecs.t(), qvecs)
    
    # GPS: Compute distances
    distances = torch.norm(querycoordinates[:, None] - poolcoordinates, dim=2)
    
    # Dataset
    print('Scores: ', scores.size())
    print('Distances: ', distances.size())
    torch_dataset = Data.TensorDataset(scores, distances)
    return Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0,)

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

    utm_coors -= np.mean(utm_coors, axis=0) #np.linalg.norm(utm_coors, axis=1)
    utm_coors /= 6000

    plt.scatter(utm_coors[:, 0], utm_coors[:, 1], color = "blue", alpha=0.2)
    plt.scatter(poolvecs[:, 0], poolvecs[:, 1], color = "red", alpha=0.2)
    # To tensor
    poolvecs = torch.Tensor(poolvecs)
    utm_coors = torch.Tensor(utm_coors)

    # Dataset
    torch_dataset = Data.TensorDataset(poolvecs, utm_coors)
    return Data.DataLoader(
        dataset=torch_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=0,)


"""
NETWORK
"""
# another way to define a network
net = torch.nn.Sequential(
    torch.nn.Linear(INPUT_DIM, HIDDEN_DIM1),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(HIDDEN_DIM1, HIDDEN_DIM2),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(HIDDEN_DIM2, HIDDEN_DIM3),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(HIDDEN_DIM3, OUTPUT_DIM),
)

# Define model
"""class CorrelationNet(torch.nn.Module):
    def __init__(self):
        super(CorrelationNet, self).__init__()
        self.input = torch.nn.Linear(INPUT_DIM, HIDDEN_DIM1)
        self.hidden1 = torch.nn.Linear(HIDDEN_DIM1, HIDDEN_DIM2)
        self.hidden2 = torch.nn.Linear(HIDDEN_DIM2, HIDDEN_DIM3)
        self.output = torch.nn.Linear(HIDDEN_DIM3, OUTPUT_DIM)

    def forward(self, x):
        x = torch.nn.LeakyReLU(self.input(x))
        x = torch.nn.LeakyReLU(self.hidden1(x))
        x = torch.nn.LeakyReLU(self.hidden2(x))
        x = self.output(x)
        return x
"""
optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

"""
TRAINING
"""

# Dataloader 
#loader = TrainDataset(pool_path, df_path, utm_path)
loader = main()

# Network
#net = CorrelationNet()
net.cuda()
loss_func.cuda()

# Train loop
losses = np.zeros(EPOCH)
for epoch in range(EPOCH):
    if epoch == EPOCH // 2:
        for g in optimizer.param_groups:
            g['lr'] = 0.005
    
    epoch_loss = 0
    for step, (batch_x, batch_y) in enumerate(loader):

        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        prediction = net(b_x)

        loss = loss_func(prediction, b_y)
        epoch_loss += loss
 
        loss.backward()         

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

torch.save(net.state_dict(), 'data/correlation_net/network.pth')