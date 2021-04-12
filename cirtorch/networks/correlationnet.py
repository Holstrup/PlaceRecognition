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
OUTPUT_DIM = 2

LR = 0.01
WD = 4e-3

network_path = 'data/exp_outputs1/mapillary_resnet50_gem_contrastive_m0.70_adam_lr1.0e-06_wd1.0e-06_nnum5_qsize2000_psize20000_bsize5_uevery5_imsize1024/model_epoch38.pth.tar'
multiscale = '[1]'
imsize = 1024
posDistThr = 25
negDistThr = 25
query_size = 2000
pool_size = 20000

t = time.strftime("%Y-%d-%m_%H:%M:%S", time.localtime())
tensorboard = SummaryWriter(f'data/correlation_runs/{INPUT_DIM}_{OUTPUT_DIM}_{t}')

"""
Dataset
"""
def standardize(tensor, dimension):
    means = tensor.mean(dim=dimension, keepdim=True)
    stds = tensor.std(dim=dimension, keepdim=True)
    return (tensor - means) / stds

def load_placereg_net():
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
    return net

def load_dataloader(place_model, dataset, transform):
    qidxs, pidxs = dataset.get_loaders()

    opt = {'batch_size': 1, 'shuffle': False, 'num_workers': 8, 'pin_memory': True}

    # Step 1: Extract Database Images - dbLoader
    dbLoader = torch.utils.data.DataLoader(
          ImagesFromList(root='', images=[dataset.dbImages[i] for i in range(
               len(dataset.dbImages))], imsize=imsize, transform=transform), **opt)
    
    poolvecs = torch.zeros(place_model.meta['outputdim'], len(dataset.dbImages)).cuda()
    for i, input in enumerate(dbLoader):
            poolvecs[:, i] = place_model(input.cuda()).data.squeeze()

    # Step 2: Extract Query Images - qLoader
    qLoader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=[
                           dataset.qImages[i] for i in qidxs], imsize=imsize, transform=transform), **opt)

    qvecs = torch.zeros(place_model.meta['outputdim'], len(qidxs)).cuda()
    for i, input in enumerate(qLoader):
            qvecs[:, i] = place_model(input.cuda()).data.squeeze()

    # GPS: get query and pool coordinates
    querycoordinates = torch.tensor(
            [dataset.gpsInfo[dataset.qImages[i][-26:-4]] for i in qidxs], dtype=torch.float)
    poolcoordinates = torch.tensor([dataset.gpsInfo[dataset.dbImages[i][-26:-4]]
                                    for i in range(len(dataset.dbImages))], dtype=torch.float)
    
    # Dataset
    input_data = poolvecs.T
    output_data = poolcoordinates

    input_data = standardize(input_data, 0)
    output_data = standardize(output_data, 0)

    input_data = input_data.cuda()
    output_data = output_data.cuda() 
    N, _ = output_data.size()

    print(input_data.size(), output_data.size())    
    
    torch_dataset = Data.TensorDataset(input_data, output_data)
    
    data_loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,)

    return data_loader


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


def plot_points(ground_truth, prediction, mode, epoch):
    plt.clf()
    plt.scatter(ground_truth.data[:, 0].numpy(), ground_truth.data[:, 1].numpy(), color = "blue", alpha=0.2)
    plt.scatter(prediction.data[:, 0].numpy(), prediction.data[:, 1].numpy(), color = "red", alpha=0.2)

    plt.title("Coordinates")

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
            
    image = PIL.Image.open(buf)
    image = transforms.ToTensor()(image).unsqueeze(0)
    tensorboard.add_image(f'Coordinates - {mode}', image[0], epoch)

def plot_correlation(ground_truth, prediction, mode, epoch):
    plt.clf()
    ground_truth = ground_truth.data.numpy()
    prediction = prediction.data.numpy()

    true_distances = np.linalg.norm(ground_truth - ground_truth[10], axis=1)
    pred_distances = np.linalg.norm(prediction - prediction[10], axis=1)

    plt.scatter(true_distances, pred_distances)
    plt.xlim([0, true_distances[-1]])
    plt.ylim([0, 2*true_distances[-1]])
    plt.title("Correlation between true distances and pred. distances")

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
            
    image = PIL.Image.open(buf)
    image = transforms.ToTensor()(image).unsqueeze(0)
    tensorboard.add_image(f'Correlation - {mode}', image[0], epoch)


"""
TRAINING
"""

def train(train_loader, place_model, net, criterion, optimizer, scheduler, epoch):
    epoch_loss = 0
    for step, (batch_x, batch_y) in enumerate(train_loader):
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        prediction = net(b_x)
        loss = criterion(prediction, b_y)

        epoch_loss += loss
        loss.backward()         

        if step == 1 and (epoch % (EPOCH // 100) == 0 or (epoch == (EPOCH-1))):
            b_y = b_y.cpu()
            prediction = prediction.cpu()
            
            plot_points(b_y, prediction, 'Train', epoch)
            plot_correlation(b_y, prediction, 'Train', epoch) 
 
    tensorboard.add_scalar('Loss/train', epoch_loss, epoch)

    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

def test(net, criterion, val_loader, epoch):
    score = 0
    net.eval()
    for step, (batch_x, batch_y) in enumerate(val_loader):
        prediction = net(batch_x)
        score = criterion(prediction, batch_y)

        if step == 0:
            batch_y = batch_y.cpu()
            prediction = prediction.cpu()
            plot_points(batch_y, prediction, 'Validation', epoch)
            plot_correlation(batch_y, prediction, 'Validation', epoch)  
            tensorboard.add_scalar('Loss/validation', score, epoch)

def main():
    place_model = load_placereg_net()
    net = CorrelationNet()

    # moving network to gpu
    place_model.cuda()
    net.cuda()

    # set up the transform
    resize = transforms.Resize((240, 320), interpolation=2)
    normalize = transforms.Normalize(
        mean=place_model.meta['mean'],
        std=place_model.meta['std']
    )
    transform = transforms.Compose([
        resize,
        transforms.ToTensor(),
        normalize
    ])
    
    train_dataset = TuplesDataset(
        name='mapillary',
        mode='train',
        imsize=imsize,
        nnum=1,
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
            nnum=1,
            qsize=float('Inf'),
            poolsize=float('Inf'),
            transform=transform,
            posDistThr=negDistThr, # Use 25 meters for both pos and neg
            negDistThr=negDistThr,
            root_dir = 'data',
            cities=''
    )
    val_loader = load_dataloader(place_model, val_dataset, transform)
    
    # Optimizer, scheduler and criterion
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=math.exp(-0.01))
    criterion = torch.nn.MSELoss().cuda()

    losses = np.zeros(EPOCH)
    for epoch in range(EPOCH):
        print(f'====> {epoch}/{EPOCH}')
        # New tuples every epoch
        train_dataset.create_epoch_tuples(place_model)
        train_loader = load_dataloader(place_model, train_dataset, transform)

        train(train_loader, place_model, net, criterion, optimizer, scheduler, epoch)

        if (epoch % (EPOCH // 100) == 0 or (epoch == (EPOCH-1))):
            with torch.no_grad():
                test(net, criterion, val_loader, epoch)

main()

"""
def local_correlation_plot(ground_truth, prediction, mode='Train', point=10):
    plt.clf()

    # Ground Truth
    distances = torch.norm(ground_truth - ground_truth[point], dim=1)
    distances, indicies = torch.sort(distances, dim=0, descending=False)

    # Predicted
    pred_distances = torch.norm(prediction - prediction[point], dim=1)
    pred_distances = pred_distances.data.numpy()
    
    i = 0
    correlated_points = []
    while i < 10:
        x = pred_distances[indicies[i]]
        y = distances[i]
        correlated_points.append([x, y])
        i += 1
    
    correlated_points = np.array(correlated_points)
    #correlated_points = correlated_points * tensor_meta[1] + tensor_meta[0] 
    plt.scatter(correlated_points[:, 1], correlated_points[:, 0])
    plt.title("Correlation between true distances and pred. distances - Locally")

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
            
    image = PIL.Image.open(buf)
    image = transforms.ToTensor()(image).unsqueeze(0)
    tensorboard.add_image(f'Local Correlation point {point}- {mode}', image[0], epoch)
"""
