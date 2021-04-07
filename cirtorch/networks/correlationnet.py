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

t = time.strftime("%Y-%d-%m_%H:%M:%S", time.localtime())
tensorboard = SummaryWriter(f'data/correlation_runs/{INPUT_DIM}_{OUTPUT_DIM}_{t}')

"""
Dataset
"""
tensor_meta = []
def standardize(tensor, dimension, save=False):
    global tensor_meta
    means = tensor.mean(dim=dimension, keepdim=True)
    stds = tensor.std(dim=dimension, keepdim=True)
    standardized_tensor = (tensor - means) / stds
    means = means.cpu()
    stds = stds.cpu()
    if save:
        tensor_meta = [means.numpy(), stds.numpy()]
    return standardized_tensor


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
        mode='train',
        imsize=imsize,
        transform=transform,
        posDistThr=posDistThr,
        negDistThr=negDistThr,
        tuple_mining='gps'
    )
    qidxs, pidxs = test_dataset.get_loaders()

    opt = {'batch_size': 1, 'shuffle': False, 'num_workers': 8, 'pin_memory': True}

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
    
    # Dataset
    input_data = poolvecs.T
    output_data = poolcoordinates

    input_data = standardize(input_data, 0)
    output_data = standardize(output_data, 0, save=True)

    print(tensor_meta)
    input_data = input_data.cuda()
    output_data = output_data.cuda() 
    N, _ = output_data.size()

    print(input_data.size(), output_data.size())    
    
    torch_dataset = Data.TensorDataset(input_data, output_data)
    train_set, val_set = torch.utils.data.random_split(torch_dataset, [N - N // 5, N // 5])    
    
    train_loader = Data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,)
    val_loader = Data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,)

    return train_loader, val_loader


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


def plot_points(ground_truth, prediction, mode='Train'):
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

def plot_correlation(ground_truth, prediction, mode='Train'):
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

def local_correlation_plot(ground_truth, prediction, mode='Train', point=10):
    plt.clf()
    #ground_truth = ground_truth.data.numpy()
    #prediction = prediction.data.numpy()

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


def test(network, validation_loader):
    score = 0
    network.eval()
    for step, (batch_x, batch_y) in enumerate(validation_loader):
        prediction = network(batch_x)
        score = loss_func(prediction, batch_y)

        if step == 1:
            batch_y = batch_y.cpu()
            prediction = prediction.cpu()
            plot_points(batch_y, prediction, 'Validation')
            plot_correlation(batch_y, prediction, 'Validation')  
            tensorboard.add_scalar('Loss/validation', score, epoch)


"""
TRAINING
"""
# Dataset
loader, val_loader = main()

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
    for step, (batch_x, batch_y) in enumerate(loader):

        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        prediction = net(b_x)

        loss = loss_func(prediction, b_y)
        epoch_loss += loss
 
        loss.backward()         

        if step == 1 and (epoch % (EPOCH // 100) == 0 or (epoch == (EPOCH-1))):
            b_y = b_y.cpu()
            prediction = prediction.cpu()
            
            plot_points(b_y, prediction, 'Train')
            plot_correlation(b_y, prediction, 'Train')
            local_correlation_plot(b_y, prediction, 'Train', 10)
            local_correlation_plot(b_y, prediction, 'Train', 42) 
            local_correlation_plot(b_y, prediction, 'Train', 58)
            local_correlation_plot(b_y, prediction, 'Train', 91)     
 
    tensorboard.add_scalar('Loss/train', epoch_loss, epoch)

    if (epoch % (EPOCH // 100) == 0 or (epoch == (EPOCH-1))):
        test(net, val_loader)

    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
