import torch
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import io
import PIL
import math
import csv

from cirtorch.datasets.genericdataset import ImagesFromList
from cirtorch.datasets.genericdataset import ImagesFromList
from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from cirtorch.datasets.traindataset import TuplesDataset
from cirtorch.datasets.datahelpers import collate_tuples, cid2filename

torch.manual_seed(1)

"""
PARAMS
"""
BATCH_SIZE = 500
EPOCH = 200

INPUT_DIM = 2048
HIDDEN_DIM1 = 1024
HIDDEN_DIM2 = 512
HIDDEN_DIM3 = 1024
OUTPUT_DIM = 2048

LR = 0.01
WD = 4e-3

network_path = 'data/exp_outputs1/mapillary_resnet50_gem_contrastive_m0.70_adam_lr1.0e-06_wd1.0e-06_nnum5_qsize2000_psize20000_bsize5_uevery5_imsize1024/model_epoch480.pth.tar'
multiscale = '[1]'
imsize = 320

posDistThr = 15
negDistThr = 25
workers = 8
query_size = 2000
pool_size = 20000

t = time.strftime("%Y-%d-%m_%H:%M:%S", time.localtime())
tensorboard = SummaryWriter(f'data/localcorrelation_runs/model_{INPUT_DIM}_{OUTPUT_DIM}_{LR}_{t}')

"""
Dataset
"""
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

def linear_regression(ground_truth, prediction, mode, epoch):
    ground_truth = ground_truth.reshape((-1, 1))
    model = LinearRegression().fit(ground_truth, prediction)
    r_sq = model.score(ground_truth, prediction)
    slope = model.coef_
    
    tensorboard.add_scalar(f'Plots{mode}/Correlation', slope, epoch)  
    tensorboard.add_scalar(f'Plots{mode}/RSq', r_sq, epoch)
    return model

def plot_points(ground_truth, prediction, mode, epoch):
    plt.clf()
    plt.scatter(ground_truth, prediction, color = "blue", alpha=0.2)
    
    x = np.linspace(0, 25, 25)
    y = x
    plt.plot(x, y, color = "green")

    model = linear_regression(ground_truth, prediction, mode, epoch)
    x = np.linspace(0, 25, 25)
    y = model.coef_ * x + model.intercept_
    plt.plot(x, y, color = "blue")

    plt.xlabel('Ground Truth Distance [GPS]')
    plt.ylabel('Predicted Distance')

    plt.title("True Distance v. Predicted Distance")

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
            
    image = PIL.Image.open(buf)
    image = transforms.ToTensor()(image).unsqueeze(0)
    tensorboard.add_image(f'Distance Correlation - {mode}', image[0], epoch)

"""
NETWORK
"""
class CorrelationNet(torch.nn.Module):
    def __init__(self):
        super(CorrelationNet, self).__init__()
        self.input = torch.nn.Linear(INPUT_DIM, HIDDEN_DIM1)
        self.hidden1 = torch.nn.Linear(HIDDEN_DIM1, HIDDEN_DIM2)
        self.hidden12 = torch.nn.Dropout(p=0.2)
        self.hidden2 = torch.nn.Linear(HIDDEN_DIM2, HIDDEN_DIM3)
        self.output = torch.nn.Linear(HIDDEN_DIM3, OUTPUT_DIM)

    def forward(self, x):
        x = F.leaky_relu(self.input(x))
        x = F.leaky_relu(self.hidden1(x))
        x = self.hidden12(x)
        x = F.leaky_relu(self.hidden2(x))
        x = self.output(x)
        return x

"""
TRAINING
"""
def distance(query, positive):
    return np.linalg.norm(np.array(query)-np.array(positive))

def distances(x, label, gps, eps=1e-6):
    # x is D x N
    dim = x.size(0) # D
    nq = torch.sum(label.data==-1) # number of tuples
    S = x.size(1) // nq # number of images per tuple including query: 1+1+n

    x1 = x[:, ::S].permute(1,0).repeat(1,S-1).view((S-1)*nq,dim).permute(1,0)
    idx = [i for i in range(len(label)) if label.data[i] != -1]
    x2 = x[:, idx]
    lbl = label[label!=-1]

    dif = x1 - x2
    D = torch.pow(dif+eps, 2).sum(dim=0).sqrt()
    dist = 1
    if len(gps) > 0:
        dist = distance(gps[0], gps[1])
    return gps, D, lbl

def mse_loss(x, label, gps, eps=1e-6, margin=25):
    dist, D, lbl = distances(x, label, gps, eps=1e-6)
    y = lbl*torch.pow((gps[0] - D),2) #+ 0.5*(1-lbl)*torch.pow(torch.clamp(margin-D, min=0),2)
    y = torch.sum(y)
    return y

def hubert_loss(x, label, gps, eps=1e-6, margin=25, delta=2.5):
    dist, D, lbl = distances(x, label, gps, eps=1e-6)
    if D[0] <= delta:
        y = lbl*torch.pow((dist - D),2)
    else:
        y = lbl*torch.abs(dist - D) - 1/2 * delta**2
    y += 0.5*(1-lbl)*torch.pow(torch.clamp(margin-D, min=0),2)
    y = torch.sum(y)
    return y

def dump_data(place_model, correlation_model, loader, epoch):
    place_model.eval()
    correlation_model.eval()

    #avg_neg_distance = val_loader.dataset.create_epoch_tuples(place_model) 
    score = 0
    for i, (input, target, gps_info) in enumerate(loader):     
        nq = len(input) # number of training tuples
        ni = len(input[0]) # number of images per tuple
        gps_info = torch.tensor(gps_info)
        
        dist_lat = np.zeros(nq)
        dist_gps = np.zeros(nq)
        images = []

        for q in range(nq):
            output = torch.zeros(OUTPUT_DIM, ni).cuda()
            for imi in range(ni):
                # compute output vector for image imi
                output[:, imi] = correlation_model(place_model(input[q][imi].cuda()).squeeze())
            loss = mse_loss(output, target[q].cuda(), gps_info[q])
            score += loss
        
            dist, D, lbl = distances(output, target[q].cuda(), gps_info[q])
            D = D.cpu()
            dist_lat[q] = D[0]
            dist_gps[q] = dist[0]
            
            #q = loader.qImages[loader.qidxs[i]]
            #p = loader.dbImages[loader.pidxs[i]][0] #TODO: Revert GetItem Randomness for this to work
            #images.append([q,p])
        
        del output
        break
    np.savetxt(f'plots/gps_{epoch}', dist_gps, delimiter=",")
    np.savetxt(f'plots/embedding_{epoch}', dist_lat, delimiter=",")
    #with open(f'plots/pictures_{epoch}.csv', "w") as f:
        #writer = csv.writer(f, dialect='excel')
        #writer.writerows(images)

def test(place_model, correlation_model, val_loader, epoch):
    place_model.eval()
    correlation_model.eval()

    #avg_neg_distance = val_loader.dataset.create_epoch_tuples(place_model) 
    score = 0
    for i, (input, target, gps_info) in enumerate(val_loader):     
        nq = len(input) # number of training tuples
        ni = len(input[0]) # number of images per tuple
        gps_info = torch.tensor(gps_info)
        dist_lat = np.zeros(nq)
        dist_gps = np.zeros(nq)

        for q in range(nq):
            output = torch.zeros(OUTPUT_DIM, ni).cuda()
            for imi in range(ni):
                # compute output vector for image imi
                output[:, imi] = correlation_model(place_model(input[q][imi].cuda()).squeeze())
            loss = mse_loss(output, target[q].cuda(), gps_info[q])
            score += loss
        
            # Only for first batch
            if i == 0:
                dist, D, lbl = distances(output, target[q].cuda(), gps_info[q])
                D = D.cpu()
                dist_lat[q] = D[0]
                dist_gps[q] = dist[0]
        
        if i == 0 and (epoch % (EPOCH // 100) == 0 or (epoch == (EPOCH-1))):
            plot_points(dist_gps, dist_lat, 'Validation', epoch)
            linear_regression(dist_gps, dist_lat, 'Training', epoch)
        
        del output
    tensorboard.add_scalar('Loss/validation', score, epoch)

def log_tuple(input, q, epoch, i):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    images = input[q][0] * std + mean
    for image_tensor in input[q][1:]:
        new_image = image_tensor * std + mean
        images = torch.cat([images, new_image], dim=0)
        tensorboard.add_images('ImageBatch: {}'.format(epoch + i), images, 0)


# Train loop
def train(train_loader, place_model, correlation_model, criterion, optimizer, scheduler, epoch):
        place_model.eval()
        correlation_model.train()
        
        #avg_neg_distance = train_loader.dataset.create_epoch_tuples(place_model) 
        
        epoch_loss = 0
        for i, (input, target, gps_info) in enumerate(train_loader):       
            nq = len(input) # number of training tuples
            ni = len(input[0]) # number of images per tuple
            #gps_info = torch.tensor(gps_info)
            dist_lat = np.zeros(nq)
            dist_gps = np.zeros(nq)
            for q in range(nq):
                output = torch.zeros(OUTPUT_DIM, ni).cuda()
                for imi in range(ni):
                    # compute output vector for image imi
                    output[:, imi] = correlation_model(place_model(input[q][imi].cuda()).squeeze())

                loss = criterion(output, target[q].cuda(), gps_info[q])
                epoch_loss += loss
                loss.backward()    

                # Only for first batch
                if i == 0 and (epoch % (EPOCH // 100) == 0 or (epoch == (EPOCH-1))):
                    dist, D, lbl = distances(output, target[q].cuda(), gps_info[q])
                    D = D.cpu()
                    dist_lat[q] = D[0]
                    dist_gps[q] = dist[0]
            #if epoch % (EPOCH // 10) == 0:
            #    dump_data(place_model, correlation_model, train_loader, epoch)
            if i == 0 and (epoch % (EPOCH // 100) == 0 or (epoch == (EPOCH-1))):
                average_dist = np.absolute(dist_gps - dist_lat)
                tensorboard.add_scalar('Distances/AvgErrorDistance', np.mean(average_dist), epoch) 
                plot_points(dist_gps, dist_lat, 'Training', epoch)
                linear_regression(dist_gps, dist_lat, 'Training', epoch)
    
        tensorboard.add_scalar('Loss/train', epoch_loss, epoch)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        del output

def main():
    # Load Networks
    net = CorrelationNet()
    model = load_placereg_net()

    # Move to GPU
    net = net.cuda()
    model = model.cuda()

    # Get transformer for dataset
    normalize = transforms.Normalize(mean=model.meta['mean'], std=model.meta['std'])
    resize = transforms.Resize((int(imsize * 3/4), imsize), interpolation=2)

    transform = transforms.Compose([
        resize,
        transforms.ToTensor(),
        normalize,
    ])

    # Load Datasets
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
        cities='debug', 
        tuple_mining='gps'
    )

    val_dataset = TuplesDataset(
            name='mapillary',
            mode='val',
            imsize=imsize,
            nnum=1,
            qsize=float('Inf'),
            poolsize=float('Inf'),
            transform=transform,
            posDistThr=posDistThr, # Use 25 meters for both pos and neg
            negDistThr=negDistThr,
            root_dir = 'data',
            cities='debug',
            tuple_mining='gps'
    )

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=workers, pin_memory=True, sampler=None,
            drop_last=True, collate_fn=collate_tuples
    )


    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=(BATCH_SIZE - 100), shuffle=False,
            num_workers=workers, pin_memory=True,
            drop_last=True, collate_fn=collate_tuples
    )

    # Optimizer, scheduler and criterion
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=math.exp(-0.01))
    criterion = mse_loss
    print('Creating epoch tuples')
    avg_neg_distance = train_loader.dataset.epoch_tuples_gps(model)
    avg_neg_distance = val_loader.dataset.epoch_tuples_gps(model)
    print('Done')
    # Train loop
    losses = np.zeros(EPOCH)
    for epoch in range(EPOCH):
        print(f'====> {epoch}/{EPOCH}')
        #dump_data(train_loader, model, net, criterion, optimizer, scheduler, epoch)
        train(train_loader, model, net, criterion, optimizer, scheduler, epoch)

        if (epoch % (EPOCH // 100) == 0 or (epoch == (EPOCH-1))):
            with torch.no_grad():
                test(model, net, val_loader, epoch)
            

            torch.save(net.state_dict(), f'data/localcorrelationnet/model_{INPUT_DIM}_{OUTPUT_DIM}_{LR}_Epoch_{epoch}.pth')

if __name__ == '__main__':
    main()
