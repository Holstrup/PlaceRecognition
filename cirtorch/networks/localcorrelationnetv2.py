#import sys
#sys.path.insert(0, "/Users/alexanderholstrup/git/VisualPlaceRecognition/cnnimageretrieval-pytorch")

import argparse
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
import random

from cirtorch.datasets.genericdataset import ImagesFromList
from cirtorch.utils.evaluate import compute_map_and_print, mapk, recall
from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from cirtorch.datasets.traindataset import TuplesDataset
from cirtorch.datasets.datahelpers import collate_tuples, cid2filename
from cirtorch.utils.view_angle import field_of_view, ious
import cirtorch.layers.functional as LF
torch.manual_seed(1)

"""
PARAMS
"""
BATCH_SIZE = 500
EPOCH = 200

INPUT_DIM = 2048
HIDDEN_DIM1 = 512
HIDDEN_DIM2 = 512
HIDDEN_DIM3 = 512
OUTPUT_DIM = 2048

LR = 0.0006  # TODO: Lower Learning Rate
WD = 4e-3

FULL_DATASET = False
dataset_path = 'data/dataset'
network_path = 'data/exp_outputs1/mapillary_resnet50_gem_contrastive_m0.70_adam_lr1.0e-06_wd1.0e-06_nnum5_qsize2000_psize20000_bsize5_uevery5_imsize1024/model_epoch480.pth.tar'
multiscale = '[1]'
imsize = 320

USE_IOU = True
PLOT_FREQ = 10
TEST_FREQ = 10
posDistThr = 15  # TODO: Try higher range
negDistThr = 25
workers = 8
query_size =2000
pool_size = 20000

t = time.strftime("%Y-%d-%m_%H:%M:%S", time.localtime())
tensorboard = SummaryWriter(f'data/localcorrelation_runs/model_{INPUT_DIM}_{OUTPUT_DIM}_{t}')

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')
parser.add_argument('--name', default='debug', type=str, metavar='N')
parser.add_argument('--loss', default='mse_loss', type=str, metavar='N')
parser.add_argument('--lr', default=0.0006, type=float, metavar='lr')  

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
    plt.scatter(ground_truth, prediction, color="blue", alpha=0.2)
    plt.scatter(ground_truth, ground_truth, color="green", alpha=0.2)

    #x = np.linspace(0, 25, 25)
    #y = x
    #plt.plot(x, y, color = "green")

    model = linear_regression(ground_truth, prediction, mode, epoch)
    x = np.linspace(0, 1, 25)
    y = model.coef_ * x + model.intercept_
    plt.plot(x, y, color="red")

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
        self.input = torch.nn.Linear(INPUT_DIM, INPUT_DIM)
        #self.hidden1 = torch.nn.Linear(HIDDEN_DIM1, HIDDEN_DIM2)
        #self.hidden12 = torch.nn.Dropout(p=0.1)
        #self.hidden2 = torch.nn.Linear(HIDDEN_DIM2, HIDDEN_DIM3)
        #self.hidden2o = torch.nn.Dropout(p=0.2)
        self.softmax = torch.nn.Softmax(dim=0)
        self.output = torch.nn.Linear(INPUT_DIM, OUTPUT_DIM)

    def forward(self, x):
        x = F.leaky_relu(self.input(x))
        #x = F.leaky_relu(self.hidden1(x))
        #x = F.leaky_relu(self.hidden2(x))
        #x = self.hidden12(x)
        #x = F.leaky_relu(self.hidden2(x))
        #x = self.hidden2o(x)
        x = F.leaky_relu(self.output(x))
        #x = self.softmax(x)
        return x


"""
TRAINING
"""

def iou_distance(query, positive):
    pol = field_of_view([query, positive])
    return ious(pol[0], pol[1:])

def distance(query, positive, iou=USE_IOU):
    if iou:
        return 1.0 - iou_distance(query, positive)[0]
    return torch.norm(query[0:2]-positive[0:2])


def distances(x, label, gps, eps=1e-6):
    # x is D x N
    dim = x.size(0)  # D
    nq = torch.sum(label == -1)  # number of tuples
    S = x.size(1) // nq  # number of images per tuple including query: 1+1+n

    x1 = x[:, ::S].permute(1, 0).repeat(
        1, S-1).view((S-1)*nq, dim).permute(1, 0)
    idx = [i for i in range(len(label)) if label[i] != -1]
    x2 = x[:, idx]
    lbl = label[label != -1]

    dif = x1 - x2
    D = torch.pow(dif+eps, 2).sum(dim=0).sqrt()
    return gps, D, lbl

def contrastive(x, label, gps, eps=1e-6, margin=0.7):
    dist, D, lbl = distances(x, label, gps, eps=1e-6)
    D = D.cuda() 
    gps = gps.cuda()
    
    y = lbl*torch.pow(D, 2) + 0.5*(1-lbl)*torch.pow(torch.clamp(margin-D, min=0),2)
    y = torch.sum(y)
    return y

def mse_loss(x, label, gps, eps=1e-6, margin=0.7):
    dist, D, lbl = distances(x, label, gps, eps=1e-6)
    D = D.cuda()
    gps = gps.cuda()

    y = (1-gps)*torch.pow((D - gps), 2) + 0.5*gps*torch.pow(torch.clamp(margin-D, min=0),2)
    y = torch.sum(y)
    return y

def hubert_loss(x, label, gps, eps=1e-6, margin=0.7, delta=0.5):
    dist, D, lbl = distances(x, label, gps, eps=1e-6)
    delta_tensor = torch.empty(D.size()).fill_(delta).cuda()
    hubert_cond = torch.where((dist - D) <= delta_tensor, torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())
    y = hubert_cond * (gps*torch.pow((dist - D), 2)) + (1-hubert_cond) * (gps*torch.abs(dist - D) - 1/2 * delta**2)
    y += 0.5*(1-gps)*torch.pow(torch.clamp(margin-D, min=0), 2)
    y = torch.sum(y)
    return y

def logistic_regression(x, label, gps, eps=1e-6, margin=posDistThr):
    dist, D, _ = distances(x, label, gps, eps=1e-6)
    half = torch.tensor(0.5).cuda()
    lbl = torch.where(gps <= half, torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())
    
    m = gps.size()[0]
    return (1/m)* torch.sum(-lbl*torch.log(D) - (1 - lbl)* torch.log(1-D))

def binary_classifier(x, label, gps, eps=1e-6, margin=posDistThr):
    dist, D, lbl = distances(x, label, gps, eps=1e-6)
    half = torch.tensor(0.5).cuda()

    binary_cond = torch.where(gps <= half, torch.tensor(0.0).cuda(), torch.tensor(1.0).cuda())
    y = torch.abs(D - binary_cond)

    y = torch.sum(y)
    return y

def dump_data(place_model, correlation_model, loader, epoch):
    place_model.eval()
    correlation_model.eval()

    #avg_neg_distance = val_loader.dataset.create_epoch_tuples(place_model)
    score = 0
    for i, (input, target, gps_info) in enumerate(loader):
        nq = len(input)  # number of training tuples
        ni = len(input[0])  # number of images per tuple
        gps_info = torch.tensor(gps_info)

        dist_lat = np.zeros(nq)
        dist_gps = np.zeros(nq)
        images = []

        for q in range(nq):
            output = torch.zeros(OUTPUT_DIM, ni).cuda()
            for imi in range(ni):
                # compute output vector for image imi
                output[:, imi] = correlation_model(
                    place_model(input[q][imi].cuda()).squeeze())
            loss = mse_loss(output, target[q].cuda(), gps_info[q].cuda())
            score += loss

            dist, D, lbl = distances(output, target[q].cuda(), gps_info[q])
            D = D.cpu()
            dist_lat[q] = gps_info[q][0]
            dist_gps[q] = dist[0]

            #q = loader.qImages[loader.qidxs[i]]
            # p = loader.dbImages[loader.pidxs[i]][0] #TODO: Revert GetItem Randomness for this to work
            # images.append([q,p])

        del output
        break
    np.savetxt(f'plots/gps_{epoch}', dist_gps, delimiter=",")
    np.savetxt(f'plots/embedding_{epoch}', dist_lat, delimiter=",")
    # with open(f'plots/pictures_{epoch}.csv', "w") as f:
    #writer = csv.writer(f, dialect='excel')
    # writer.writerows(images)


def test_correlation(correlation_model, criterion, epoch):
    qvecs_test = torch.from_numpy(np.loadtxt(
        f'{dataset_path}/val/qvecs.txt', delimiter=','))
    poolvecs_test = torch.from_numpy(np.loadtxt(
        f'{dataset_path}/val/poolvecs.txt', delimiter=','))

    qpool_test = torch.from_numpy(np.loadtxt(
        f'{dataset_path}/val/qpool.txt', delimiter=','))
    ppool_test = torch.from_numpy(np.loadtxt(
        f'{dataset_path}/val/ppool.txt', delimiter=','))

    qcoordinates_test = torch.from_numpy(np.loadtxt(
        f'{dataset_path}/val/qcoordinates.txt', delimiter=','))
    pcoordinates_test = torch.from_numpy(np.loadtxt(
        f'{dataset_path}/val/dbcoordinates.txt', delimiter=','))

    # to cuda
    qvecs_test = qvecs_test.cuda()
    poolvecs_test = poolvecs_test.cuda()

    # eval mode
    correlation_model.eval()

    dist_lat = []
    dist_gps = []
    epoch_loss = 0
    for i in range(len(qpool_test)):
        q = int(qpool_test[i])
        positives = ppool_test[i][ppool_test[i] != -1]

        target = torch.ones(1+len(positives))
        target[0] = -1

        output = torch.zeros((OUTPUT_DIM, 1+len(positives))).cuda()
        gps_out = torch.ones(len(positives))

        output[:, 0] = correlation_model(qvecs_test[:, i].float())
        q_utm = qcoordinates_test[q]

        for i, p in enumerate(positives):
            output[:, i + 1] = correlation_model(poolvecs_test[:, int(p)].float()).cuda()
            gps_out[i] = distance(q_utm, pcoordinates_test[int(p)])

        loss = criterion(output, target.cuda(), gps_out.cuda())
        epoch_loss += loss

        _, D, _ = distances(output, target, gps_out)
        D = D.cpu()
        dist_lat.extend(D.tolist())
        dist_gps.extend(gps_out.tolist())

    plot_points(np.array(dist_gps), np.array(dist_lat), 'Test', epoch)
    tensorboard.add_scalar('Loss/validation', epoch_loss, epoch)


def log_tuple(input, batchid, gps_info):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = input[0] * std + mean
    distance_string = ''
    for i, image_tensor in enumerate(input[1:]):
        new_image = image_tensor * std + mean
        images = torch.cat([images, new_image], dim=0)
        distance_string += '_' + str(round(gps_info[i].item(), 1))
    tensorboard.add_images('Batch_{}{}'.format(
        batchid, distance_string), images, 0)


# Train loop
def train(train_loader, place_net, correlation_model, criterion, optimizer, scheduler, epoch):
    # train mode
    place_net.eval()
    correlation_model.train()

    RANDOM_TUPLE = random.randint(0, 100)
    avg_neg_distance = train_loader.dataset.create_epoch_tuples(place_net)
    tensorboard.add_scalar('Loss/AvgNegDistanceTrain', avg_neg_distance, epoch)

    dist_lat = []
    dist_gps = []
    epoch_loss = 0

    acc_forward_pass_time = 0
    for i, (input, target, gps_info) in enumerate(train_loader):       
        nq = len(input) # number of training tuples
        ni = len(input[0]) # number of images per tuple
        dist_lat = np.zeros(nq)
        dist_gps = np.zeros(nq)
        log_image = random.randint(0,nq - 1)

        
        for q in range(nq):
            output = torch.zeros(OUTPUT_DIM, ni).cuda()
            forward_pass_time = time.time()
            for imi in range(ni):
                # compute output vector for image imi
                #x = correlation_model(place_net(input[q][imi].cuda()).squeeze())
                #output[:, imi] = x / torch.norm(x) #LF.l2n(correlation_model(place_model(input[q][imi].cuda()).squeeze()))
                output[:, imi] = correlation_model(place_net(input[q][imi].cuda()).squeeze())
            acc_forward_pass_time += time.time() - forward_pass_time

            gps_out = torch.tensor(gps_info[q])
            loss = criterion(output, target[q].cuda(), gps_out)
            loss.backward()
            epoch_loss += loss

    tensorboard.add_scalar('Loss/train', epoch_loss, epoch)
    tensorboard.add_scalar('Timing/forward_pass_time', acc_forward_pass_time, epoch)

    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    del output

def validation(val_loader, place_net, correlation_model, criterion, epoch):
    # train mode
    place_net.eval()
    correlation_model.eval()

    avg_neg_distance = val_loader.dataset.create_epoch_tuples(place_net)
    tensorboard.add_scalar('Loss/AvgNegDistanceVal', avg_neg_distance, epoch)

    epoch_loss = 0
    acc_forward_pass_time = 0
    for i, (input, target, gps_info) in enumerate(val_loader):       
        nq = len(input) # number of training tuples
        ni = len(input[0]) # number of images per tuple

        for q in range(nq):
            output = torch.zeros(OUTPUT_DIM, ni).cuda()
            forward_pass_time = time.time()
            for imi in range(ni):
                # compute output vector for image imi
                #x = correlation_model(place_net(input[q][imi].cuda()).squeeze())
                #output[:, imi] = x / torch.norm(x) #LF.l2n(correlation_model(place_model(input[q][imi].cuda()).squeeze()))
                output[:, imi] = correlation_model(place_net(input[q][imi].cuda()).squeeze())
            acc_forward_pass_time += time.time() - forward_pass_time

            gps_out = torch.tensor(gps_info[q])
            loss = criterion(output, target[q].cuda(), gps_out)
            epoch_loss += loss

    tensorboard.add_scalar('Loss/validation', epoch_loss, epoch)
    tensorboard.add_scalar('Timing/forward_pass_time_val', acc_forward_pass_time, epoch)
    
    del output

def test(place_net, correlation_model):

    print('>> Evaluating network on test datasets...')

    # for testing we use image size of max 1024
    imsize = 1024

    posDistThr = 25
    negDistThr = 25

    # moving network to gpu and eval mode
    place_net.cuda()
    place_net.eval()

    correlation_model.cuda()
    correlation_model.eval()
    # set up the transform
    resize = transforms.Resize((240,320), interpolation=2)
    # Get transformer for dataset
    normalize = transforms.Normalize(mean=place_net.meta['mean'], std=place_net.meta['std'])
    resize = transforms.Resize((int(imsize * 3/4), imsize), interpolation=2)

    transform = transforms.Compose([
        resize,
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = TuplesDataset(
        name='mapillary+',
        mode='val', # Test on validation set during training
        imsize=imsize,
        nnum=5,
        qsize=query_size,
        poolsize=pool_size,
        transform=transform,
        posDistThr=posDistThr,
        negDistThr=negDistThr,
        root_dir = 'data'
    )

    qidxs, pidxs = test_dataset.get_loaders()

    opt = {'batch_size': 1, 'shuffle': False, 'num_workers': 8, 'pin_memory': True}

    # evaluate on test datasets
    datasets = ['mapillary'] #args.test_datasets.split(',')
    for dataset in datasets: 
        start = time.time()

        print('>> {}: Extracting...'.format(dataset))
        
        # Step 1: Extract Database Images - dbLoader
        print('>> {}: Extracting Database Images...'.format(dataset))
        dbLoader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=[test_dataset.dbImages[i] for i in range(len(test_dataset.dbImages))], imsize=imsize, transform=transform),
            **opt)        
        
        poolvecs = torch.zeros(OUTPUT_DIM, len(test_dataset.dbImages)).cuda()
        for i, input in enumerate(dbLoader):
            poolvecs[:, i] = correlation_model(place_net(input.cuda()).data.squeeze())

        # Step 2: Extract Query Images - qLoader
        print('>> {}: Extracting Query Images...'.format(dataset))
        qLoader = torch.utils.data.DataLoader(
                ImagesFromList(root='', images=[test_dataset.qImages[i] for i in qidxs], imsize=imsize, transform=transform),
                **opt)

        qvecs = torch.zeros(OUTPUT_DIM, len(qidxs)).cuda()
        for i, input in enumerate(qLoader):
            qvecs[:, i] = correlation_model(place_net(input.cuda()).data.squeeze())

        # Step 3: Ranks 
        scores = torch.mm(poolvecs.t(), qvecs)
        scores, ranks = torch.sort(scores, dim=0, descending=True) #Dim1? 
        ranks = ranks.cpu().numpy()
        ranks = np.transpose(ranks)

        scores = scores.cpu().numpy()
        scores = np.transpose(scores)

        print('>> {}: Computing Recall and Map'.format(dataset))
        k = 5
        ks = [5]
        mean_ap = mapk(ranks, pidxs, k)
        rec = recall(ranks, pidxs, ks)

        print('>> Achieved mAP: {} and Recall {}'.format(mean_ap, rec))
        return mean_ap, rec

def main():
    args = parser.parse_args()
    # Load Networks
    place_net = load_placereg_net()
    net = CorrelationNet()

    # Move to GPU
    place_net = place_net.cuda()
    net = net.cuda()

    # Get transformer for dataset
    normalize = transforms.Normalize(mean=place_net.meta['mean'], std=place_net.meta['std'])
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
        nnum=5,
        qsize=query_size,
        poolsize=pool_size,
        transform=transform,
        posDistThr=posDistThr,
        negDistThr=negDistThr, 
        root_dir = 'data',
        tuple_mining='default',
        ###cities='debug'
    )
    
    val_dataset = TuplesDataset(
            name='mapillary',
            mode='val',
            imsize=imsize,
            nnum=5,
            qsize=float('Inf'),
            poolsize=float('Inf'),
            transform=transform,
            posDistThr=posDistThr, # Use 25 meters for both pos and neg
            negDistThr=negDistThr,
            root_dir = 'data',
            tuple_mining='default'
    )
    

    # Dataloaders
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
    

    if args.loss == 'hubert_loss':
        criterion = hubert_loss 
    elif args.loss == 'mse_loss':
        criterion = mse_loss
    elif args.loss == 'logistic':
        criterion = logistic_regression
    elif args.loss == 'binary':
        criterion = binary_classifier
    elif args.loss == 'contrastive':
        criterion = contrastive

    LR = float(args.lr)

    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=math.exp(-0.01))
    
    # Train loop
    losses = np.zeros(EPOCH)
    for epoch in range(EPOCH):
        epoch_start_time = time.time()
        print(f'====> {epoch}/{EPOCH}')

        train(train_loader, place_net, net, criterion, optimizer, scheduler, epoch)
        tensorboard.add_scalar('Timing/train_epoch', time.time() - epoch_start_time, epoch)

        
        if (epoch % TEST_FREQ == 0 or (epoch == (EPOCH-1))):
            with torch.no_grad():
                validation(val_loader, place_net, net, criterion, epoch)
                test_correlation(net, criterion, epoch)
                test(place_net, net)
                tensorboard.add_scalar('Timing/test_epoch', time.time() - epoch_start_time, epoch)
            
            if args.name != 'debug':
                torch.save(net.state_dict(), f'data/localnet_exp/{args.name}/model_epoch_{epoch}.pth')

if __name__ == '__main__':
    main()
