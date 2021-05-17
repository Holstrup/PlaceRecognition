import argparse
import os
import time
import pickle
import pdb
import math
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.model_zoo import load_url
from torchvision import transforms

from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from cirtorch.datasets.datahelpers import cid2filename
from cirtorch.datasets.traindataset import TuplesDataset
from cirtorch.datasets.testdataset import configdataset
from cirtorch.datasets.genericdataset import ImagesFromList
from cirtorch.utils.download import download_train, download_test
from cirtorch.utils.whiten import whitenlearn, whitenapply
from cirtorch.utils.evaluate import mapk, recall
from cirtorch.utils.general import get_data_root, htime

PRETRAINED = {
    'retrievalSfM120k-vgg16-gem'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-vgg16-gem-b4dcdc6.pth',
    'retrievalSfM120k-resnet101-gem'    : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth',
    # new networks with whitening learned end-to-end
    'rSfM120k-tl-resnet50-gem-w'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet50-gem-w-97bf910.pth',
    'rSfM120k-tl-resnet101-gem-w'       : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
    'rSfM120k-tl-resnet152-gem-w'       : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet152-gem-w-f39cada.pth',
    'gl18-tl-resnet50-gem-w'            : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet50-gem-w-83fdc30.pth',
    'gl18-tl-resnet101-gem-w'           : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth',
    'gl18-tl-resnet152-gem-w'           : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet152-gem-w-21278d5.pth',
}

datasets_names = ['mapillary']
whitening_names = []

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Testing')

# network
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--network-path', '-npath', metavar='NETWORK',
                    help="pretrained network or network path (destination where network is saved)")
group.add_argument('--network-offtheshelf', '-noff', metavar='NETWORK',
                    help="off-the-shelf network, in the format 'ARCHITECTURE-POOLING' or 'ARCHITECTURE-POOLING-{reg-lwhiten-whiten}'," + 
                        " examples: 'resnet101-gem' | 'resnet101-gem-reg' | 'resnet101-gem-whiten' | 'resnet101-gem-lwhiten' | 'resnet101-gem-reg-whiten'")

# test options
parser.add_argument('--datasets', '-d', metavar='DATASETS', default='oxford5k,paris6k',
                    help="comma separated list of test datasets: " + 
                        " | ".join(datasets_names) + 
                        " (default: 'oxford5k,paris6k')")
parser.add_argument('--image-size', default=1024, type=int, metavar='N',
                    help='maximum size of longer image side used for training (default: 1024)')
parser.add_argument('--multiscale', '-ms', metavar='MULTISCALE', default='[1]', 
                    help="use multiscale vectors for testing, " + 
                    " examples: '[1]' | '[1, 1/2**(1/2), 1/2]' | '[1, 2**(1/2), 1/2**(1/2)]' (default: '[1]')")
parser.add_argument('--whitening', '-w', metavar='WHITENING', default=None, choices=whitening_names,
                    help="dataset used to learn whitening for testing: " + 
                        " | ".join(whitening_names) +
                        " (default: None)")
parser.add_argument('--generate-plot', default=False, type=bool, metavar='PLOT',
                    help='Generates a plot over embedding distance and geographical distance')

# GPU ID
parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                    help="gpu id used for testing (default: '0')")

visualisation = {}

def hook_fn(m, i, o):
  visualisation[m] = o 

def get_all_layers(net):
  for name, layer in net._modules.items():
    #If it is a sequential, don't register a hook on it
    # but recursively register hook on all it's module children
    if isinstance(layer, torch.nn.Sequential):
      get_all_layers(layer)
    else:
      # it's a non sequential. Register a hook
      layer.register_forward_hook(hook_fn)

def main():
    args = parser.parse_args()
    imsize = args.image_size

    # check if there are unknown datasets
    for dataset in args.datasets.split(','):
        if dataset not in datasets_names:
            raise ValueError('Unsupported or unknown dataset: {}!'.format(dataset))

    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # loading network from path
    if args.network_path is not None:

        print(">> Loading network:\n>>>> '{}'".format(args.network_path))
        if args.network_path in PRETRAINED:
            # pretrained networks (downloaded automatically)
            state = load_url(PRETRAINED[args.network_path], model_dir=os.path.join(get_data_root(), 'networks'))
        else:
            # fine-tuned network from path
            state = torch.load(args.network_path)

        # parsing net params from meta
        # architecture, pooling, mean, std required
        # the rest has default values, in case that is doesnt exist
        net_params = {}
        net_params['architecture'] = state['meta']['architecture']
        net_params['pooling'] = state['meta']['pooling']
        net_params['local_whitening'] = state['meta'].get('local_whitening', False)
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

    # loading offtheshelf network
    elif args.network_offtheshelf is not None:
        
        # parse off-the-shelf parameters
        offtheshelf = args.network_offtheshelf.split('-')
        net_params = {}
        net_params['architecture'] = offtheshelf[0]
        net_params['pooling'] = offtheshelf[1]
        net_params['local_whitening'] = 'lwhiten' in offtheshelf[2:]
        net_params['regional'] = 'reg' in offtheshelf[2:]
        net_params['whitening'] = 'whiten' in offtheshelf[2:]
        net_params['pretrained'] = True

        # load off-the-shelf network
        print(">> Loading off-the-shelf network:\n>>>> '{}'".format(args.network_offtheshelf))
        net = init_network(net_params)
        print(">>>> loaded network: ")
        print(net.meta_repr())

    # setting up the multi-scale parameters
    ms = list(eval(args.multiscale))
    if len(ms)>1 and net.meta['pooling'] == 'gem' and not net.meta['regional'] and not net.meta['whitening']:
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
    resize = transforms.Resize((240,320), interpolation=2)
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
        mode='test',
        #mode='train',
        imsize=imsize,
        transform=transform,
        posDistThr=posDistThr,
        negDistThr=negDistThr,
        tuple_mining='gps',
        cities='debug'
    )
    qidxs, pidxs = test_dataset.get_loaders()

    opt = {'batch_size': 1, 'shuffle': False, 'num_workers': 8, 'pin_memory': True}

    # evaluate on test datasets
    datasets = datasets_names
    for dataset in datasets: 
        start = time.time()
        
        print('>> {}: Extracting...'.format(dataset))
        
        # Step 1: Extract Database Images - dbLoader
        print('>> {}: Extracting Database Images...'.format(dataset))
        dbLoader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=[test_dataset.dbImages[i] for i in range(len(test_dataset.dbImages))], imsize=imsize, transform=transform),
            **opt)        
        poolvecs = torch.zeros(net.meta['outputdim'], len(test_dataset.dbImages)).cuda()
        for i, input in enumerate(dbLoader):
            poolvecs[:, i] = net(input.cuda()).data.squeeze()

        # Step 2: Extract Query Images - qLoader
        print('>> {}: Extracting Query Images...'.format(dataset))
        qLoader = torch.utils.data.DataLoader(
                ImagesFromList(root='', images=[test_dataset.qImages[i] for i in qidxs], imsize=imsize, transform=transform),
                **opt)
        

        qvecs = torch.zeros(net.meta['outputdim'], len(qidxs)).cuda()
        for i, input in enumerate(qLoader):
            qvecs[:, i] = net(input.cuda()).data.squeeze()
        
        # GPS: get query and pool coordinates
        querycoordinates = torch.tensor([test_dataset.gpsInfo[test_dataset.qImages[i][-26:-4]] for i in qidxs], dtype=torch.float)
        poolcoordinates = torch.tensor([test_dataset.gpsInfo[test_dataset.dbImages[i][-26:-4]] for i in range(len(test_dataset.dbImages))], dtype=torch.float)

        # GPS: Compute distances
        distances = torch.norm(querycoordinates[:, None] - poolcoordinates, dim=2)
            
        # GPS: Sort distances
        distances, indicies = torch.sort(distances, dim=1, descending=False)
        
        # Step 3: Ranks 
        scores = torch.mm(poolvecs.t(), qvecs)
        #scores, ranks = torch.sort(scores, dim=0, descending=True) # Euclidan distance is 1 - Score 
        
        #ranks = ranks.cpu().numpy()
        #ranks = np.transpose(ranks)

        scores = scores.cpu().numpy()
        scores = np.transpose(scores)

        if not args.generate_plot:
            print('>>> {}: Generating Correlation Data'.format(dataset))
            gpsinfo = test_dataset.gpsInfo
            angleInfo = test_dataset.angleInfo
            all_gps = np.zeros((len(qidxs),10))
            all_emb = np.zeros((len(qidxs),10))
            all_ang = np.zeros((len(qidxs),10))
            all_pics = []
            for q in range(len(qidxs)):
                positive = 0
                gps = []
                emb = []
                pictures = [test_dataset.qImages[qidxs[q]].split('/')[-1][:-4]]
                angles = []
                while distances[q, positive] < 50 and positive < 10:
                    index = indicies[q, positive]
                    emb.append(scores[q, index].item())
                    gps.append(distances[q, positive])
                    pictures.append(test_dataset.dbImages[index])
                    
                    key = test_dataset.dbImages[index].split('/')[-1][:-4]
                    angles.append(angleInfo[key])
                    
                    positive += 1
                emb = np.array(emb)
                gps = np.array(gps)
                all_gps[q, :min(10, len(gps))] = gps
                all_emb[q, :min(10, len(emb))] = emb
                #all_ang[q, :min(10, len(emb))] = angles
                all_pics.append(pictures)

             
            output_plot(all_gps, all_emb)
            np.savetxt("plots/gps.csv", all_gps, delimiter=",")
            np.savetxt("plots/embedding.csv", all_emb, delimiter=",")
            np.savetxt("plots/angles.csv", all_ang, delimiter=",")
            with open("plots/pictures.csv", "w") as f:
                writer = csv.writer(f, dialect='excel')
                writer.writerows(all_pics)
        
        elif args.generate_plot:
            print('>>> {}: Generating Distance Plot'.format(dataset))
            gpsinfo = test_dataset.gpsInfo
            angleInfo = test_dataset.angleInfo

            scores = np.ones(np.shape(scores)) - scores
            plt.imshow(scores, interpolation='nearest')
            plt.colorbar()
            plt.savefig('plots/q_scores_heatmap')
            plt.clf() 
            
            scores = torch.mm(poolvecs.t(), poolvecs)
            scores = scores.cpu().numpy()
            scores = np.ones(np.shape(scores)) - scores
            plt.imshow(scores, interpolation='nearest')
            plt.colorbar()
            plt.savefig('plots/pool_scores_heatmap')
            plt.clf()

            distances = torch.norm(querycoordinates[:, None] - poolcoordinates, dim=2)
            plt.imshow(distances, interpolation='nearest')
            plt.colorbar()
            plt.savefig('plots/gps_distances_heatmap')
            plt.clf() 

            indicator = torch.where(distances > 25, torch.zeros(distances.size()), torch.ones(distances.size()))
            plt.imshow(indicator, interpolation='nearest')
            plt.colorbar()
            plt.savefig('plots/gps_indicator')
            plt.clf() 
            

            """
            distance_matrix = np.zeros((len(qidxs), 50))
            gps_matrix = np.zeros((max(50, len(qidxs)), 50))
            for q in range(max(50, len(qidxs))):
                for i in range(50):
                    index = indicies[q, i]
                    gps_matrix[q,i] = distances[q, i]
                    distance_matrix[q, i] = scores[q, index].item()
                
            plt.imshow(distance_matrix, interpolation='nearest')
            plt.colorbar()
            plt.savefig('plots/distance_matrix')
            """
            

        print('>> {}: elapsed time: {}'.format(dataset, htime(time.time()-start)))

    

def output_plot(gps, emb):
    gps_flattened = gps.flatten()
    emb = 1-emb
    emb__flattened = emb.flatten()

    fig, axs = plt.subplots(1, 1, figsize=(20,10))
    axs.scatter(gps[gps != 0], emb[gps != 0], color='b')
    axs.set_xlim([0, 25])
    axs.set_ylim([0.0, 1.0])
    axs.set_title('Aggregation of distances')
    fig.savefig('plots/correlation_all_points')

def distance(query, positive):
    return np.linalg.norm(np.array(query)-np.array(positive))

if __name__ == '__main__':
    main()
