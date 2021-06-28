import argparse
import os
import time
import pickle
import pdb

import numpy as np
import sklearn

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
from cirtorch.networks.localcorrelationnetv2 import CorrelationNet

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
parser.add_argument('--whitening-network', default=None, type=str) 
parser.add_argument('--whitening-outdim', default=128, type=int) 

# test options
parser.add_argument('--datasets', '-d', metavar='DATASETS', default='mapillary',
                    help="comma separated list of test datasets: " + 
                        " | ".join(datasets_names) + 
                        " (default: 'mapillary')")
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

parser.add_argument('--pca', default=False, type=bool, metavar='Whitening_PCA')

# GPU ID
parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                    help="gpu id used for testing (default: '0')")

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

    OUTPUT_DIM = net.meta['outputdim']

    if args.whitening_network:
        state = torch.load(args.whitening_network) 
        whitening_net = CorrelationNet()
        whitening_net.load_state_dict(state) 
        whitening_net.cuda()
        whitening_net.eval()

        #whitening_net = torch.load(args.whitening_network)
        #whitening_net.cuda()
        #whitening_net.eval()

        OUTPUT_DIM = args.whitening_outdim
    
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
        imsize=imsize,
        transform=transform,
        posDistThr=posDistThr,
        negDistThr=negDistThr,
    )
    qidxs, pidxs = test_dataset.get_loaders()
    print('1', len(qidxs))
    #test_dataset.filter_positive_loader()
    #print('2', len(qidxs))
    #qidxs, pidxs = test_dataset.get_loaders()
    #print('3', len(qidxs))
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
        poolvecs = torch.zeros(OUTPUT_DIM, len(test_dataset.dbImages)).cuda()
        for i, input in enumerate(dbLoader):
            if args.whitening_network:
                poolvecs[:, i] = whitening_net(net(input.cuda()).data.squeeze())
            else:
                poolvecs[:, i] = net(input.cuda()).data.squeeze()

        # Step 2: Extract Query Images - qLoader
        print('>> {}: Extracting Query Images...'.format(dataset))
        qLoader = torch.utils.data.DataLoader(
                ImagesFromList(root='', images=[test_dataset.qImages[i] for i in qidxs], imsize=imsize, transform=transform),
                **opt)

        qvecs = torch.zeros(OUTPUT_DIM, len(qidxs)).cuda()
        for i, input in enumerate(qLoader):
            if args.whitening_network:
                qvecs[:, i] = whitening_net(net(input.cuda()).data.squeeze())
            else:
                qvecs[:, i] = net(input.cuda()).data.squeeze()
        
        if args.pca:
            poolvecs = poolvecs.cpu()
            qvecs = qvecs.cpu()

            pca = sklearn.decomposition.PCA(n_components=2048)
            pca.fit(poolvecs.T)

            poolvecs = pca.transform(poolvecs.T)
            qvecs = pca.transform(qvecs.T)
            
            poolvecs = torch.from_numpy(poolvecs).cuda()
            qvecs = torch.from_numpy(qvecs).cuda()

            poolvecs = poolvecs.T
            qvecs = qvecs.T

        # Step 3: Ranks 
        scores = torch.mm(poolvecs.t(), qvecs)
        scores, ranks = torch.sort(scores, dim=0, descending=True) # Euclidan distance is 1 - Score 
        
        ranks = ranks.cpu().numpy()
        ranks = np.transpose(ranks)

        
        scores = scores.detach().cpu().numpy()
        #scores = scores.cpu().numpy()
        scores = np.transpose(scores)

        if args.generate_plot:
            print('>>> {}: Generating Plot'.format(dataset))
            gpsinfo = test_dataset.gpsInfo
            k = 5
            gpsdistances = np.zeros((ranks.shape[0], k))
            embeddingdistances = np.zeros((ranks.shape[0],k))
            for qidx in range(ranks.shape[0]):
                points = ranks[qidx,:k]
                q = test_dataset.qImages[qidx].split('/')[-1][:-4]
                qcoor = gpsinfo[q]
                ps = [test_dataset.dbImages[i].split('/')[-1][:-4] for i in points]
                for i in range(len(ps)):
                    ps[i] = distance(qcoor, gpsinfo[ps[i]])

                ps = np.array(ps)
                embed = np.array(scores[qidx,:k])
                gpsdistances[qidx,:] = ps
                embeddingdistances[qidx,:] = embed

            print(gpsdistances)
            np.savetxt("gps.csv", gpsdistances, delimiter=",")
            np.savetxt("embedding.csv", embeddingdistances, delimiter=",")
        print('>> {}: Computing Recall and Map'.format(dataset))
        k = 5
        ks = [1, 5, 10]
        mean_ap = mapk(ranks, pidxs, k)
        rec = recall(ranks, pidxs, ks)
        
        print('>> Recall (1,5,10): {}'.format(rec))
        print('>> mAP 5: {}'.format(mean_ap))
        print('>> {}: elapsed time: {}'.format(dataset, htime(time.time()-start)))

def distance(query, positive):
    return np.linalg.norm(np.array(query)-np.array(positive))

if __name__ == '__main__':
    main()
