import argparse
import os
import time
import pickle
import pdb
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url
from torchvision import transforms, models

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

# GPU ID
parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                    help="gpu id used for testing (default: '0')")

###
class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class BaseNet(nn.Module):
    def __init__(self, backbone, global_pool=None, poolkernel=7, norm=None, p=3, num_clusters=64):
        super(BaseNet, self).__init__()
        self.backbone = backbone
        for name, param in self.backbone.named_parameters():
                n=param.size()[0]
        self.feature_length=n
        if global_pool == "max":
            self.pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        elif global_pool == "avg":
            self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        elif global_pool == "GeM":
            self.pool=GeM(p=p)
        else:
            self.pool = None
        self.norm=norm

    def forward(self, x0):
        out = self.backbone.forward(x0)
        out = self.pool.forward(out).squeeze(-1).squeeze(-1)
        if self.norm == "L2":
            out=nn.functional.normalize(out)
        return out

def get_backbone(name):
    if name == "resnet18":
        backbone = models.resnet18(pretrained=True)
    elif name == "resnet34":
        backbone = models.resnet34(pretrained=True)
    elif name == "resnet152":
        backbone = models.resnet152(pretrained=True)
    elif name == "resnet50":
        backbone = models.resnet50(pretrained=True)
    if "resnet" in name:
        backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))
    if name == "densenet161":
        backbone = models.densenet161(pretrained=True).features
    elif name == "densenet121":
        backbone = models.densenet121(pretrained=True).features
    elif name == "vgg16":
        backbone = models.vgg16(pretrained=True).features
    return backbone

def create_model(name, pool, last_layer=None, norm=None, p_gem=3, num_clusters=64, mode="siamese"):
    
    backbone = get_backbone(name)
    layers = len(list(backbone.children()))

    if last_layer is None:
        last_layer = layers
    elif "densenet" in name:
        last_layer=last_layer*2
    elif "vgg" in name:
    	last_layer=last_layer*8-2
    aux = 0
    for c in backbone.children():

        if aux < layers - last_layer:
            print(aux, c._get_name(), "IS FROZEN")
            for p in c.parameters():
                p.requires_grad = False
        else:
            print(aux, c._get_name(), "IS TRAINED")
        aux += 1
    
    return BaseNet(backbone, pool, norm=norm, p=p_gem)
###




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
        #Create model and load weights
        pool='GeM'      #params.pool
        backbone = 'resnet50'   #params.backbone
        norm = 'L2'       #params.norm
        model_file = '' #params.model_file
        net = create_model(backbone, pool, norm=norm, mode="single")
        try:
            net.load_state_dict(torch.load(model_file)["model_state_dict"])
        except:
            net.load_state_dict(torch.load(model_file)["state_dict"])
    
    net.eval()
    net.cuda()
    
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
        negDistThr=negDistThr
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

        # Step 3: Ranks 
        scores = torch.mm(poolvecs.t(), qvecs)
        scores, ranks = torch.sort(scores, dim=0, descending=True) # Euclidan distance is 1 - Score 
        
        ranks = ranks.cpu().numpy()
        ranks = np.transpose(ranks)

        scores = scores.cpu().numpy()
        scores = np.transpose(scores)

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
