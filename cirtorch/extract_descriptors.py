import sys
sys.path.insert(0, "/Users/alexanderholstrup/git/VisualPlaceRecognition/cnnimageretrieval-pytorch")

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from cirtorch.datasets.datahelpers import imresize, default_loader
from cirtorch.datasets.traindataset import TuplesDataset
from cirtorch.datasets.genericdataset import ImagesFromList

dataset = 'train'
root_path = f'data/dataset/{dataset}'
posDistThr = 25
network_path = 'data/exp_outputs1/mapillary_resnet50_gem_contrastive_m0.70_adam_lr1.0e-06_wd1.0e-06_nnum5_qsize2000_psize20000_bsize5_uevery5_imsize1024/model_epoch480.pth.tar'
multiscale = '[1]'
def load_placereg_net():
    # loading network from path
    if network_path is not None:
        state = torch.load(network_path, map_location=torch.device('cpu'))

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

net = load_placereg_net()
net.cuda()
net.eval()
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

normalize = transforms.Normalize(mean=net.meta['mean'], std=net.meta['std'])
imsize = 320
resize = transforms.Resize((int(imsize * 3/4), imsize), interpolation=2)

transform = transforms.Compose([
        resize,
        transforms.ToTensor(),
        normalize,
])

train_dataset = TuplesDataset(
        name='mapillary',
        mode=dataset,
        imsize=imsize,
        nnum=0,
        qsize=float('Inf'),
        poolsize=float('Inf'),
        transform=transform,
        posDistThr=posDistThr,
        negDistThr=posDistThr, 
        root_dir = 'data',
        cities='debug',
        tuple_mining='default'
    )
qidxs, pidxs = train_dataset.get_loaders()
opt = {'batch_size': 1, 'shuffle': False, 'num_workers': 8, 'pin_memory': True}

dbLoader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=[train_dataset.dbImages[i] for i in range(len(train_dataset.dbImages))], imsize=imsize, transform=transform),
            **opt)        

poolvecs = torch.zeros(net.meta['outputdim'], len(train_dataset.dbImages)).cuda()
for i, input in enumerate(dbLoader):
    poolvecs[:, i] = net(input.cuda()).data.squeeze()

qLoader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=[train_dataset.qImages[i] for i in train_dataset.qpool], imsize=imsize, transform=transform),
                **opt)

qvecs = torch.zeros(net.meta['outputdim'], len(qidxs)).cuda()
for i, input in enumerate(qLoader):
    qvecs[:, i] = net(input.cuda()).data.squeeze()

i = 120
q = train_dataset.qpool[i]
positives = train_dataset.ppool[i]
for pos in positives:
    print(q, pos, torch.pow((qvecs[:, q] - poolvecs[:, pos])+1e-6, 2).sum(dim=0).sqrt())

"""

poolvecs = poolvecs.cpu().detach().numpy() 
np.savetxt(f'{root_path}/poolvecs.txt', poolvecs, delimiter=',')

qvecs = qvecs.cpu().detach().numpy() 
np.savetxt(f'{root_path}/qvecs.txt', qvecs, delimiter=',')

np.savetxt(f'{root_path}/qpool.txt', train_dataset.qpool, delimiter=',')
ppool = np.zeros((len(train_dataset.ppool), 100)) - 1
for i, p in enumerate(train_dataset.ppool):
    ppool[i, 0:np.shape(p)[0]] = p

np.savetxt(f'{root_path}/ppool.txt', ppool, delimiter=',')

np.savetxt(f'{root_path}/qImages.txt', train_dataset.qImages, delimiter=',', fmt="%s")
np.savetxt(f'{root_path}/dbImages.txt', train_dataset.dbImages, delimiter=',', fmt="%s")

qcoordinates = np.array([train_dataset.gpsInfo[q[-26:-4]].extend(train_dataset.angleInfo[q[-26:-4]]) for q in train_dataset.qImages])
np.savetxt(f'{root_path}/qcoordinates.txt', qcoordinates, delimiter=',')

dbcoordinates = np.array([train_dataset.gpsInfo[q[-26:-4]].extend(train_dataset.angleInfo[q[-26:-4]]) for q in train_dataset.dbImages])
np.savetxt(f'{root_path}/dbcoordinates.txt', dbcoordinates, delimiter=',')
"""
