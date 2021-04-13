import torch
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

from sklearn.linear_model import LinearRegression
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
from cirtorch.datasets.datahelpers import collate_tuples, cid2filename

torch.manual_seed(1)

"""
PARAMS
"""
INPUT_DIM = 2048
HIDDEN_DIM1 = 1024
HIDDEN_DIM2 = 512
HIDDEN_DIM3 = 256
OUTPUT_DIM = 128  # TODO: Is this right?

datasets_names = ['mapillary']
place_model_path = 'data/exp_outputs1/mapillary_resnet50_gem_contrastive_m0.70_adam_lr1.0e-06_wd1.0e-06_nnum5_qsize2000_psize20000_bsize5_uevery5_imsize1024/model_epoch38.pth.tar'
correlation_model_path = 'data/localcorrelationnet/model_'
multiscale = '[1]'
imsize = 320

posDistThr = 25
negDistThr = 25
workers = 8
query_size = 2000
pool_size = 20000

t = time.strftime("%Y-%d-%m_%H:%M:%S", time.localtime())

"""
Network
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


def load_correlationnet(model_path):
    correlationnet = CorrelationNet()
    correlationnet.load_state_dict(torch.load(model_path))
    correlationnet.eval()
    return correlationnet


placenet = load_placereg_net(place_model_path)
correlationnet = load_correlationnet(correlation_model_path)

# moving network to gpu and eval mode
placenet.cuda()
correlationnet.cuda()
placenet.eval()
correlationnet.eval()


# set up the transform
resize = transforms.Resize((240, 320), interpolation=2)
normalize = transforms.Normalize(
    mean=placenet.meta['mean'],
    std=placenet.meta['std']
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

opt = {'batch_size': 1, 'shuffle': False, 'num_workers': 8, 'pin_memory': True}

start = time.time()

# Step 1: Extract Database Images - dbLoader
print('>> {}: Extracting Database Images...')
dbLoader = torch.utils.data.DataLoader(
    ImagesFromList(root='', images=[test_dataset.dbImages[i] for i in range(
        len(test_dataset.dbImages))], imsize=imsize, transform=transform),
    **opt)

poolvecs = torch.zeros(OUTPUT_DIM, len(test_dataset.dbImages)).cuda()

for i, input in enumerate(dbLoader):
    poolvecs[:, i] = correlationnet(placenet(input.cuda()).data.squeeze())

# Step 2: Extract Query Images - qLoader
print('>> {}: Extracting Query Images...')
qLoader = torch.utils.data.DataLoader(
    ImagesFromList(root='', images=[
        test_dataset.qImages[i] for i in qidxs], imsize=imsize, transform=transform),
    **opt)

qvecs = torch.zeros(OUTPUT_DIM, len(qidxs)).cuda()

for i, input in enumerate(qLoader):
    qvecs[:, i] = correlationnet(placenet(input.cuda()).data.squeeze())

# Step 3: Ranks
scores = torch.mm(poolvecs.t(), qvecs)
scores = scores.cpu().numpy()
scores = np.transpose(scores)

# GPS: get query and pool coordinates
querycoordinates = torch.tensor(
    [test_dataset.gpsInfo[test_dataset.qImages[i][-26:-4]] for i in qidxs], dtype=torch.float)
poolcoordinates = torch.tensor([test_dataset.gpsInfo[test_dataset.dbImages[i][-26:-4]]
                                for i in range(len(test_dataset.dbImages))], dtype=torch.float)

# GPS: Compute distances
distances = torch.norm(querycoordinates[:, None] - poolcoordinates, dim=2)

# GPS: Sort distances
distances, indicies = torch.sort(distances, dim=1, descending=False)


print('>>> {}: Generating Correlation Data'.format(dataset))
gpsinfo = test_dataset.gpsInfo
angleInfo = test_dataset.angleInfo
all_gps = np.zeros((len(qidxs), 10))
all_emb = np.zeros((len(qidxs), 10))
all_ang = np.zeros((len(qidxs), 10))
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
    all_pics.append(pictures)

output_plot(all_gps, all_emb)
np.savetxt("plots/gps.csv", all_gps, delimiter=",")
np.savetxt("plots/embedding.csv", all_emb, delimiter=",")
np.savetxt("plots/angles.csv", all_ang, delimiter=",")
with open("plots/pictures.csv", "w") as f:
    writer = csv.writer(f, dialect='excel')
    writer.writerows(all_pics)
