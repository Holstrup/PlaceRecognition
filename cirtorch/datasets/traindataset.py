import os
import pickle
import pdb

import pandas as pd
from os.path import join
import numpy as np
import torch
import torch.utils.data as data
import sys
from sklearn.neighbors import NearestNeighbors
import random

from cirtorch.datasets.datahelpers import default_loader, imresize, cid2filename
from cirtorch.datasets.genericdataset import ImagesFromList
from cirtorch.utils.general import get_data_root

default_cities = {
    'train': ["zurich", "london", "boston", "melbourne", "amsterdam","helsinki",
              "tokyo","toronto","saopaulo","moscow","trondheim","paris","bangkok",
              "budapest","austin","berlin","ottawa","phoenix","goa","amman","nairobi","manila"],
    'val': ["cph", "sf"],
    'test': ["miami","athens","buenosaires","stockholm","bengaluru","kampala"]
}

default_cities_debug = {
    'train': ["zurich", "london"],
    'val': ["cph", "sf"],
    'test': ["miami"]
}

class TuplesDataset(data.Dataset):
    """Data loader that loads training and validation tuples of 
        Radenovic etal ECCV16: CNN image retrieval learns from BoW

    Args:
        name (string): dataset name: 'retrieval-sfm-120k'
        mode (string): 'train' or 'val' for training and validation parts of dataset
        imsize (int, Default: None): Defines the maximum size of longer image side
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        loader (callable, optional): A function to load an image given its path.
        nnum (int, Default:5): Number of negatives for a query image in a training tuple
        qsize (int, Default:1000): Number of query images, ie number of (q,p,n1,...nN) tuples, to be processed in one epoch
        poolsize (int, Default:10000): Pool size for negative images re-mining

     Attributes:
        images (list): List of full filenames for each image (qimages + dbimages)
        clusters (list): List of clusterID per image (nonNegIdx)
        qpool (list): List of all query image indexes (qidx)
        ppool (list): List of positive image indexes, each corresponding to query at the same position in qpool (pidx)

        qidxs (list): List of qsize query image indexes to be processed in an epoch (qidx without self)
        pidxs (list): List of qsize positive image indexes, each corresponding to query at the same position in qidxs (pidx without self)
        nidxs (list): List of qsize tuples of negative images
                        Each nidxs tuple contains nnum images corresponding to query image at the same position in qidxs (nidxs without self)

        Lists qidxs, pidxs, nidxs are refreshed by calling the ``create_epoch_tuples()`` method, 
            ie new q-p pairs are picked and negative images are remined
    """

    def __init__(self, name, mode='train', imsize=None, nnum=5, qsize=2000, poolsize=20000, transform=None, loader=default_loader, posDistThr=10, negDistThr=25, root_dir = 'data', cities = '', tuple_mining='default'):

        if name.startswith('mapillary'):
            # Parameters  
            task = 'im2im'
            seq_length = 1
            subtask = 'all'
            positive_sampling = True

            # initializing
            assert mode in ('train', 'val', 'test')
            assert task in ('im2im', 'im2seq', 'seq2im', 'seq2seq')
            assert subtask in ('all', 's2w', 'w2s', 'o2n', 'n2o', 'd2n', 'n2d')
            assert seq_length % 2 == 1
            assert (task == 'im2im' and seq_length == 1) or (task != 'im2im' and seq_length > 1)

            if cities == 'debug':
                self.cities = default_cities_debug[mode]
            elif cities in default_cities:
                self.cities = default_cities[cities]
            elif cities == '':
                self.cities = default_cities[mode]
            else:
                self.cities = cities.split(',')
            
            nNeg = nnum

            self.dbImages = []
            self.qImages = []
            self.images = []
            self.qpool = [] # qImages -> qidx
            self.ppool = [] # dbImages -> pidx
            self.qidxs = [] # qIdx
            self.pidxs = [] # pIdx
            self.nidxs = [] # nonNegIdx
            self.clusters = [] # nonNegIdx
            self.sideways = []
            self.night = []

            # hyper-parameters
            self.nNeg = nNeg
            self.margin = 0.1
            self.posDistThr = posDistThr
            self.negDistThr = negDistThr

            # flags
            self.cache = None
            self.exclude_panos = True
            self.mode = mode
            self.subtask = subtask

            # other
            self.transform = transform
            self.query_keys_with_no_match = []
            self.gpsInfo = {}
            self.angleInfo = {}
            self.tuple_mining = tuple_mining

            # define sequence length based on task
            if task == 'im2im':
                seq_length_q, seq_length_db = 1, 1

            # load data
            for city in self.cities:
                print("=====> {}".format(city))

                subdir = 'test' if city in default_cities['test'] else 'train_val'

                # get len of images from cities so far for indexing
                #_lenImg = len(self.images)
                _lenQ = len(self.qImages)
                _lenDb = len(self.dbImages)

                # when GPS / UTM is available
                if self.mode in ['train','val','test']:
                    # load query data
                    qData = pd.read_csv(join(root_dir, subdir, city, 'query', 'postprocessed.csv'), index_col = 0)
                    qDataRaw = pd.read_csv(join(root_dir, subdir, city, 'query', 'raw.csv'), index_col = 0)
                    self.addGpsInfo(qData, qDataRaw)

                    # load database data
                    dbData = pd.read_csv(join(root_dir, subdir, city, 'database', 'postprocessed.csv'), index_col = 0)
                    dbDataRaw = pd.read_csv(join(root_dir, subdir, city, 'database', 'raw.csv'), index_col = 0)
                    self.addGpsInfo(dbData, dbDataRaw)

                    # arange based on task
                    qSeqKeys, qSeqIdxs = self.arange_as_seq(qData, join(root_dir, subdir, city, 'query'), seq_length_q)
                    dbSeqKeys, dbSeqIdxs = self.arange_as_seq(dbData, join(root_dir, subdir, city, 'database'), seq_length_db)

                    # filter based on subtasks
                    if self.mode in ['val', 'test']:
                        qidxs = pd.read_csv(join(root_dir, subdir, city, 'query', 'subtask_index.csv'), index_col = 0)
                        dbIdx = pd.read_csv(join(root_dir, subdir, city, 'database', 'subtask_index.csv'), index_col = 0)

                        # find all the sequence where the center frame belongs to a subtask
                        val_frames = np.where(qidxs[self.subtask])[0]
                        qSeqKeys, qSeqIdxs = self.filter(qSeqKeys, qSeqIdxs, val_frames)

                        val_frames = np.where(dbIdx[self.subtask])[0]
                        dbSeqKeys, dbSeqIdxs = self.filter(dbSeqKeys, dbSeqIdxs, val_frames)
                    
                    # filter based on panorama data
                    if self.exclude_panos:
                        panos_frames = np.where((qDataRaw['pano'] == False).values)[0]
                        qSeqKeys, qSeqIdxs = self.filter(qSeqKeys, qSeqIdxs, panos_frames)

                        panos_frames = np.where((dbDataRaw['pano'] == False).values)[0]
                        dbSeqKeys, dbSeqIdxs = self.filter(dbSeqKeys, dbSeqIdxs, panos_frames)

                    unique_qSeqIdx = np.unique(qSeqIdxs)
                    unique_dbSeqIdx = np.unique(dbSeqIdxs)

                    # if a combination of city, task and subtask is chosen, where there are no query/dabase images, then continue to next city
                    if len(unique_qSeqIdx) == 0 or len(unique_dbSeqIdx) == 0: continue

                    self.qImages.extend(qSeqKeys)
                    self.dbImages.extend(dbSeqKeys)

                    qData = qData.loc[unique_qSeqIdx]
                    dbData = dbData.loc[unique_dbSeqIdx]

                    # useful indexing functions
                    seqIdx2frameIdx = lambda seqIdx, seqIdxs : seqIdxs[seqIdx]
                    frameIdx2seqIdx = lambda frameIdx, seqIdxs: np.where(seqIdxs == frameIdx)[0][1]
                    frameIdx2uniqFrameIdx = lambda frameIdx, uniqFrameIdx : np.where(np.in1d(uniqFrameIdx, frameIdx))[0]
                    uniqFrameIdx2seqIdx = lambda frameIdxs, seqIdxs : np.where(np.in1d(seqIdxs,frameIdxs).reshape(seqIdxs.shape))[0]

                    # utm coordinates
                    utmQ = qData[['easting', 'northing']].values.reshape(-1,2)
                    utmDb = dbData[['easting', 'northing']].values.reshape(-1,2)

                    # find positive images for training
                    neigh = NearestNeighbors(algorithm = 'brute')
                    neigh.fit(utmDb)
                    D, I = neigh.radius_neighbors(utmQ, self.posDistThr)

                    if mode == 'train':
                        nD, nI = neigh.radius_neighbors(utmQ, self.negDistThr)

                    night, sideways, index = qData['night'].values, (qData['view_direction'] == 'Sideways').values, qData.index

                    for q_seq_idx in range(len(qSeqKeys)):

                        q_frame_idxs = seqIdx2frameIdx(q_seq_idx, qSeqIdxs)
                        q_uniq_frame_idx = frameIdx2uniqFrameIdx(q_frame_idxs, unique_qSeqIdx)

                        p_uniq_frame_idxs = np.unique([p for pos in I[q_uniq_frame_idx] for p in pos])

                        # the query image has at least one positive
                        if len(p_uniq_frame_idxs) > 0:
                            p_seq_idx = np.unique(uniqFrameIdx2seqIdx(unique_dbSeqIdx[p_uniq_frame_idxs], dbSeqIdxs))
                            self.ppool.append(p_seq_idx + _lenDb)
                            self.qpool.append(q_seq_idx + _lenQ)

                            # in training we have two thresholds, one for finding positives and one for finding images that we are certain are negatives.
                            if self.mode == 'train':

                                n_uniq_frame_idxs = np.unique([n for nonNeg in nI[q_uniq_frame_idx] for n in nonNeg])
                                n_seq_idx = np.unique(uniqFrameIdx2seqIdx(unique_dbSeqIdx[n_uniq_frame_idxs], dbSeqIdxs))

                                self.clusters.append(n_seq_idx + _lenDb)

                                # gather meta which is useful for positive sampling
                                if sum(night[np.in1d(index, q_frame_idxs)]) > 0: self.night.append(len(self.qpool)-1)
                                if sum(sideways[np.in1d(index, q_frame_idxs)]) > 0: self.sideways.append(len(self.qpool)-1)

                        else:
                            query_key = qSeqKeys[q_seq_idx].split('/')[-1][:-4]
                            self.query_keys_with_no_match.append(query_key)
        
                # when GPS / UTM / pano info is not available    
                elif self.mode in ['test2']:

                    # load images for subtask
                    qidxs = pd.read_csv(join(root_dir, subdir, city, 'query', 'subtask_index.csv'), index_col = 0)
                    dbIdx = pd.read_csv(join(root_dir, subdir, city, 'database', 'subtask_index.csv'), index_col = 0)

                    # arange in sequences
                    qSeqKeys, qSeqIdxs = self.arange_as_seq(qidxs, join(root_dir, subdir, city, 'query'), seq_length_q)
                    dbSeqKeys, dbSeqIdxs = self.arange_as_seq(dbIdx, join(root_dir, subdir, city, 'database'), seq_length_db)

                    # filter query based on subtask
                    val_frames = np.where(qidxs[self.subtask])[0]
                    qSeqKeys, qSeqIdxs = self.filter(qSeqKeys, qSeqIdxs, val_frames)

                    # filter database based on subtask
                    val_frames = np.where(dbIdx[self.subtask])[0]
                    dbSeqKeys, dbSeqIdxs = self.filter(dbSeqKeys, dbSeqIdxs, val_frames)

                    self.qImages.extend(qSeqKeys)
                    self.dbImages.extend(dbSeqKeys)

                    self.qpool.extend(list(range(_lenQ, len(qSeqKeys) + _lenQ)))

            # if a combination of cities, task and subtask is chosen, where there are no query/database images, then exit
            if len(self.dbImages) == 0:
                print("Exiting...")
                print("A combination of cities, task and subtask have been chosen, where there are no query/database images.")
                print("Try choosing a different subtask or more cities")
                sys.exit()

            # creates self.images 
            #self.images = np.asarray(self.images) #self.qidxs + self.pidxs + self.nidxs
            self.dbImages = np.asarray(self.dbImages)
            self.qImages = np.asarray(self.qImages)

            # cast to np.arrays for indexing during training
            self.qidxs = np.asarray(self.qidxs)
            self.pidxs = np.asarray(self.pidxs)
            self.nidxs = np.asarray(self.nidxs)
            self.clusters = np.asarray(self.clusters)
            self.qpool = np.asarray(self.qpool)
            self.ppool = np.asarray(self.ppool)
            self.sideways = np.asarray(self.sideways)
            self.night = np.asarray(self.night)

            # decide device type ( important for triplet mining )
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.threads = 8
            self.bs = 24

            if mode == 'train':
                # for now always 1-1 lookup.
                self.negCache = np.asarray([np.empty((0,), dtype=int)]*len(self.qpool))

                # calculate weights for positive sampling
                if positive_sampling:
                    self.__calcSamplingWeights__()
                else:
                    self.weights = np.ones(len(self.qpool)) / float(len(self.qpool))
            

            self.name = name
            self.mode = mode
            self.imsize = imsize

            self.nnum = nnum
            self.qsize = min(qsize, len(self.qpool))
            self.poolsize = min(poolsize, len(self.ppool))
            self.qidxs = None
            self.pidxs = None
            self.nidxs = None

            self.transform = transform
            self.loader = loader
            self.print_freq = 10

                    
        else:
            raise(RuntimeError("Unknown dataset name!"))

        if name.startswith('retrieval-SfM') or name.startswith('gl'):
            # initializing tuples dataset
            self.name = name
            self.mode = mode
            self.imsize = imsize
            self.clusters = db['cluster']
            self.qpool = db['qidxs']
            self.ppool = db['pidxs']

            ## If we want to keep only unique q-p pairs 
            ## However, ordering of pairs will change, although that is not important
            # qpidxs = list(set([(self.qidxs[i], self.pidxs[i]) for i in range(len(self.qidxs))]))
            # self.qidxs = [qpidxs[i][0] for i in range(len(qpidxs))]
            # self.pidxs = [qpidxs[i][1] for i in range(len(qpidxs))]

            # size of training subset for an epoch
            self.nnum = nnum
            self.qsize = min(qsize, len(self.qpool))
            self.poolsize = min(poolsize, len(self.images))
            self.qidxs = None
            self.pidxs = None
            self.nidxs = None

            self.transform = transform
            self.loader = loader

            self.print_freq = 10

    def get_loaders(self):
        # draw qsize random queries for tuples
        idxs2qpool = torch.randperm(len(self.qpool))
        self.qidxs = [self.qpool[i] for i in range(len(self.qpool))]
        self.pidxs = [self.ppool[i] for i in range(len(self.ppool))]
        return self.qidxs, self.pidxs

    def addGpsInfo(self, dataframe, dataframe_raw):
        for index, row in dataframe.iterrows():
            self.gpsInfo[row['key']] = [row['easting'], row['northing']]
        
        for index, row in dataframe_raw.iterrows():
            self.angleInfo[row['key']] = [row['ca']]

    def __calcSamplingWeights__(self):
        # length of query
        N = len(self.qidxs)

        # initialize weights
        self.weights = np.ones(N)
        """
        # weight higher if from night or sideways facing
        if len(self.night) != 0:
            self.weights[self.night] += N / len(self.night)
        if len(self.sideways) != 0:
            self.weights[self.sideways] += N / len(self.sideways)

        # print weight information
        print("#Sideways [{}/{}]; #Night; [{}/{}]".format(len(self.sideways), N, len(self.night), N))
        print("Forward and Day weighted with {:.4f}".format(1))
        if len(self.night) != 0:
            print("Forward and Night weighted with {:.4f}".format(1 + N/len(self.night)))
        if len(self.sideways) != 0:
            print("Sideways and Day weighted with {:.4f}".format( 1 + N/len(self.sideways)))
        if len(self.sideways) != 0 and len(self.night) != 0:
            print("Sideways and Night weighted with {:.4f}".format(1 + N/len(self.night) + N/len(self.sideways)))
        """

    def filter(self, seqKeys, seqIdxs, center_frame_condition):
        keys, idxs = [], []
        for key, idx in zip(seqKeys, seqIdxs):
            if idx[len(idx) // 2] in center_frame_condition:
                keys.append(key)
                idxs.append(idx)
        return keys, np.asarray(idxs)
    
    def arange_as_seq(self, data, path, seq_length):

        seqInfo = pd.read_csv(join(path, 'seq_info.csv'), index_col = 0)

        seq_keys, seq_idxs = [], []
        for idx in data.index:

            # edge cases.
            if idx < (seq_length//2) or idx >= (len(seqInfo) - seq_length//2): continue

            # find surrounding frames in sequence
            seq_idx = np.arange(-seq_length//2, seq_length//2) + 1 + idx
            seq = seqInfo.iloc[seq_idx]

            # the sequence must have the same sequence key and must have consecutive frames
            if len(np.unique(seq['sequence_key'])) == 1 and (seq['frame_number'].diff()[1:] == 1).all():
                seq_key = ','.join([join(path, 'images', key + '.jpg') for key in seq['key']])

                seq_keys.append(seq_key)
                seq_idxs.append(seq_idx)

        return seq_keys, np.asarray(seq_idxs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            images tuple (q,p,n1,...,nN): Loaded train/val tuple at index of self.qidxs
        """
        if self.__len__() == 0:
            raise(RuntimeError("List qidxs is empty. Run ``dataset.create_epoch_tuples(net)`` method to create subset for train/val!"))

        output = []
        # query image
        output.append(self.loader(self.qImages[self.qidxs[index]]))

        # positive image
        pos_index = random.randint(0, len(self.dbImages[self.pidxs[index]])-1)
        output.append(self.loader(self.dbImages[self.pidxs[index]][pos_index])) 
        # negative images
        for i in range(len(self.nidxs[index])):
            output.append(self.loader(self.dbImages[self.nidxs[index][i]]))

        if self.imsize is not None:
            output = [imresize(img, self.imsize) for img in output]
        
        if self.transform is not None:
            output = [self.transform(output[i]).unsqueeze_(0) for i in range(len(output))]

        target = torch.Tensor([-1, 1] + [0]*len(self.nidxs[index]))
        distances = self.getGpsInformation(index, pos_index)
        return (output, target, distances)

    def getGpsInformation(self, index, pos_index):
        distances = []
        qid = self.qImages[self.qidxs[index]].split('/')[-1][:-4]
        pid = self.dbImages[self.pidxs[index]][pos_index].split('/')[-1][:-4]
        distances.append(self.distance(self.gpsInfo.get(qid), self.gpsInfo.get(pid)))
        for i in range(len(self.nidxs[index])):
            nid = self.dbImages[self.nidxs[index][i]].split('/')[-1][:-4]
            distances.append(self.distance(self.gpsInfo.get(qid), self.gpsInfo.get(nid)))
        return distances

    def __len__(self):
        # if not self.qidxs:
        #     return 0
        # return len(self.qidxs)
        return self.qsize

    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        fmt_str += '    Name and mode: {} {}\n'.format(self.name, self.mode)
        fmt_str += '    Number of images: {}\n'.format(len(self.dbImages))
        fmt_str += '    Number of training tuples: {}\n'.format(len(self.qpool))
        fmt_str += '    Number of negatives per tuple: {}\n'.format(self.nnum)
        fmt_str += '    Number of tuples processed in an epoch: {}\n'.format(self.qsize)
        fmt_str += '    Pool size for negative remining: {}\n'.format(self.poolsize)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
    #def distance(self, query, positive):
    #    return np.linalg.norm(np.array(query)-np.array(positive))

    def distance(self, query, positive):
        return torch.norm(torch.tensor(query) - torch.tensor(positive), dim=-1)

    def create_epoch_tuples(self, net):
        
        if self.tuple_mining == 'default':
            return self.epoch_tuples_standard(net)
        elif self.tuple_mining == 'semihard':
            return self.epoch_tuples_semihard(net)
        else:
            return self.epoch_tuples_gps(net)


    def epoch_tuples_standard(self, net):

        print('>> Creating tuples for an epoch of {}-{}...'.format(self.name, self.mode))
        print(">>>> used network: ")
        print(net.meta_repr())

        ## ------------------------
        ## SELECTING POSITIVE PAIRS
        ## ------------------------

        # draw qsize random queries for tuples
        idxs2qpool = torch.randperm(len(self.qpool))[:self.qsize]
        self.qidxs = [self.qpool[i] for i in idxs2qpool]
        self.pidxs = [self.ppool[i] for i in idxs2qpool]
        ## ------------------------
        ## SELECTING NEGATIVE PAIRS
        ## ------------------------

        # if nnum = 0 create dummy nidxs
        # useful when only positives used for training
        if self.nnum == 0:
            self.nidxs = [[] for _ in range(len(self.qidxs))]
            return 0

        # draw poolsize random images for pool of negatives images
        idxs2images = torch.randperm(len(self.ppool))[:self.poolsize]

        # prepare network
        net.cuda()
        net.eval()

        # no gradients computed, to reduce memory and increase speed
        with torch.no_grad():

            print('>> Extracting descriptors for query images...')
            opt = {'batch_size': 1, 'shuffle': False, 'num_workers': 8, 'pin_memory': True}
            loader = torch.utils.data.DataLoader(
                ImagesFromList(root='', images=[self.qImages[i] for i in self.qidxs], imsize=self.imsize, transform=self.transform),
                **opt)

            # extract query vectors
            qvecs = torch.zeros(net.meta['outputdim'], len(self.qidxs)).cuda()
            for i, input in enumerate(loader):
                qvecs[:, i] = net(input.cuda()).data.squeeze()
                if (i+1) % self.print_freq == 0 or (i+1) == len(self.qidxs):
                    print('\r>>>> {}/{} done...'.format(i+1, len(self.qidxs)), end='')

            
            # prepare negative pool data loader
            print('>> Extracting descriptors for negative pool...')
            opt = {'batch_size': 1, 'shuffle': False, 'num_workers': 8, 'pin_memory': True}
            loader = torch.utils.data.DataLoader(
                ImagesFromList(root='', images=[self.dbImages[i] for i in idxs2images], imsize=self.imsize, transform=self.transform),
                **opt
            )

            # extract negative pool vectors
            poolvecs = torch.zeros(net.meta['outputdim'], len(idxs2images)).cuda()
            for i, input in enumerate(loader):
                poolvecs[:, i] = net(input.cuda()).data.squeeze()
                if (i+1) % self.print_freq == 0 or (i+1) == len(idxs2images):
                    print('\r>>>> {}/{} done...'.format(i+1, len(idxs2images)), end='')

            print('>> Searching for hard negatives...')
            # compute dot product scores and ranks on GPU
            scores = torch.mm(poolvecs.t(), qvecs)
            scores, ranks = torch.sort(scores, dim=0, descending=True)
            avg_ndist = torch.tensor(0).float().cuda()  # for statistics
            n_ndist = torch.tensor(0).float().cuda()  # for statistics
            # selection of negative examples
            self.nidxs = []

            for q in range(len(self.qidxs)):
                # do not use query cluster, those images are potentially positive
                nidxs = []
                clusters = []
                r = 0
                if self.mode == 'train':
                    clusters = self.clusters[idxs2qpool[q]]
                 
                while len(nidxs) < self.nnum:
                    potential = int(idxs2images[ranks[r, q]])
                    # take at most one image from the same cluster
                    if (potential not in clusters) and (potential not in self.pidxs[q]):
                        nidxs.append(potential)
                        clusters = np.append(clusters, np.array(potential))
                        avg_ndist += torch.pow(qvecs[:,q]-poolvecs[:,ranks[r, q]]+1e-6, 2).sum(dim=0).sqrt()
                        n_ndist += 1
                    r += 1
                self.nidxs.append(nidxs)
                
            print('>>>> Average negative l2-distance: {:.2f}'.format(avg_ndist/n_ndist))
            print('>>>> Done')
        return (avg_ndist/n_ndist).item()  # return average negative l2-distance

    def epoch_tuples_gps(self, net):
            print('>> Creating tuples for an epoch of {}-{}...'.format(self.name, self.mode))
            # draw qsize random queries for tuples
            idxs2qpool = torch.randperm(len(self.qpool))[:self.qsize]
            # draw poolsize random images for pool of negatives images
            idxs2images = torch.randperm(len(self.ppool))[:self.poolsize]

            ## ------------------------
            ## SELECTING POSITIVE PAIRS
            ## ------------------------

            self.qidxs = [self.qpool[i] for i in idxs2qpool]
            self.pidxs = [self.ppool[i] for i in idxs2qpool]

            ## ------------------------
            ## SELECTING NEGATIVE PAIRS
            ## ------------------------

            # if nnum = 0 create dummy nidxs
            # useful when only positives used for training
            if self.nnum == 0:
                self.nidxs = [[] for _ in range(len(self.qidxs))]
                return 0

            # get query and pool coordinates
            querycoordinates = torch.tensor([self.gpsInfo[self.qImages[i][-26:-4]] for i in self.qidxs], dtype=torch.float)
            poolcoordinates = torch.tensor([self.gpsInfo[self.dbImages[i][-26:-4]] for i in idxs2images], dtype=torch.float)

            """self.positive_distances = []
            for i in range(len(self.qidxs)):
                positives = self.pidxs[i]
                query = querycoordinates[i]
                query_distances = []
                for pos in positives:
                    positive = querycoordinates[pos]
                    query_distances.append(distance(query, positive))
                self.positive_distances.append(query_distances)"""

            # compute distances
            distances = torch.norm(querycoordinates[:, None] - poolcoordinates, dim=2)
            
            # sort distances
            distances, indicies = torch.sort(distances, dim=1, descending=False)

            # selection of negative examples
            self.nidxs = []
            self.distances = []

            # Statistics
            avg_ndist = torch.tensor(0).float().cuda()  # for statistics
            n_ndist = torch.tensor(0).float().cuda()  # for statistics
            for q in range(len(self.qidxs)):
                nidxs = []
                dist_to_query = []
                r = 0
                while len(nidxs) < self.nnum:
                    #TODO: This will choose the same negatives every time (assuming the samples are the same)
                    # Is this dangerous? Do we risk overtraining on a few samples, because we choose them very often?
                    if (len(nidxs) < self.nnum // 2):
                        potential = int(idxs2images[indicies[q, r]]) 
                    else:
                        r = random.randint(len(nidxs), len(idxs2images)-1)
                        potential = int(idxs2images[indicies[q, r]])
                    
                    #TODO: Do we still need the cluster information? 
                    # An advantage could be, that we can 'diversify' the negatives a bit more because we exclude images from its cluster 
                    # clusters = self.clusters[idxs2qpool[q]]
                    #print(float(distances[r, q]), self.negDistThr, float(distances[r, q]) >= self.negDistThr) 
                    if float(distances[q, r]) >= self.negDistThr and (potential not in nidxs):
                        nidxs.append(potential)
                        avg_ndist += distances[q, r]
                        n_ndist += 1
                        dist_to_query.append(distances[q, r])
                    r += 1
                self.distances.append(dist_to_query)
                self.nidxs.append(nidxs)
            return (avg_ndist/n_ndist).item() # return average negative gps distance


    def epoch_tuples_semihard(self, net):

        print('>> Creating tuples for an epoch of {}-{}...'.format(self.name, self.mode))
        print(">>>> used network: ")
        ## SELECTING POSITIVE PAIRS
        ## ------------------------

        # draw qsize random queries for tuples
        idxs2qpool = torch.randperm(len(self.qpool))[:self.qsize]
        self.qidxs = [self.qpool[i] for i in idxs2qpool]
        self.pidxs = [np.random.choice((self.ppool[i]), 1) for i in idxs2qpool]
        ## ------------------------
        ## SELECTING NEGATIVE PAIRS
        ## ------------------------

        # if nnum = 0 create dummy nidxs
        # useful when only positives used for training
        if self.nnum == 0:
            self.nidxs = [[] for _ in range(len(self.qidxs))]
            return 0

        # draw poolsize random images for pool of negatives images
        idxs2images = torch.randperm(len(self.ppool))[:self.poolsize]

        # prepare network
        net.cuda()
        net.eval()

        # no gradients computed, to reduce memory and increase speed
        with torch.no_grad():

            print('>> Extracting descriptors for query images...')
            opt = {'batch_size': 1, 'shuffle': False, 'num_workers': 8, 'pin_memory': True}
            loader = torch.utils.data.DataLoader(
                ImagesFromList(root='', images=[self.qImages[i] for i in self.qidxs], imsize=self.imsize, transform=self.transform),
                **opt)

            # extract query vectors
            qvecs = torch.zeros(net.meta['outputdim'], len(self.qidxs)).cuda()
            for i, input in enumerate(loader):
                qvecs[:, i] = net(input.cuda()).data.squeeze()
                if (i+1) % self.print_freq == 0 or (i+1) == len(self.qidxs):
                    print('\r>>>> {}/{} done...'.format(i+1, len(self.qidxs)), end='')
            
            print('>> Extracting descriptors for positive images...')
            opt = {'batch_size': 1, 'shuffle': False, 'num_workers': 8, 'pin_memory': True}
            loader = torch.utils.data.DataLoader(
                ImagesFromList(root='', images=[self.dbImages[i[0]] for i in self.pidxs], imsize=self.imsize, transform=self.transform),
                **opt)

            # extract query vectors
            pvecs = torch.zeros(net.meta['outputdim'], len(self.pidxs)).cuda()
            for i, input in enumerate(loader):
                pvecs[:, i] = net(input.cuda()).data.squeeze()
                if (i+1) % self.print_freq == 0 or (i+1) == len(self.pidxs):
                    print('\r>>>> {}/{} done...'.format(i+1, len(self.pidxs)), end='')
            
            # prepare negative pool data loader
            print('>> Extracting descriptors for negative pool...')
            opt = {'batch_size': 1, 'shuffle': False, 'num_workers': 8, 'pin_memory': True}
            loader = torch.utils.data.DataLoader(
                ImagesFromList(root='', images=[self.dbImages[i] for i in idxs2images], imsize=self.imsize, transform=self.transform),
                **opt
            )

            # extract negative pool vectors
            poolvecs = torch.zeros(net.meta['outputdim'], len(idxs2images)).cuda()
            for i, input in enumerate(loader):
                poolvecs[:, i] = net(input.cuda()).data.squeeze()
                if (i+1) % self.print_freq == 0 or (i+1) == len(idxs2images):
                    print('\r>>>> {}/{} done...'.format(i+1, len(idxs2images)), end='')

            print('>> Searching for semi hard negatives...')
            # compute dot product scores and ranks on GPU
            scores = torch.mm(poolvecs.t(), qvecs)
            scores, ranks = torch.sort(scores, dim=0, descending=True)
            avg_ndist = torch.tensor(0).float().cuda()  # for statistics
            n_ndist = torch.tensor(0).float().cuda()  # for statistics
            # selection of negative examples
            self.nidxs = []
            for q in range(len(self.qidxs)):
                # do not use query cluster, those images are potentially positive
                nidxs = []
                clusters = []
                r = 0
                if self.mode == 'train':
                    clusters = self.clusters[idxs2qpool[q]]
                pos_dist = torch.pow(qvecs[:,q]-pvecs[:,q]+1e-6, 2).sum(dim=0).sqrt()
                while len(nidxs) < self.nnum:
                    potential = int(idxs2images[ranks[r, q]])
                    neg_dist = torch.pow(qvecs[:,q]-poolvecs[:,ranks[r, q]]+1e-6, 2).sum(dim=0).sqrt()
                    # take at most one image from the same cluster
                    if (potential not in clusters) and ((potential not in self.pidxs[q]) and (neg_dist > pos_dist)):
                        nidxs.append(potential)
                        clusters = np.append(clusters, np.array(potential))
                        avg_ndist += neg_dist
                        n_ndist += 1
                    r += 1
                self.nidxs.append(nidxs)
                
            print('>>>> Average negative l2-distance: {:.2f}'.format(avg_ndist/n_ndist))
            print('>>>> Done')
        return (avg_ndist/n_ndist).item()  # return average negative l2-distance
