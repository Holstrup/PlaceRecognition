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

from cirtorch.datasets.datahelpers import default_loader, imresize, cid2filename
from cirtorch.datasets.genericdataset import ImagesFromList
from cirtorch.utils.general import get_data_root

default_cities = {
                'train': ["zurich"],
                'val': ['zurich'],
                'test': ["buenosaires"]
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
        images (list): List of full filenames for each image
        clusters (list): List of clusterID per image
        qpool (list): List of all query image indexes
        ppool (list): List of positive image indexes, each corresponding to query at the same position in qpool

        qidxs (list): List of qsize query image indexes to be processed in an epoch
        pidxs (list): List of qsize positive image indexes, each corresponding to query at the same position in qidxs
        nidxs (list): List of qsize tuples of negative images
                        Each nidxs tuple contains nnum images corresponding to query image at the same position in qidxs

        Lists qidxs, pidxs, nidxs are refreshed by calling the ``create_epoch_tuples()`` method, 
            ie new q-p pairs are picked and negative images are remined
    """

    def __init__(self, name, mode, imsize=None, nnum=5, qsize=2000, poolsize=20000, transform=None, loader=default_loader):

        if not (mode == 'train' or mode == 'val'):
            raise(RuntimeError("MODE should be either train or val, passed as string"))

        if name.startswith('retrieval-SfM'):
            # setting up paths
            data_root = get_data_root()
            db_root = os.path.join(data_root, 'train', name)
            ims_root = os.path.join(db_root, 'ims')
    
            # loading db
            db_fn = os.path.join(db_root, '{}.pkl'.format(name))
            with open(db_fn, 'rb') as f:
                db = pickle.load(f)[mode]
    
            # setting fullpath for images
            self.images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]

        elif name.startswith('gl'):
            ## TODO: NOT IMPLEMENTED YET PROPOERLY (WITH AUTOMATIC DOWNLOAD)

            # setting up paths
            db_root = '/mnt/fry2/users/datasets/landmarkscvprw18/recognition/'
            ims_root = os.path.join(db_root, 'images', 'train')
    
            # loading db
            db_fn = os.path.join(db_root, '{}.pkl'.format(name))
            with open(db_fn, 'rb') as f:
                db = pickle.load(f)[mode]
    
            # setting fullpath for images
            self.images = [os.path.join(ims_root, db['cids'][i]+'.jpg') for i in range(len(db['cids']))]
        elif name.startswith('mapillary'):
            # Parameters 
            root_dir = 'data/mapillary' #get_data_root()
            cities = ''
            nNeg = 5
            transform = None
            mode = 'train'
            task = 'im2im'
            subtask = 'all'
            seq_length = 1
            posDistThr = 10
            negDistThr = 25
            cached_queries = 1000
            cached_negatives = 1000
            positive_sampling = True

            # initializing
            assert mode in ('train', 'val', 'test')
            assert task in ('im2im', 'im2seq', 'seq2im', 'seq2seq')
            assert subtask in ('all', 's2w', 'w2s', 'o2n', 'n2o', 'd2n', 'n2d')
            assert seq_length % 2 == 1
            assert (task == 'im2im' and seq_length == 1) or (task != 'im2im' and seq_length > 1)

            if cities in default_cities:
                self.cities = default_cities[cities]
            elif cities == '':
                self.cities = default_cities[mode]
            else:
                self.cities = cities.split(',')

            self.qidxs = [] # qIdx
            self.qpool = [] # qImages
            self.pidxs = [] # pIdx
            self.nidxs = [] # nonNegIdx
            self.ppool = [] # dbImages
            self.sideways = []
            self.night = []

            # hyper-parameters
            self.nNeg = nNeg
            self.margin = 0.1
            self.posDistThr = posDistThr
            self.negDistThr = negDistThr
            self.cached_queries = cached_queries
            self.cached_negatives = cached_negatives

            # flags
            self.cache = None
            self.exclude_panos = True
            self.mode = mode
            self.subtask = subtask

            # other
            self.transform = transform
            self.query_keys_with_no_match = []

            # define sequence length based on task
            if task == 'im2im':
                seq_length_q, seq_length_db = 1, 1

            # load data
            for city in self.cities:
                print("=====> {}".format(city))

                subdir = 'test' if city in default_cities['test'] else 'train_val'

                # get len of images from cities so far for indexing
                _lenQ = len(self.qpool)
                _lenDb = len(self.ppool)

                # when GPS / UTM is available
                if self.mode in ['train','val']:
                    # load query data
                    qData = pd.read_csv(join(root_dir, subdir, city, 'query', 'postprocessed.csv'), index_col = 0)
                    qDataRaw = pd.read_csv(join(root_dir, subdir, city, 'query', 'raw.csv'), index_col = 0)

                    # load database data
                    dbData = pd.read_csv(join(root_dir, subdir, city, 'database', 'postprocessed.csv'), index_col = 0)
                    dbDataRaw = pd.read_csv(join(root_dir, subdir, city, 'database', 'raw.csv'), index_col = 0)

                    # arange based on task
                    qSeqKeys, qSeqIdxs = self.arange_as_seq(qData, join(root_dir, subdir, city, 'query'), seq_length_q)
                    dbSeqKeys, dbSeqIdxs = self.arange_as_seq(dbData, join(root_dir, subdir, city, 'database'), seq_length_db)

                    # filter based on subtasks
                    if self.mode in ['val']:
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

                    self.qpool.extend(qSeqKeys)
                    self.ppool.extend(dbSeqKeys)

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

                            self.pidxs.append(p_seq_idx + _lenDb)
                            self.qidxs.append(q_seq_idx + _lenQ)

                            # in training we have two thresholds, one for finding positives and one for finding images that we are certain are negatives.
                            if self.mode == 'train':

                                n_uniq_frame_idxs = np.unique([n for nonNeg in nI[q_uniq_frame_idx] for n in nonNeg])
                                n_seq_idx = np.unique(uniqFrameIdx2seqIdx(unique_dbSeqIdx[n_uniq_frame_idxs], dbSeqIdxs))

                                self.nidxs.append(n_seq_idx + _lenDb)

                                # gather meta which is useful for positive sampling
                                if sum(night[np.in1d(index, q_frame_idxs)]) > 0: self.night.append(len(self.qidxs)-1)
                                if sum(sideways[np.in1d(index, q_frame_idxs)]) > 0: self.sideways.append(len(self.qidxs)-1)

                        else:
                            query_key = qSeqKeys[q_seq_idx].split('/')[-1][:-4]
                            self.query_keys_with_no_match.append(query_key)
        
                # when GPS / UTM / pano info is not available    
                elif self.mode in ['test']:

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

                    self.qpool.extend(qSeqKeys)
                    self.ppool.extend(dbSeqKeys)

                    # add query index
                    self.qidxs.extend(list(range(_lenQ, len(qSeqKeys) + _lenQ)))

            # if a combination of cities, task and subtask is chosen, where there are no query/database images, then exit
            if len(self.qpool) == 0 or len(self.ppool) == 0:
                print("Exiting...")
                print("A combination of cities, task and subtask have been chosen, where there are no query/database images.")
                print("Try choosing a different subtask or more cities")
                sys.exit()

            # creates self.images 
            self.images = self.qidxs + self.pidxs + self.nidxs

            # cast to np.arrays for indexing during training
            self.qidxs = np.asarray(self.qidxs)
            self.qpool = np.asarray(self.qpool)
            self.pidxs = np.asarray(self.pidxs)
            self.nidxs = np.asarray(self.nidxs)
            self.ppool = np.asarray(self.ppool)
            self.sideways = np.asarray(self.sideways)
            self.night = np.asarray(self.night)

            # decide device type ( important for triplet mining )
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.threads = 8
            self.bs = 24

            if mode == 'train':
                # for now always 1-1 lookup.
                self.negCache = np.asarray([np.empty((0,), dtype=int)]*len(self.qidxs))

                # calculate weights for positive sampling
                if positive_sampling:
                    self.__calcSamplingWeights__()
                else:
                    self.weights = np.ones(len(self.qidxs)) / float(len(self.qidxs))
            

            self.name = name
            self.mode = mode
            self.imsize = imsize

            self.nnum = nnum
            self.qsize = min(qsize, len(self.qpool))
            self.poolsize = min(poolsize, len(self.images))
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

    def __calcSamplingWeights__(self):
        # length of query
        N = len(self.qIdx)

        # initialize weights
        self.weights = np.ones(N)

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
        output.append(self.loader(self.images[self.qidxs[index]]))
        # positive image
        output.append(self.loader(self.images[self.pidxs[index]]))
        # negative images
        for i in range(len(self.nidxs[index])):
            output.append(self.loader(self.images[self.nidxs[index][i]]))

        if self.imsize is not None:
            output = [imresize(img, self.imsize) for img in output]
        
        if self.transform is not None:
            output = [self.transform(output[i]).unsqueeze_(0) for i in range(len(output))]

        target = torch.Tensor([-1, 1] + [0]*len(self.nidxs[index]))

        return output, target

    def __len__(self):
        # if not self.qidxs:
        #     return 0
        # return len(self.qidxs)
        return self.qsize

    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        fmt_str += '    Name and mode: {} {}\n'.format(self.name, self.mode)
        fmt_str += '    Number of images: {}\n'.format(len(self.images))
        fmt_str += '    Number of training tuples: {}\n'.format(len(self.qpool))
        fmt_str += '    Number of negatives per tuple: {}\n'.format(self.nnum)
        fmt_str += '    Number of tuples processed in an epoch: {}\n'.format(self.qsize)
        fmt_str += '    Pool size for negative remining: {}\n'.format(self.poolsize)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def create_epoch_tuples(self, net):

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
        idxs2images = torch.randperm(len(self.images))[:self.poolsize]

        # prepare network
        net.cuda()
        net.eval()

        # no gradients computed, to reduce memory and increase speed
        with torch.no_grad():

            print('>> Extracting descriptors for query images...')
            # prepare query loader
            loader = torch.utils.data.DataLoader(
                ImagesFromList(root='', images=[self.images[i] for i in self.qidxs], imsize=self.imsize, transform=self.transform),
                batch_size=1, shuffle=False, num_workers=8, pin_memory=True
            )
            # extract query vectors
            qvecs = torch.zeros(net.meta['outputdim'], len(self.qidxs)).cuda()
            for i, input in enumerate(loader):
                qvecs[:, i] = net(input.cuda()).data.squeeze()
                if (i+1) % self.print_freq == 0 or (i+1) == len(self.qidxs):
                    print('\r>>>> {}/{} done...'.format(i+1, len(self.qidxs)), end='')
            print('')

            print('>> Extracting descriptors for negative pool...')
            # prepare negative pool data loader
            loader = torch.utils.data.DataLoader(
                ImagesFromList(root='', images=[self.images[i] for i in idxs2images], imsize=self.imsize, transform=self.transform),
                batch_size=1, shuffle=False, num_workers=8, pin_memory=True
            )
            # extract negative pool vectors
            poolvecs = torch.zeros(net.meta['outputdim'], len(idxs2images)).cuda()
            for i, input in enumerate(loader):
                poolvecs[:, i] = net(input.cuda()).data.squeeze()
                if (i+1) % self.print_freq == 0 or (i+1) == len(idxs2images):
                    print('\r>>>> {}/{} done...'.format(i+1, len(idxs2images)), end='')
            print('')

            print('>> Searching for hard negatives...')
            # compute dot product scores and ranks on GPU
            scores = torch.mm(poolvecs.t(), qvecs)
            scores, ranks = torch.sort(scores, dim=0, descending=True)
            avg_ndist = torch.tensor(0).float().cuda()  # for statistics
            n_ndist = torch.tensor(0).float().cuda()  # for statistics
            # selection of negative examples
            self.nidxs = []
            for q in range(len(self.qidxs)):
                # do not use query cluster,
                # those images are potentially positive
                qcluster = self.clusters[self.qidxs[q]]
                clusters = [qcluster]
                nidxs = []
                r = 0
                while len(nidxs) < self.nnum:
                    potential = idxs2images[ranks[r, q]]
                    # take at most one image from the same cluster
                    if not self.clusters[potential] in clusters:
                        nidxs.append(potential)
                        clusters.append(self.clusters[potential])
                        avg_ndist += torch.pow(qvecs[:,q]-poolvecs[:,ranks[r, q]]+1e-6, 2).sum(dim=0).sqrt()
                        n_ndist += 1
                    r += 1
                self.nidxs.append(nidxs)
            print('>>>> Average negative l2-distance: {:.2f}'.format(avg_ndist/n_ndist))
            print('>>>> Done')

        return (avg_ndist/n_ndist).item()  # return average negative l2-distance
