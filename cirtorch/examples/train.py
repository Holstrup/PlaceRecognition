import argparse
import os
import shutil
import time
import math
import pickle
import pdb
import io
import PIL

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

import torchvision.transforms as transforms
import torchvision.models as models

from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from cirtorch.layers.loss import ContrastiveLoss, TripletLoss, LinearWeightedContrastiveLoss, LinearOverWeightedContrastiveLoss, RegressionContrastiveLoss, LogTobitLoss, LearntLogTobitLoss, ContrastiveLossVariant, GeneralizedContrastiveLoss, GeneralizedMSELoss
from cirtorch.datasets.datahelpers import collate_tuples, cid2filename
from cirtorch.datasets.traindataset import TuplesDataset
from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.download import download_train, download_test
from cirtorch.utils.whiten import whitenlearn, whitenapply
from cirtorch.utils.evaluate import compute_map_and_print, mapk, recall
from cirtorch.utils.general import get_data_root, htime
from torch.utils.tensorboard import SummaryWriter
from cirtorch.datasets.genericdataset import ImagesFromList

training_dataset_names = ['mapillary']
test_datasets_names = ['mapillary']
test_whiten_names = []#['retrieval-SfM-30k', 'retrieval-SfM-120k']

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
pool_names = ['mac', 'spoc', 'gem', 'gemmp']
loss_names = ['contrastive', 'triplet', 'LinearWeightedContrastive', 'LinearOverWeightedContrastive', 'RegressionWeightedContrastiveLoss', 'LogTobitWeightedLoss', 'LearntLogTobitWeightedLoss', 'ContrastiveWeightedLossVariant', "WeightedGeneralizedContrastiveLoss", "WeightedGeneralizedMSELoss"]
optimizer_names = ['sgd', 'adam']


parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')

# export directory, training and val datasets, test datasets
parser.add_argument('directory', metavar='EXPORT_DIR',
                    help='destination where trained network should be saved')
parser.add_argument('--training-dataset', '-d', metavar='DATASET', default='retrieval-SfM-120k', choices=training_dataset_names,
                    help='training dataset: ' + 
                        ' | '.join(training_dataset_names) +
                        ' (default: retrieval-SfM-120k)')
parser.add_argument('--no-val', dest='val', action='store_false',
                    help='do not run validation')
parser.add_argument('--test-datasets', '-td', metavar='DATASETS', default='mapillary',
                    help='comma separated list of test datasets: ' + 
                        ' | '.join(test_datasets_names) + 
                        ' (default: mapillary)')
parser.add_argument('--test-whiten', metavar='DATASET', default='', choices=test_whiten_names,
                    help='dataset used to learn whitening for testing: ' + 
                        ' | '.join(test_whiten_names) + 
                        ' (default: None)')
parser.add_argument('--test-freq', default=20, type=int, metavar='N', 
                    help='run test evaluation every N epochs (default: 1)')

parser.add_argument('--cities', metavar='CITIES', default='', help='city mode')
parser.add_argument('--tuple-mining', metavar='TUPLES', default='default', help='tuple mining')
parser.add_argument('--posDistThr', metavar='POSITIVEDIST', default=15, help='tuple mining', type=int)
parser.add_argument('--negDistThr', metavar='NEGATIVEDIST', default=25, help='tuple mining', type=int)

# network architecture and initialization options
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet101', choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet101)')
parser.add_argument('--pool', '-p', metavar='POOL', default='gem', choices=pool_names,
                    help='pooling options: ' +
                        ' | '.join(pool_names) +
                        ' (default: gem)')
parser.add_argument('--local-whitening', '-lw', dest='local_whitening', action='store_true',
                    help='train model with learnable local whitening (linear layer) before the pooling')
parser.add_argument('--regional', '-r', dest='regional', action='store_true',
                    help='train model with regional pooling using fixed grid')
parser.add_argument('--whitening', '-w', dest='whitening', action='store_true',
                    help='train model with learnable whitening (linear layer) after the pooling')
parser.add_argument('--not-pretrained', dest='pretrained', action='store_false',
                    help='initialize model with random weights (default: pretrained on imagenet)')
parser.add_argument('--loss', '-l', metavar='LOSS', default='LinearWeightedContrastive',
                    choices=loss_names,
                    help='training loss options: ' +
                        ' | '.join(loss_names) +
                        ' (default: LinearWeightedContrastive)')
parser.add_argument('--loss-margin', '-lm', metavar='LM', default=0.7, type=float,
                    help='loss margin: (default: 0.7)')

# train/val options specific for image retrieval learning
parser.add_argument('--image-size', default=320, type=int, metavar='N',
                    help='maximum size of longer image side used for training (default: 320)')
parser.add_argument('--neg-num', '-nn', default=5, type=int, metavar='N',
                    help='number of negative image per train/val tuple (default: 5)')
parser.add_argument('--query-size', '-qs', default=2000, type=int, metavar='N',
                    help='number of queries randomly drawn per one train epoch (default: 2000)')
parser.add_argument('--pool-size', '-ps', default=20000, type=int, metavar='N',
                    help='size of the pool for hard negative mining (default: 20000)')

# standard train/val options
parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                    help='gpu id used for training (default: 0)')
parser.add_argument('--workers', '-j', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run (default: 100)')
parser.add_argument('--batch-size', '-b', default=5, type=int, metavar='N', 
                    help='number of (q,p,n1,...,nN) tuples in a mini-batch (default: 5)')
parser.add_argument('--update-every', '-u', default=5, type=int, metavar='N',
                    help='update model weights every N batches, used to handle really large batches, ' + 
                        'batch_size effectively becomes update_every x batch_size (default: 1)')
parser.add_argument('--optimizer', '-o', metavar='OPTIMIZER', default='adam',
                    choices=optimizer_names,
                    help='optimizer options: ' +
                        ' | '.join(optimizer_names) +
                        ' (default: adam)')
parser.add_argument('--lr', '--learning-rate', default=1e-6, type=float,
                    metavar='LR', help='initial learning rate (default: 1e-6)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-6)')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='FILENAME',
                    help='name of the latest checkpoint (default: None)')

min_loss = float('inf')

def main():
    global args, min_loss, writer, global_epoch
    global_epoch = 0
    args = parser.parse_args()

    # manually check if there are unknown test datasets
    for dataset in args.test_datasets.split(','):
        if dataset not in test_datasets_names:
            raise ValueError('Unsupported or unknown test dataset: {}!'.format(dataset))

    # check if test dataset are downloaded
    # and download if they are not
    #download_train(get_data_root())
    #download_test(get_data_root())

    # create export dir if it doesnt exist
    directory = "{}".format(args.training_dataset)
    directory += "_{}".format(args.arch)
    directory += "_{}".format(args.pool)
    if args.local_whitening:
        directory += "_lwhiten"
    if args.regional:
        directory += "_r"
    if args.whitening:
        directory += "_whiten"
    if not args.pretrained:
        directory += "_notpretrained"
    directory += "_{}_m{:.2f}".format(args.loss, args.loss_margin)
    directory += "_{}_lr{:.1e}_wd{:.1e}".format(args.optimizer, args.lr, args.weight_decay)
    directory += "_nnum{}_qsize{}_psize{}".format(args.neg_num, args.query_size, args.pool_size)
    directory += "_bsize{}_uevery{}_imsize{}".format(args.batch_size, args.update_every, args.image_size)
    t = time.strftime("%Y-%d-%m_%H:%M:%S", time.localtime())
    writer = SummaryWriter('data/runs/{}_{}_{}'.format(args.arch, args.loss, t))

    args.directory = os.path.join(args.directory, directory)
    print(">> Creating directory if it does not exist:\n>> '{}'".format(args.directory))
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    # set cuda visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    # set random seeds
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    # initialize model
    if args.pretrained:
        print(">> Using pre-trained model '{}'".format(args.arch))
    else:
        print(">> Using model from scratch (random weights) '{}'".format(args.arch))
    model_params = {}
    model_params['architecture'] = args.arch
    model_params['pooling'] = args.pool
    model_params['local_whitening'] = args.local_whitening
    model_params['regional'] = args.regional
    model_params['whitening'] = args.whitening
    # model_params['mean'] = ...  # will use default
    # model_params['std'] = ...  # will use default
    model_params['pretrained'] = args.pretrained
    model = init_network(model_params)

    # move network to gpu
    model.cuda()

    if 'Weighted' in args.loss:
        posDistThr=args.posDistThr #15
        negDistThr=args.negDistThr #25
    else:
        posDistThr=args.posDistThr #15
        negDistThr=args.negDistThr #25
        
    # define loss function (criterion) and optimizer
    if args.loss == 'contrastive':
        criterion = ContrastiveLoss(margin=args.loss_margin).cuda()
    elif args.loss == 'LinearWeightedContrastive':
        criterion = LinearWeightedContrastiveLoss(margin=args.loss_margin, gpsmargin=posDistThr).cuda()
    elif args.loss == 'LinearOverWeightedContrastive':
        criterion = LinearOverWeightedContrastiveLoss(margin=args.loss_margin, gpsmargin=posDistThr).cuda()
    elif args.loss == 'RegressionWeightedContrastiveLoss':
        criterion = RegressionContrastiveLoss(margin=args.loss_margin, gpsmargin=posDistThr).cuda()
    elif args.loss == 'triplet':
        criterion = TripletLoss(margin=args.loss_margin).cuda()
    elif args.loss == 'LogTobitWeightedLoss':
        criterion = LogTobitLoss(margin=args.loss_margin, gpsmargin=posDistThr).cuda()
    elif args.loss == 'LearntLogTobitWeightedLoss':
        criterion = LearntLogTobitLoss(margin=args.loss_margin, gpsmargin=posDistThr, scaling=15).cuda()
    elif args.loss == 'ContrastiveWeightedLossVariant':
        criterion = ContrastiveLossVariant().cuda()
    elif args.loss == 'WeightedGeneralizedContrastiveLoss':
        criterion = GeneralizedContrastiveLoss().cuda()
    elif args.loss == 'WeightedGeneralizedMSELoss':
        criterion = GeneralizedMSELoss().cuda()
    else:
        raise(RuntimeError("Loss {} not available!".format(args.loss)))

    # parameters split into features, pool, whitening 
    # IMPORTANT: no weight decay for pooling parameter p in GeM or regional-GeM
    parameters = []
    # add feature parameters
    parameters.append({'params': model.features.parameters()})
    # add local whitening if exists
    if model.lwhiten is not None:
        parameters.append({'params': model.lwhiten.parameters()})
    # add pooling parameters (or regional whitening which is part of the pooling layer!)
    if not args.regional:
        # global, only pooling parameter p weight decay should be 0
        if args.pool == 'gem':
            parameters.append({'params': model.pool.parameters(), 'lr': args.lr*10, 'weight_decay': 0})
        elif args.pool == 'gemmp':
            parameters.append({'params': model.pool.parameters(), 'lr': args.lr*100, 'weight_decay': 0})
    else:
        # regional, pooling parameter p weight decay should be 0, 
        # and we want to add regional whitening if it is there
        if args.pool == 'gem':
            parameters.append({'params': model.pool.rpool.parameters(), 'lr': args.lr*10, 'weight_decay': 0})
        elif args.pool == 'gemmp':
            parameters.append({'params': model.pool.rpool.parameters(), 'lr': args.lr*100, 'weight_decay': 0})
        if model.pool.whiten is not None:
            parameters.append({'params': model.pool.whiten.parameters()})
    # add final whitening if exists
    if model.whiten is not None:
        parameters.append({'params': model.whiten.parameters()})
    
    #TODO: Add Scaling Parameter & Save
    #if 'Learnt' in args.loss:
    #    parameters.append({'params': criterion.parameters()})

    # define optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(parameters, args.lr, weight_decay=args.weight_decay)

    # define learning rate decay schedule
    exp_decay = math.exp(-0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay)

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        args.resume = os.path.join(args.directory, args.resume)
        if os.path.isfile(args.resume):
            # load checkpoint weights and update model and optimizer
            print(">> Loading checkpoint:\n>> '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            min_loss = checkpoint['min_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(">>>> loaded checkpoint:\n>>>> '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            # important not to forget scheduler updating
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay, last_epoch=checkpoint['epoch']-1)
        else:
            print(">> No checkpoint found at '{}'".format(args.resume))
    
    # Data loading code
    print('MEAN: ' + str(model.meta['mean']))
    print('STD: ' + str(model.meta['std']))
    normalize = transforms.Normalize(mean=model.meta['mean'], std=model.meta['std'])
    imsize = args.image_size
    resize = transforms.Resize((int(imsize * 3/4), imsize), interpolation=2)

    transform = transforms.Compose([
        resize,
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = TuplesDataset(
        name=args.training_dataset,
        mode='train',
        imsize=imsize,
        nnum=args.neg_num,
        qsize=args.query_size,
        poolsize=args.pool_size,
        transform=transform,
        posDistThr=posDistThr,
        negDistThr=negDistThr, 
        root_dir = 'data',
        cities=args.cities,
        tuple_mining=args.tuple_mining
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None,
        drop_last=True, collate_fn=collate_tuples
    )
    if args.val:
        val_dataset = TuplesDataset(
            name=args.training_dataset,
            mode='val',
            imsize=imsize,
            nnum=args.neg_num,
            qsize=float('Inf'),
            poolsize=float('Inf'),
            transform=transform,
            posDistThr=negDistThr, # Use 25 meters for both pos and neg
            negDistThr=negDistThr,
            root_dir = 'data',
            cities=args.cities,
            tuple_mining=args.tuple_mining
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
            drop_last=True, collate_fn=collate_tuples
        )

    # evaluate the network before starting
    #test(args.test_datasets, model)

    # initialize timers 
    train_epoch = 0
    train_time = 0
    val_time = 0
    test_time = 0
    for epoch in range(start_epoch, args.epochs):
        global_epoch = epoch
        print('> Starting Epoch {}/{}'.format(start_epoch, args.epochs))
        epoch_start = time.time()
        # set manual seeds per epoch
        np.random.seed(epoch)
        torch.manual_seed(epoch)
        torch.cuda.manual_seed_all(epoch)

        # adjust learning rate for each epoch
        scheduler.step()
        # # debug printing to check if everything ok
        lr_feat = optimizer.param_groups[0]['lr']
        lr_pool = optimizer.param_groups[1]['lr']
        writer.add_scalar('LearningRate/Feature', lr_feat, epoch)
        writer.add_scalar('LearningRate/Pooling', lr_pool, epoch)
        # print('>> Features lr: {:.2e}; Pooling lr: {:.2e}'.format(lr_feat, lr_pool))
        
        # train for one epoch on train set
        train_start = time.time()
        print('>> Training Epoch {}/{}'.format(start_epoch, args.epochs))
        loss = train(train_loader, model, criterion, optimizer, epoch)
        train_time += time.time() - train_start
        writer.add_scalar('Loss/train', loss, epoch)
        writer.add_scalar('Timing/CumulativeTraining', train_time, epoch)

        # evaluate on validation set
        if args.val and (epoch + 1) % args.test_freq == 0: # Only validate every N epochs s
            with torch.no_grad():
                print('>> Validation Epoch {}/{}'.format(start_epoch, args.epochs))
                val_start = time.time()
                loss = validate(val_loader, model, criterion, epoch)
                val_time += time.time() - val_start
                writer.add_scalar('Timing/CumulativeValidation', val_time, epoch)
                writer.add_scalar('Loss/validation', loss, epoch)
        
        # evaluate on test datasets every test_freq epochs
        if (epoch + 1) % args.test_freq == 0:
            with torch.no_grad():
                print('>> Test Epoch {}/{}'.format(start_epoch, args.epochs))
                test_start = time.time()
                mAP, rec = test(args.test_datasets, model)
                
                # remember best loss and save checkpoint
                is_best = loss < min_loss
                min_loss = min(loss, min_loss) # And max
                save_checkpoint({
                'epoch': epoch + 1,
                'meta': model.meta,
                'state_dict': model.state_dict(),
                'min_loss': min_loss,
                'optimizer' : optimizer.state_dict(),
                }, is_best, args.directory)
                
                test_time += time.time() - test_start
                writer.add_scalar('Timing/CumulativeTest', test_time, epoch)
                writer.add_scalar('Test/mAP', mAP, epoch)
                writer.add_scalar('Test/Recall', rec, epoch)

        train_epoch += time.time() - epoch_start
        writer.add_scalar('Timing/CumulativeEpoch', train_epoch, epoch)

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # create tuples for training
    start_time = time.time()
    avg_neg_distance = train_loader.dataset.create_epoch_tuples(model)
    writer.add_scalar('Timing/TrainCreateTuples', time.time() - start_time, epoch)
    writer.add_scalar('Embeddings/AvgNegDistanceTrain', avg_neg_distance, epoch)

    # switch to train mode
    model.train()
    model.apply(set_batchnorm_eval)

    # zero out gradients
    optimizer.zero_grad()

    end = time.time()
    for i, (input, target, gps_info) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        nq = len(input) # number of training tuples
        ni = len(input[0]) # number of images per tuple
        gps_info = torch.tensor(gps_info)
        for q in range(nq):
            if i % 400 == 0:
                batchid = 400 * epoch + i 

                # Calculate positive distance
                #dist = distance(gps_info[q][0], gps_info[q][1])
                writer.add_scalar('GPSDistance/Postive', gps_info[q][0], batchid)

                # Calculate hardest negative distance
                #dist = distance(gps_info[q][0], gps_info[q][2])
                #writer.add_scalar('GPSDistance/HardestNegative', dist, batchid)

                if gps_info[q][0] < 5: # meters
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
                    images = input[q][0] * std + mean

                    for image_tensor in input[q][1:]:
                        new_image = image_tensor * std + mean
                        images = torch.cat([images, new_image], dim=0)
                    
                    writer.add_images('ImageBatch: {}'.format(batchid), images, 0)

            output = torch.zeros(model.meta['outputdim'], ni).cuda()
            for imi in range(ni):
                # compute output vector for image imi
                output[:, imi] = model(input[q][imi].cuda()).squeeze()

            # reducing memory consumption:
            # compute loss for this query tuple only
            # then, do backward pass for one tuple only
            # each backward pass gradients will be accumulated
            # the optimization step is performed for the full batch later
            if 'Learnt' in args.loss:
                loss = criterion(output, target[q].cuda(), gps_info[q], epoch=epoch)
                if i % 200 == 0:
                    batchid = 400 * epoch + i 
                    writer.add_scalar('Embeddings/Weighting', criterion.weighting, batchid)
                    writer.add_scalar('Embeddings/LearntScaling', criterion.scaling, epoch)
            elif 'Weighted' in args.loss:
                loss = criterion(output, target[q].cuda(), gps_info[q].cuda())
                if i % 200 == 0:
                    batchid = 400 * epoch + i 
                    writer.add_scalar('Embeddings/Weighting', criterion.mse_loss, batchid)
            else:
                loss = criterion(output, target[q].cuda())
            losses.update(loss.item())
            loss.backward()

        if (i + 1) % args.update_every == 0:
            # do one step for multiple batches
            # accumulated gradients are used
            optimizer.step()
            # zero out gradients so we can 
            # accumulate new ones over batches
            optimizer.zero_grad()
            # print('>> Train: [{0}][{1}/{2}]\t'
            #       'Weight update performed'.format(
            #        epoch+1, i+1, len(train_loader)))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0 or i == 0 or (i+1) == len(train_loader):
            writer.add_scalar('Loss/train_batch', losses.avg, epoch)
            print('>> Train: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch+1, i+1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

    return losses.avg


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # create tuples for validation
    start_time = time.time()
    avg_neg_distance = val_loader.dataset.create_epoch_tuples(model)
    writer.add_scalar('Timing/TrainCreateTuples', time.time() - start_time, epoch)
    writer.add_scalar('Embeddings/AvgNegDistanceValidation', avg_neg_distance, epoch)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, gps_info) in enumerate(val_loader):

        nq = len(input) # number of training tuples
        ni = len(input[0]) # number of images per tuple
        output = torch.zeros(model.meta['outputdim'], nq*ni).cuda()
        for q in range(nq):
            for imi in range(ni):
                output[:, q*ni + imi] = model(input[q][imi].cuda()).squeeze()
            
        # no need to reduce memory consumption (no backward pass):
        # compute loss for the full batch
        #TODO: Fix loss func to be able to take a batch - For now using contrastive loss
        gps_info = torch.tensor(gps_info)
        gps_out = torch.flatten(gps_info)
        loss = criterion(output, torch.cat(target).cuda(), gps_out.cuda())

        # record loss
        losses.update(loss.item()/nq, nq)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0 or i == 0 or (i+1) == len(val_loader):
            writer.add_scalar('Loss/val_batch', losses.avg, epoch)
            print('>> Val: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch+1, i+1, len(val_loader), batch_time=batch_time, loss=losses))

    return losses.avg

def test(datasets, net):

    print('>> Evaluating network on test datasets...')

    # for testing we use image size of max 1024
    image_size = 1024

    posDistThr = 25
    negDistThr = 25

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
    imsize = args.image_size
    test_dataset = TuplesDataset(
        name=args.training_dataset,
        mode='val', # Test on validation set during training
        imsize=imsize,
        nnum=args.neg_num,
        qsize=args.query_size,
        poolsize=args.pool_size,
        transform=transform,
        posDistThr=posDistThr,
        negDistThr=negDistThr,
        root_dir = 'data',
        cities=''
    )
    qidxs, pidxs = test_dataset.get_loaders()

    opt = {'batch_size': 1, 'shuffle': False, 'num_workers': 8, 'pin_memory': True}

    # evaluate on test datasets
    datasets = args.test_datasets.split(',')
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

        for q in range(1, 20, 5):
            gen_plot(q, scores, ranks, pidxs)


        print('>> Achieved mAP: {} and Recall {}'.format(mean_ap, rec))
        print('>> {}: elapsed time: {}'.format(dataset, htime(time.time()-start)))
        return mean_ap, rec

def save_checkpoint(state, is_best, directory):
    filename = os.path.join(directory, 'model_epoch%d.pth.tar' % state['epoch'])
    torch.save(state, filename)
    if is_best:
        filename_best = os.path.join(directory, 'model_best.pth.tar')
        shutil.copyfile(filename, filename_best)

def distance(query, positive):
    return torch.norm(torch.tensor(query)-torch.tensor(positive))

def gen_plot(q, scores, ranks, pidxs):
    global writer
    plt.figure()
    q_scores = scores[q,:20]
    q_ranks = ranks[q,:20]
    q_pidx = pidxs[q]
    positives = []
    negatives = []
    positive_ranks = []
    negative_ranks = []

    for i in range(len(q_ranks)):
        if q_ranks[i] in q_pidx:
            positives.append(1-q_scores[i])
            positive_ranks.append(i)
        else:
            negatives.append(1-q_scores[i])
            negative_ranks.append(i)
    print(negative_ranks, negatives)
    plt.scatter(positive_ranks, positives, color='g')
    plt.scatter(negative_ranks, negatives, color='r')
    plt.ylim(0.0, 1.0)
    plt.title("Closest Point to Query")

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    
    image = PIL.Image.open(buf)
    image = transforms.ToTensor()(image).unsqueeze(0)
    writer.add_image(f'Image_{q}', image[0], global_epoch)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_batchnorm_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        # freeze running mean and std:
        # we do training one image at a time
        # so the statistics would not be per batch
        # hence we choose freezing (ie using imagenet statistics)
        m.eval()
        # # freeze parameters:
        # # in fact no need to freeze scale and bias
        # # they can be learned
        # # that is why next two lines are commented
        # for p in m.parameters():
            # p.requires_grad = False


if __name__ == '__main__':
    main()
