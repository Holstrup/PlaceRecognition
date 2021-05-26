import sys
sys.path.insert(0, "/Users/alexanderholstrup/git/VisualPlaceRecognition/cnnimageretrieval-pytorch")

from cirtorch.datasets.traindataset import TuplesDataset
from cirtorch.utils.view_angle import field_of_view, ious, plot_fov

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from os.path import join

MODE = 'train'
root = 'data'

default_cities = {
    'train': ["zurich", "london", "boston", "melbourne", "amsterdam","helsinki",
              "tokyo","toronto","saopaulo","moscow","trondheim","paris","bangkok",
              "budapest","austin","berlin","ottawa","phoenix","goa","amman","nairobi","manila"],
    'val': ["cph", "sf"],
    'test': ["miami","athens","buenosaires","stockholm","bengaluru","kampala"]
}

for city in default_cities[MODE]:
    data = {}
    train_dataset = TuplesDataset(
        name='mapillary',
        mode='train',
        qsize=float('Inf'),
        poolsize=float('Inf'),
        posDistThr=50,
        negDistThr=50, 
        root_dir = root,
        cities=city,
    )

    for i, qIdx in enumerate(train_dataset.qpool):
        # qID from qIdx
        qId = train_dataset.qImages[qIdx][-26:-4]
            
        # positives from index i
        positives = train_dataset.ppool[i]
        dbIds = [train_dataset.dbImages[j][-26:-4] for j in positives]

        # get coordinates
        points = [train_dataset.gpsInfo[qId] + train_dataset.angleInfo[qId]]
        points.extend([train_dataset.gpsInfo[dbId] + train_dataset.angleInfo[dbId] for dbId in dbIds])

        pol = field_of_view(points)
        data[qId] = [dbIds, ious(pol[0], pol[1:])]
    json.dump(data, open(join(root, 'train_val', city, 'query', 'ious.json'), 'w+'))