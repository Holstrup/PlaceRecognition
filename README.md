TODO

## HPC Connect
ssh abho@login2.hpc.dtu.dk

## Port Forwarding 
ssh abho@login2.hpc.dtu.dk -g -L6006:n-62-20-9.hpccluster.dtu.dk:6006 -N

See more here (https://gitlab.com/schiotz/jupyter-DTU-databar)[GBAR]

## Tensorboard
tensorboard --logdir runs/RUN_ID --bind_all

## Datasets left
#austin  bangkok  berlin  boston  manila  sf  tokyo  toronto
rsync options source destination
rsync -r data/train_val/CITY/ data/mapillary/train_val/CITY/
rsync -r  data/mapillary/metadata/train_val/CITY/MODE/ data/train/mapillary/train_val/CITY/MODE

## Alias 
alias train="python3 -m cirtorch.examples.train outputs --training-dataset 'mapillary'"