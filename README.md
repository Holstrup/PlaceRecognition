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
rsync -r data/train_val/cph/ data/mapillary/train_val/cph/