## HPC Connect
ssh USERNAME@login2.hpc.dtu.dk

## Port Forwarding 
ssh USERNAME@login2.hpc.dtu.dk -g -L6006:URL:6006 -N

See more here (https://gitlab.com/schiotz/jupyter-DTU-databar)[GBAR]

## Tensorboard
tensorboard --logdir runs/RUN_ID --bind_all

## Alias 
alias train="python3 -m cirtorch.examples.train outputs --training-dataset 'mapillary'"
