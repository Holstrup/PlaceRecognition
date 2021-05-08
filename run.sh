#!/bin/sh
### General options
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J Training
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 23:55
# request 5GB of system-memory
#BSUB -R "rusage[mem=15GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u abho@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o stdout/gpu-%J.out
#BSUB -e stdout/gpu_%J.err
# -- end of LSF options --
### Load modules 
echo "Started"
export PYTHONPATH="${PYTHONPATH}:/zhome/5d/1/117324/Documents/PlaceRecognition" 
module unload cuda
module unload cudann
module load cuda/9.0
module load cudnn/v7.0.5-prod-cuda-9.0
module load python3/3.7.5
echo "Loaded Modules"
### Create a virtual environment for Python3
### python3 -m venv hello_hpc
### Activate virtual environment
source ~/hello_hpc/bin/activate
### pip3 install -r requirements.txt
echo "Starting Training"
###python3 -m cirtorch.examples.train data/outputs --training-dataset 'mapillary' --arch 'resnet50' --loss 'triplet' --loss-margin 0.1 --neg-num 5 --tuple-mining 'semihard'
###python3 -m cirtorch.examples.train data/outputs --training-dataset 'mapillary' --arch 'resnet50' --loss 'ContrastiveWeightedLossVariant' --neg-num 6 --tuple-mining 'gps' --posDistThr 15 --negDistThr 15 --resume 'model_epoch260'
###python3 -m cirtorch.examples.train data/outputs --training-dataset 'mapillary' --arch 'resnet50' --loss 'WeightedGeneralizedContrastiveLoss' --neg-num 6 --tuple-mining 'gps' --posDistThr 10 --negDistThr 10 --resume 'model_epoch260.pth.tar'
###python3 -m cirtorch.examples.train data/outputs --training-dataset 'mapillary' --arch 'resnet50' --loss 'WeightedGeneralizedMSELoss' --neg-num 6 --tuple-mining 'gps' --posDistThr 10 --negDistThr 10 --resume 'model_epoch260.pth.tar'
### python3 cirtorch/networks/localcorrelationnet.py
###python3 cirtorch/networks/correlationnet.py
python3 cirtorch/networks/localcorrelationnet.py
echo "Finished Training"
