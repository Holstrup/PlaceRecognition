#!/bin/sh

### General options

### –- specify queue --
#BSUB -q gpuv100

### -- set the job Name --
#BSUB -J testjob

### -- ask for number of cores (default: 1) --
#BSUB -n 1

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 4:00

# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"

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
#BSUB -o gpu-%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

### - Interactive environment: sxm2sh -

# Load modules 
module unload cuda
module unload cudann
module load cuda/9.0
module load cudnn/v7.0.5-prod-cuda-9.0
module load python3/3.7.5

#Create a virtual environment for Python3
python3 -m venv hello_hpc

#Activate virtual environment
source ~/hello_hpc/bin/activate

echo "Installing Dependencies"
pip3 install -r requirements.txt

echo "Starting Training"
python3 -m cirtorch.examples.train outputs --training-dataset 'mapillary'
echo "Finished Training"

#scp -r /Users/alexanderholstrup/Desktop/train_val/zurich abho@login2.hpc.dtu.dk:~/Documents/VisualPlaceRecognition/data/mapillary/train_val
# ~/Documents/VisualPlaceRecognition/data/mapillary/train_val/zurich
# sxm2sh
# ~/Documents/VisualPlaceRecognition
# python3 -m cirtorch.examples.train outputs --training-dataset 'mapillary'
# [Errno 2] No such file or directory: 'data/mapillary/train_val/melbourne/query/postprocessed.csv'
