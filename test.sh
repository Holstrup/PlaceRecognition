#!/bin/sh
### General options
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J Test
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 00:30
# request 5GB of system-memory
#BSUB -R "rusage[mem=10GB]"
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
### echo "Created Virt. E."
### Activate virtual environment
source ~/hello_hpc/bin/activate
### echo "Installing Dependencies"
### echo $PWD
### pip3 install -r requirements.txt
echo "Starting Training"
##python3 cirtorch/examples/test_mapillary.py --datasets 'mapillary' --network-path data/exp_outputs1/mapillary_resnet50_gem_contrastive_m0.70_adam_lr1.0e-06_wd1.0e-06_nnum5_qsize2000_psize20000_bsize5_uevery5_imsize1024/model_epoch480.pth.tar

###python3 cirtorch/examples/test_mapillary.py --datasets 'mapillary' --network-path data/exp_outputs1/mapillary_resnet50_gem_contrastive_m0.70_adam_lr1.0e-06_wd1.0e-06_nnum5_qsize2000_psize20000_bsize5_uevery5_imsize1024/model_epoch480.pth.tar --whitening-network data/localnet_exp/contrastive_loss_fulldataset_lr0.0001/model_epoch_150.pth --whitening-outdim 2048
python3 cirtorch/examples/test_mapillary.py --datasets 'mapillary' --network-path data/exp_outputs1/mapillary_resnet50_gem_contrastive_m0.70_adam_lr1.0e-06_wd1.0e-06_nnum5_qsize2000_psize20000_bsize5_uevery5_imsize1024/model_epoch480.pth.tar --whitening-network data/localnet_exp/contrastive_loss_fulldataset_lr0.0001/model_epoch_140.pth --whitening-outdim 2048
###python3 cirtorch/examples/extract_tuples.py --datasets 'mapillary' --network-path data/exp_outputs1/mapillary_resnet50_gem_contrastive_m0.70_adam_lr1.0e-06_wd1.0e-06_nnum5_qsize2000_psize20000_bsize5_uevery5_imsize1024/model_epoch480.pth.tar --whitening-network data/localnet_exp/contrastive_loss_fulldataset_lr0.0001/model_epoch_160.pth --whitening-outdim 2048
###python3 cirtorch/examples/extract_tuples.py --datasets 'mapillary' --network-path data/outputs/mapillary_resnet50_gem_whiten_contrastive_m0.70_adam_lr1.0e-06_wd1.0e-06_nnum6_qsize2000_psize20000_bsize5_uevery5_imsize320/model_epoch180.pth.tar
###python3 cirtorch/examples/test_mapillary.py --datasets 'mapillary' --network-path data/outputs/mapillary_resnet501024_gem_whiten_contrastive_m0.70_adam_lr1.0e-06_wd1.0e-06_nnum6_qsize2000_psize20000_bsize5_uevery5_imsize320/model_epoch180.pth.tar
###python3 -m cirtorch.examples.test_mapillary --network-path 'data/references/MSLS_resnet50_GeM_480_CL.pth' --datasets 'mapillary'
echo "Finished Training"
