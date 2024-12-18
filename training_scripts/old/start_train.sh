NCPUS = 8
MEM = 32
NGPUS = 1
GPU = 'A100'
NODES = 1

#PBS -N CheX_LP
#PBS -l select=1:ncpus=56:mem=250gb:ngpus=1:gpu_model=a100:phase=27,walltime=08:00:00
#PBS -M jplineb@clemson.edu
#PBS -j oe