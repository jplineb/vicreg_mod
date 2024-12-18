#!/bin/bash
#PBS -N vindrcxr_index
#PBS -l select=1:ncpus=36:mem=175gb,walltime=8:00:00
#PBS -M jplineb@clemson.edu
#PBS -j oe

echo "----------------------------"
echo "Executing on host: "$HOSTNAME
echo "Building vindr-cxr index"
echo "----------------------------"

# jobperf -record -w -rate 10s -http &

# Load modules
module load anaconda3/2022.05-gcc/9.5.0
module load cuda/11.6.2-gcc/9.5.0

# Activate Conda and CD
source activate pda
cd /home/jplineb/VICReg/vicreg_mod/

# Build Index
python notebooks/build_vindrcxr_index.py