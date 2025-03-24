#!/bin/bash

snakemake \
    --executor slurm \
    --jobs 1 \
    --latency-wait 60 \
    --printshellcmds \
    --use-conda \
    # --rerun-incomplete \