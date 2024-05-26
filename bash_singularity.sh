#!/bin/sh
set -x
export SINGULARITYENV_SLURM_CPUS_ON_NODE=$SLURM_CPUS_ON_NODE
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
singularity shell --nv --no-home --cleanenv --env PYTHONNOUSERSITE=1 $ENTVAR/equivariant-posteriors/image.img bash
