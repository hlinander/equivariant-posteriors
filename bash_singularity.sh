#!/bin/sh
set -x
export SINGULARITYENV_SLURM_CPUS_ON_NODE=$SLURM_CPUS_ON_NODE
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
singularity shell --nv --no-home --cleanenv --env PYTHONNOUSERSITE=1 $WEATHER/containers/dn5794ypyy9aq0jyxwlffvckhqm6g188-singularity-image-equivariant-posteriors.img bash
