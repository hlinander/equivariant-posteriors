#!/bin/sh
set -x
export SINGULARITYENV_SLURM_CPUS_ON_NODE=$SLURM_CPUS_ON_NODE
singularity shell --nv --no-home --cleanenv --env PYTHONNOUSERSITE=1 $ENTVAR/equivariant-posteriors/image.img bash
