#!/bin/sh
set -x
export SINGULARITYENV_SLURM_CPUS_ON_NODE=$SLURM_CPUS_ON_NODE
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export SINGULARITYENV_SLURM_JOB_ID=$SLURM_JOB_ID
# singularity shell --nv --no-home --cleanenv --env TMPDIR=$TMPDIR --env PYTHONNOUSERSITE=1 --bind $HOME/sw $WEATHER/containers/22cp0by16xv0sms5v6icf55mz8h7jimw-singularity-image-equivariant-posteriors.sif bash
# singularity shell --nv --no-home --cleanenv --env TMPDIR=$TMPDIR --env PYTHONNOUSERSITE=1 --bind $HOME/sw $WEATHER/containers/2n8lnfzlyq5gv694hsrsfw6dj4ysgvai-singularity-image-equivariant-posteriors.sif bash
# singularity shell --nv --no-home --cleanenv --env TMPDIR=$TMPDIR --env PYTHONNOUSERSITE=1 --bind $HOME/sw $WEATHER/containers/7mka1y8wv5nb3gan5y1aajdxgsdjswiz-singularity-image-equivariant-posteriors.sif bash
#singularity shell --nv --no-home --cleanenv --env TMPDIR=$TMPDIR --env PYTHONNOUSERSITE=1 --bind $HOME/sw $WEATHER/containers/vps21pis6xs4z3xvqnvfxg9ny55kd5lx-singularity-image-equivariant-posteriors.sif bash
singularity shell --nv --no-home --cleanenv --env TMPDIR=$TMPDIR --env PYTHONNOUSERSITE=1 --bind $HOME/sw --bind $HOME $WEATHER/containers/x86.sif bash
