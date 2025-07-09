#!/bin/sh
set -x
export SINGULARITYENV_SLURM_CPUS_ON_NODE=$SLURM_CPUS_ON_NODE
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export SINGULARITYENV_SLURM_JOB_ID=$SLURM_JOB_ID
export APPTAINERENV_PREPEND_PATH=$HOME/bin
singularity shell --nv --no-home --cleanenv --env TMPDIR=$TMPDIR --env PYTHONNOUSERSITE=1 -H /proj/heal_pangu/users/x_hamli  --bind $HOME $WEATHER/containers/x86.sif bash
