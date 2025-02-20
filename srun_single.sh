#!/bin/sh
set -x
GPU=${GPU:-A40:1}
TIMESTAMP=$(date -d "today" +"%Y%m%d%H%M")
SCRIPT_FILE=slurm_tmp/$TIMESTAMP.sh
mkdir -p slurm_tmp
mkdir -p slurm_log
cat <<EOM >$SCRIPT_FILE
#!/bin/bash
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES
export SINGULARITYENV_SLURM_JOB_ID=\$SLURM_JOB_ID
export SINGULARITYENV_SLURM_CPUS_ON_NODE=\$SLURM_CPUS_ON_NODE

singularity exec --nv --cleanenv --no-home --env COLUMNS=200 --env LINES=60 --env SLURM_ARRAY_TASK_ID=\$SLURM_ARRAY_TASK_ID --env PYTHONNOUSERSITE=1 $WEATHER/containers/7mka1y8wv5nb3gan5y1aajdxgsdjswiz-singularity-image-equivariant-posteriors.sif sh $@
EOM

sbatch -o slurm_log/slurm_%x.%j.log -A $SLURM_PROJECT -p $SLURM_PARTITION -N 1 -n 1 --gpus-per-node=${GPU} -t 3-10:00:00 $SCRIPT_FILE
