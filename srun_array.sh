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
singularity exec --nv --cleanenv --no-home --env COLUMNS=200 --env LINES=60 --env SLURM_ARRAY_TASK_ID=\$SLURM_ARRAY_TASK_ID --env PYTHONNOUSERSITE=1 $ENTVAR/equivariant-posteriors/image.img sh ${@:2}
EOM

sbatch --array=0-$(($1-1)) -o slurm_log/slurm_%x.%j.log -A $SLURM_PROJECT -p $SLURM_PARTITION -N 1 -n 1 --gpus-per-node=${GPU} -t 2-10:00:00 $SCRIPT_FILE
