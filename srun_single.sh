#!/bin/sh
set -x
GPU=${GPU:-1}
TIMESTAMP=$(date -d "today" +"%Y%m%d%H%M")
SCRIPT_FILE=slurm_tmp/$TIMESTAMP.sh
mkdir -p slurm_tmp
mkdir -p slurm_log
cat <<EOM >$SCRIPT_FILE
#!/bin/bash
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES
export SINGULARITYENV_SLURM_JOB_ID=\$SLURM_JOB_ID
export SINGULARITYENV_SLURM_CPUS_ON_NODE=\$SLURM_CPUS_ON_NODE
export SINGULARITYENV_BATCHSIZE=\$BATCHSIZE
export SINGULARITYENV_LEADTIME=\$LEADTIME
export SINGULARITYENV_EID=\$EID
export SINGULARITYENV_NOCOPY=1

singularity exec --nv --cleanenv --no-home --env COLUMNS=200 --env LINES=60 -H /proj/heal_pangu/users/x_hamli --env SLURM_ARRAY_TASK_ID=\$SLURM_ARRAY_TASK_ID --env PYTHONNOUSERSITE=1 $WEATHER/containers/x86.sif sh $@
EOM

sbatch -o slurm_log/slurm_%x.%j.log -A $SLURM_PROJECT --gpus=${GPU} -t 3-00:00:00 $SCRIPT_FILE
