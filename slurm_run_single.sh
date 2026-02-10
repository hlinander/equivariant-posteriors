#!/bin/sh
set -x
GPU=${GPU:-1}
TIMESTAMP=$(date -d "today" +"%Y%m%d%H%M")
SCRIPT_FILE=slurm_tmp/$TIMESTAMP.sh
mkdir -p slurm_tmp
mkdir -p slurm_log
cat <<EOM >$SCRIPT_FILE
#!/bin/bash
$@
EOM

sbatch -o slurm_log/slurm_%x.%j.log -A $SLURM_PROJECT -p $SLURM_PARTITION --gpus=${GPU} -t 2-00:00:00 $SCRIPT_FILE
