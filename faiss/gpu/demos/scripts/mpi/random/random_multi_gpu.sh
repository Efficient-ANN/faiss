#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p ict_gpu       #Fila (partition) a ser utilizada
#SBATCH --time=6:00:00
#SBATCH --account=petrobrasiageo
#SBATCH --exclusive         #Utilização exclusiva dos nós
#SBATCH --job-name="a1test"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guns945@gmail.com

module load boost/1.73_gnu cmake/3.17.3 gcc/10.2 gcc/7.4 
module load cuda/11.0
module load /scratch/app/modulos/sequana/current openmpi/gnu/4.0.1_sequana
set -x

# FAISS_DIR=/home/alanp/git/faiss_imipq
FAISS_DIR=/petrobr/parceirosbr/petrobrasiageo/willian.barreiros/git/a/faiss_imipq
# FAISS_DIR=/home_cerberus/speed/willianjunior/git/faiss_imipq

DEMO_DIR=${FAISS_DIR}/build/faiss/gpu/demos
# DEMO_DIR=${FAISS_DIR}/faiss/gpu/demos
RESULT_DIR=${FAISS_DIR}/demo_out
RANDOM_DIR=${FAISS_DIR}/faiss/gpu/demos/scripts/mpi/random
PROF_SCRIPT_DIR=${FAISS_DIR}/faiss/gpu/demos/scripts/mpi/sift

# MAX_MEM=3221225472
# MEM_STR=3GB
N_THREADS=8
USE_GPU=1
USE_SHARD=1
PRINT_GPU_MEM=0
MAX_MEM=8589934592
MEM_STR=8GB
# MAX_MEM=17179869184
# MEM_STR=16GB

export DEMO_DIR
export RESULT_DIR
export MAX_MEM
export MEM_STR
export N_THREADS
export USE_GPU
export N_GPUS
export USE_SHARD
export USE_PRECOMP
export USE_IMI
export N_CENTROIDS
export SEARCH
export PRINT_GPU_MEM
export Q_INIT
export Q_END
export N_INIT
export N_END
export K_INIT
export K_END
export PROF_SCRIPT_DIR
export USE_PROF
export BASE_SIZE
export NUM_DIMENSIONS
export RANDOM_DIR
export CENTROID_LIST
export USE_NVPROF
export COPY_PER_SHARD

# IMIPQ

USE_NVPROF=1
COPY_PER_SHARD=1

# 1M
BASE_SIZE=1000000
NUM_DIMENSIONS=128
CENTROID_LIST="2059"
USE_IMI=1
USE_PRECOMP=1

Q_INIT=3
Q_END=4
${RANDOM_DIR}/random_multi_gpu_run.sh

Q_INIT=7
Q_END=8
${RANDOM_DIR}/random_multi_gpu_run.sh

# 10M
BASE_SIZE=10000000
NUM_DIMENSIONS=128
CENTROID_LIST="4276"
USE_IMI=1
USE_PRECOMP=1

Q_INIT=3
Q_END=4
${RANDOM_DIR}/random_multi_gpu_run.sh
Q_INIT=7
Q_END=8
${RANDOM_DIR}/random_multi_gpu_run.sh

# 100M
BASE_SIZE=100000000
NUM_DIMENSIONS=128
CENTROID_LIST="8316"
USE_IMI=1
USE_PRECOMP=1

Q_INIT=3
Q_END=4
${RANDOM_DIR}/random_multi_gpu_run.sh
Q_INIT=7
Q_END=8
${RANDOM_DIR}/random_multi_gpu_run.sh
