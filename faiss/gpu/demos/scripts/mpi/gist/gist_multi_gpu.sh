#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p ict_gpu       #Fila (partition) a ser utilizada
#SBATCH --time=6:00:00
#SBATCH --account=petrobrasiageo
#SBATCH --exclusive         #Utilização exclusiva dos nós
#SBATCH --job-name="a1test"

# module load boost/1.73_gnu cmake/3.17.3 gcc/10.2 gcc/7.4 
# module load cuda/11.0
# module load /scratch/app/modulos/sequana/current openmpi/gnu/4.0.1_sequana
# set -x

# FAISS_DIR=/home/alanp/git/faiss_imipq
FAISS_DIR=/petrobr/parceirosbr/petrobrasiageo/willian.barreiros/git/a/faiss_imipq

DEMO_DIR=${FAISS_DIR}/build/faiss/gpu/demos
RESULT_DIR=${FAISS_DIR}/demo_out
GIST_SCRIPT_DIR=${FAISS_DIR}/faiss/gpu/demos/scripts/mpi/gist
PROF_SCRIPT_DIR=${FAISS_DIR}/faiss/gpu/demos/scripts/mpi/sift

# MAX_MEM=3221225472
# MEM_STR=3GB
N_THREADS=8
USE_GPU=1
USE_SHARD=1
PRINT_GPU_MEM=0
Q_INIT=10
Q_END=11
MAX_MEM=8589934592
MEM_STR=8GB

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

# IMIPQ
CENTROID_LIST="500"
USE_IMI=1
USE_PRECOMP=1

N_GPUS_LIST="1 2"
SEARCH_LIST="0 1"

N_INIT=7
N_END=8
K_INIT=7
K_END=8

for N_GPUS in ${N_GPUS_LIST};
do
    for N_CENTROIDS in ${CENTROID_LIST};
    do
        for SEARCH in ${SEARCH_LIST};
        do
            ${GIST_SCRIPT_DIR}/gist_multi_gpu_wrapper.sh
        done
    done
done


N_INIT=6
N_END=7
K_INIT=6
K_END=7

for N_GPUS in ${N_GPUS_LIST};
do
    for N_CENTROIDS in ${CENTROID_LIST};
    do
        for SEARCH in ${SEARCH_LIST};
        do
            ${GIST_SCRIPT_DIR}/gist_multi_gpu_wrapper.sh
        done
    done
done