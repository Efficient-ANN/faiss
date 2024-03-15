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
# FAISS_DIR=/petrobr/parceirosbr/petrobrasiageo/willian.barreiros/git/a/faiss_imipq
FAISS_DIR=/home_cerberus/speed/willianjunior/git/faiss_imipq

# DEMO_DIR=${FAISS_DIR}/build/faiss/gpu/demos
DEMO_DIR=${FAISS_DIR}/faiss/gpu/demos
RESULT_DIR=${FAISS_DIR}/demo_out
GIST_SCRIPT_DIR=${FAISS_DIR}/faiss/gpu/demos/scripts/mpi/gist

MAX_MEM=3221225472
MEM_STR=3GB
N_THREADS=8
USE_GPU=1
N_GPUS=1
USE_SHARD=1
PRINT_GPU_MEM=0
# MAX_MEM=28991029248
# MEM_STR=27GB

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

# IMIPQ
CENTROID_LIST="128 226 256 333 419 500 746 1266 2059"
USE_IMI=1
USE_PRECOMP=1
for N_CENTROIDS in ${CENTROID_LIST};
do
    for SEARCH in 0 1;
    do
        ${GIST_SCRIPT_DIR}/gist_wrapper.sh
    done
done

# IVFPQ
USE_IMI=0
CENTROID_LIST="250 500 1000 2000 4000"
for N_CENTROIDS in ${CENTROID_LIST};
do
    USE_PRECOMP=1
    SEARCH=0
    ${GIST_SCRIPT_DIR}/gist_wrapper.sh

    SEARCH=1
    for USE_PRECOMP in 1;
    do
        ${GIST_SCRIPT_DIR}/gist_wrapper.sh
    done
done
