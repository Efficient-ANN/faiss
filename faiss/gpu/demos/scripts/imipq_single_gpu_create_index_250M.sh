#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p ict_gpu       #Fila (partition) a ser utilizada
#SBATCH --time=2:00:00
#SBATCH --account=petrobrasiageo
#SBATCH --exclusive         #Utilização exclusiva dos nós
#SBATCH --job-name="a1test"



# Num. GPUs: 1
# Tempo esperado: 2:00:00

module load boost/1.73_gnu cmake/3.17.3 gcc/7.4_sequana cuda/11.8
module load /scratch/app/modulos/sequana/current openmpi/gnu/4.0.1_sequana


set -x

BIGANN_DIR=/petrobr/parceirosbr/petrobrasiageo/willian.barreiros/git/gab/pqnns-multi-stream/data/sift1bi
RESULT_DIR=/petrobr/parceirosbr/petrobrasiageo/willian.barreiros/git/alan/faiss_imipq/demo_out
DEMO_DIR=/petrobr/parceirosbr/petrobrasiageo/willian.barreiros/git/alan/faiss_imipq/build/faiss/gpu/demos

./${DEMO_DIR}/demo_imipq_gpu_sift_m 128 15560 8 8 ${BIGANN_DIR}/bigann_learn.bvecs 32381696 ${BIGANN_DIR}/bigann_base.bvecs 250000000 ${BIGANN_DIR}/bigann_base.bvecs 0 "" 4 14 0 12 0 11 0 1 1 1 28991029248 ${RESULT_DIR}/coarse/coarse_imipq15560_32M_m_replica.bin ${RESULT_DIR}/index/index_imipq15560_32M_m_replica_250M.bin 0 0 &> ${RESULT_DIR}/outs/1_gpus_imipq1000M_15560_32M_m_shard_27GB_1_gpu_250M.txt 

