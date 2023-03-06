#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p ict_gpu       #Fila (partition) a ser utilizada
#SBATCH --time=9:00:00
#SBATCH --account=petrobrasiageo
#SBATCH --exclusive         #Utilização exclusiva dos nós
#SBATCH --job-name="a1test"

module load boost/1.73_gnu cmake/3.17.3 gcc/7.4_sequana cuda/11.8
module load /scratch/app/modulos/sequana/current openmpi/gnu/4.0.1_sequana

set -x

BIGANN_DIR=/petrobr/parceirosbr/petrobrasiageo/willian.barreiros/git/gab/pqnns-multi-stream/data/sift1bi
RESULT_DIR=/petrobr/parceirosbr/petrobrasiageo/willian.barreiros/git/alan/faiss_imipq/demo_out
DEMO_DIR=/petrobr/parceirosbr/petrobrasiageo/willian.barreiros/git/alan/faiss_imipq/build/faiss/gpu/demos

${DEMO_DIR}/demo_imipq_gpu_sift_m 128 15560 8 8 ${BIGANN_DIR}/bigann_learn.bvecs 32381696 ${BIGANN_DIR}/bigann_base.bvecs 1000000000 ${BIGANN_DIR}/bigann_base.bvecs 0 "" 4 14 0 12 0 11 0 2 2 0 28991029248 ${RESULT_DIR}/coarse/coarse_imipq15560_32M_m_replica.bin ${RESULT_DIR}/index/index_imipq15560_32M_m_replica.bin 1 0 &> ${RESULT_DIR}/outs/1_gpus_imipq1000M_15560_32M_m_27GB_query_2_gpu.txt
