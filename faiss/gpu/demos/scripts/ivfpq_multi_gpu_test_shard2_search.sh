#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p ict_gpu       #Fila (partition) a ser utilizada
#SBATCH --time=4:00:00
#SBATCH --account=petrobrasiageo
#SBATCH --exclusive         #Utilização exclusiva dos nós
#SBATCH --job-name="a1test"

module load boost/1.73_gnu cmake/3.17.3 gcc/10.2 cuda/11.8 gcc/7.4
module load /scratch/app/modulos/sequana/current openmpi/gnu/4.0.1_sequana

set -x

BIGANN_DIR=/petrobr/parceirosbr/petrobrasiageo/willian.barreiros/git/gab/pqnns-multi-stream/data/sift1bi
RESULT_DIR=/petrobr/parceirosbr/petrobrasiageo/willian.barreiros/git/alan/faiss_imipq/demo_out
DEMO_DIR=/petrobr/parceirosbr/petrobrasiageo/willian.barreiros/git/alan/faiss_imipq/build/faiss/gpu/demos

${DEMO_DIR}/demo_ivfpq_gpu_sift_m 128 126491 8 8 ${BIGANN_DIR}/bigann_learn.bvecs 32381696 ${BIGANN_DIR}/bigann_base.bvecs 500000000 ${BIGANN_DIR}/bigann_base.bvecs 0 "" 12 13 0 12 0 11 0 1 16 2 1 28991029248 ${RESULT_DIR}/coarse/coarse_ivfpq126491_32M.bin ${RESULT_DIR}/index/index_ivfpq126491_32M_500M.bin 1 0 0 3 | tee ${RESULT_DIR}/outs/ivfpq_multi_gpu_test_shard2.txt

