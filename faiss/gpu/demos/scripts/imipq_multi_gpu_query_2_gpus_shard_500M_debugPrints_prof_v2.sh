#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p ict_gpu       #Fila (partition) a ser utilizada
#SBATCH --time=3:00:00
#SBATCH --account=petrobrasiageo
#SBATCH --exclusive         #Utilização exclusiva dos nós
#SBATCH --job-name="a1test"

module load boost/1.73_gnu cmake/3.17.3 gcc/10.2 gcc/7.4 nvhpc/22.3
#module load cuda/11.8
module load cuda/11.4
module load /scratch/app/modulos/sequana/current openmpi/gnu/4.0.1_sequana

set -x

echo $CUDA_HOME

ls $CUDA_HOME

nvprof --version

BIGANN_DIR=/petrobr/parceirosbr/petrobrasiageo/willian.barreiros/git/g/pqnns-multi-stream/data/sift1bi
RESULT_DIR=/petrobr/parceirosbr/petrobrasiageo/willian.barreiros/git/a/faiss_imipq/demo_out
DEMO_DIR=/petrobr/parceirosbr/petrobrasiageo/willian.barreiros/git/a/faiss_imipq/build/faiss/gpu/demos

nvprof --print-gpu-trace ${DEMO_DIR}/demo_imipq_gpu_sift_m_v2 128 15560 8 8 ${BIGANN_DIR}/bigann_learn.bvecs 32381696 ${BIGANN_DIR}/bigann_base.bvecs 500000000 ${BIGANN_DIR}/bigann_base.bvecs 0 "" 9 14 6 7 6 8 0 48 2 1 28991029248 ${RESULT_DIR}/coarse/coarse_imipq15560_32M_m_replica.bin ${RESULT_DIR}/index/index_imipq15560_32M_m_replica_500M.bin 1 0 0 1 1 | tee ${RESULT_DIR}/outs/1_gpus_imipq1000M_15560_32M_m_shard_27GB_2_gpu_500M_debugPrints_prof.txt
