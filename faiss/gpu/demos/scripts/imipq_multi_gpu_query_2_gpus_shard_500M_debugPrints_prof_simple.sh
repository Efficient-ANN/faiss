#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p ict_gpu       #Fila (partition) a ser utilizada
#SBATCH --time=0:01:00
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

BIGANN_DIR=/petrobr/parceirosbr/petrobrasiageo/willian.barreiros/git/gab/pqnns-multi-stream/data/sift1bi
RESULT_DIR=/petrobr/parceirosbr/petrobrasiageo/willian.barreiros/git/alan/faiss_imipq/demo_out
DEMO_DIR=/petrobr/parceirosbr/petrobrasiageo/willian.barreiros/git/alan/faiss_imipq/build/faiss/gpu/demos

${DEMO_DIR}/demo_imipq_gpu_sift_m 128 16 8 8 ${BIGANN_DIR}/bigann_learn.bvecs 128 ${BIGANN_DIR}/bigann_base.bvecs 256 ${BIGANN_DIR}/bigann_base.bvecs 0 "" 9 14 6 7 6 8 0 8 2 1 4194304 ${RESULT_DIR}/coarse/coarse_imipq16_128_m_replica.bin ${RESULT_DIR}/index/index_imipq16_128_m_replica_256.bin 0 0 0 1 1 | tee ${RESULT_DIR}/outs/1_gpus_imipq256_16_128_m_shard_4MB_2_gpu_128_debugPrints_prof_simple.txt

nvprof --print-gpu-trace ${DEMO_DIR}/demo_imipq_gpu_sift_m 128 16 8 8 ${BIGANN_DIR}/bigann_learn.bvecs 128 ${BIGANN_DIR}/bigann_base.bvecs 256 ${BIGANN_DIR}/bigann_base.bvecs 0 "" 9 14 6 7 6 8 0 8 2 1 4194304 ${RESULT_DIR}/coarse/coarse_imipq16_128_m_replica.bin ${RESULT_DIR}/index/index_imipq16_128_m_replica_256.bin 1 0 0 1 1 | tee ${RESULT_DIR}/outs/1_gpus_imipq256_16_128_m_shard_4MB_2_gpu_128_debugPrints_prof_simple.txt
faiss/gpu/demos/scripts/imipq_multi_gpu_query_2_gpus_shard_500M_debugPrints_prof_simple.sh