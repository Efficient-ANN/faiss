#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p ict_gpu       #Fila (partition) a ser utilizada
#SBATCH --time=0:30:00
#SBATCH --account=petrobrasiageo
#SBATCH --exclusive         #Utilização exclusiva dos nós
#SBATCH --job-name="a1test"

module load boost/1.73_gnu cmake/3.17.3 gcc/10.2 gcc/7.4 nvhpc/22.3
module load cuda/11.0
module load /scratch/app/modulos/sequana/current openmpi/gnu/4.0.1_sequana

set -x

echo $CUDA_HOME

ls $CUDA_HOME

nvprof --version

RESULT_DIR=/petrobr/parceirosbr/petrobrasiageo/willian.barreiros/git/a/faiss_imipq/demo_out
DEMO_DIR=/petrobr/parceirosbr/petrobrasiageo/willian.barreiros/git/a/faiss_imipq/build/faiss/gpu/demos

while true; do nvidia-smi >> gpu.log; sleep 2; done &

mpirun -np 1  ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 128 16 8 8 "" 624 "" 256 "" 0 "" 6 7 6 7 6 7 0 8 1 1 0 1 4194304 ${RESULT_DIR}/coarse/index_coarse_imipq16_624_m_replica_rand.bin ${RESULT_DIR}/index/index_imipq16_624_m_replica_256_rand.bin 0 0 0 1 1 | tee ${RESULT_DIR}/outs/imipq_mpi_simple_prof_shard_1_build.txt

nvprof --print-gpu-trace mpirun -np 1 ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 128 16 8 8 "" 624 "" 256 "" 0 "" 6 7 6 7 6 7 0 8 1 1 0 1 4194304 ${RESULT_DIR}/coarse/index_coarse_imipq16_624_m_replica_rand.bin ${RESULT_DIR}/index/index_imipq16_624_m_replica_256_rand.bin 1 0 0 1 1 | tee ${RESULT_DIR}/outs/imipq_mpi_simple_prof_shard_1_search.txt

mpirun -np 2  ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 128 16 8 8 "" 624 "" 256 "" 0 "" 6 7 6 7 6 7 0 8 2 1 0 1 4194304 ${RESULT_DIR}/coarse/index_coarse_imipq16_624_m_replica_rand.bin ${RESULT_DIR}/index/index_imipq16_624_m_replica_256_rand.bin 0 0 0 1 1 | tee ${RESULT_DIR}/outs/imipq_mpi_simple_prof_shard_2_build.txt

nvprof --print-gpu-trace mpirun -np 2  ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 128 16 8 8 "" 624 "" 256 "" 0 "" 6 7 6 7 6 7 0 8 2 1 0 1 4194304 ${RESULT_DIR}/coarse/index_coarse_imipq16_624_m_replica_rand.bin ${RESULT_DIR}/index/index_imipq16_624_m_replica_256_rand.bin 1 0 0 1 1 | tee ${RESULT_DIR}/outs/imipq_mpi_simple_prof_shard_2_search.txt

mpirun -np 2  ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 128 16 8 8 "" 624 "" 256 "" 0 "" 6 7 6 7 6 7 0 8 2 1 0 0 4194304 ${RESULT_DIR}/coarse/index_coarse_imipq16_624_m_replica_rand.bin ${RESULT_DIR}/index/index_imipq16_624_m_replica_256_rand.bin 0 0 0 1 1 | tee ${RESULT_DIR}/outs/imipq_mpi_simple_prof_replica_2_build.txt

nvprof --print-gpu-trace mpirun -np 2  ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 128 16 8 8 "" 624 "" 256 "" 0 "" 6 7 6 7 6 7 0 8 2 1 0 0 4194304 ${RESULT_DIR}/coarse/index_coarse_imipq16_624_m_replica_rand.bin ${RESULT_DIR}/index/index_imipq16_624_m_replica_256_rand.bin 1 0 0 1 1 | tee ${RESULT_DIR}/outs/imipq_mpi_simple_prof_replica_2_search.txt
