#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p ict_gpu       #Fila (partition) a ser utilizada
#SBATCH --time=8:00:00
#SBATCH --account=petrobrasiageo
#SBATCH --exclusive         #Utilização exclusiva dos nós
#SBATCH --job-name="a1test"

module load boost/1.73_gnu cmake/3.17.3 gcc/10.2 gcc/7.4 
#module nvhpc/22.3
module load cuda/11.0
module load /scratch/app/modulos/sequana/current openmpi/gnu/4.0.1_sequana
set -x

BIGANN_DIR=/petrobr/parceirosbr/petrobrasiageo/willian.barreiros/git/g/pqnns-multi-stream/data/sift1bi
RESULT_DIR=/petrobr/parceirosbr/petrobrasiageo/willian.barreiros/git/a/faiss_imipq/demo_out
DEMO_DIR=/petrobr/parceirosbr/petrobrasiageo/willian.barreiros/git/a/faiss_imipq/build/faiss/gpu/demos

# IMIPQ CPU
mpirun -np 1 ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 128 15560 8 8 ${BIGANN_DIR}/bigann_learn.bvecs 32381696 ${BIGANN_DIR}/bigann_base.bvecs 1000000000 ${BIGANN_DIR}/bigann_query.bvecs 0 ${BIGANN_DIR}/gnd/idx_1000M.ivecs 0 1 0 12 0 11 0 48 1 1 1 1 28991029248 ${RESULT_DIR}/coarse/coarse_imipq15560_32M_m_replica_full.bin ${RESULT_DIR}/index/index_imipq15560_32M_m_replica.bin 1 0 0 1 1 1 1 0 | tee ${RESULT_DIR}/outs/cpu_imipq1000M_15560_32M_m_27GB_numQueries1.txt

# IVFPQ CPU
mpirun -np 1 ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 128 126491 8 8 ${BIGANN_DIR}/bigann_learn.bvecs 32381696 ${BIGANN_DIR}/bigann_base.bvecs 1000000000 ${BIGANN_DIR}/bigann_query.bvecs 0 ${BIGANN_DIR}/gnd/idx_1000M.ivecs 0 1 0 12 0 11 0 48 1 1 1 1 28991029248 ${RESULT_DIR}/coarse/coarse_ivfpq126491_32M_full.bin ${RESULT_DIR}/index/index_ivfpq126491_32M.bin 1 0 0 1 1 1 0 0 | tee ${RESULT_DIR}/outs/cpu_ivfpq1000M_126491_32M_m_27GB_numQueries1.txt

# IMIPQ single process
nvprof --print-gpu-trace mpirun -np 1 ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 128 15560 8 8 ${BIGANN_DIR}/bigann_learn.bvecs 32381696 ${BIGANN_DIR}/bigann_base.bvecs 1000000000 ${BIGANN_DIR}/bigann_query.bvecs 0 "" 3 4 6 7 6 7 0 48 1 1 1 1 28991029248 ${RESULT_DIR}/coarse/coarse_imipq15560_32M_m_replica_full.bin ${RESULT_DIR}/index/index_imipq15560_32M_m_replica.bin 1 0 0 1 0 1 1 1 | tee ${RESULT_DIR}/outs/mpi_1_1gpu_imipq1000M_15560_32M_m_27GB_1Mqueries.txt

# IMIPQ multi process

# Indexing
mpirun -np 2 ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 128 15560 8 8 ${BIGANN_DIR}/bigann_learn.bvecs 32381696 ${BIGANN_DIR}/bigann_base.bvecs 500000000 ${BIGANN_DIR}/bigann_query.bvecs 0 "" 11 12 6 7 6 7 0 48 1 1 0 1 28991029248 ${RESULT_DIR}/coarse/coarse_imipq15560_32M_m_replica_full.bin ${RESULT_DIR}/index/index_imipq15560_32M_m_replica_500M.bin 0 0 0 1 0 1 1 1 | tee ${RESULT_DIR}/outs/mpi_2_1gpu_imipq500M_15560_32M_m_27GB_1Mqueries.txt

# Searching
nvprof --print-gpu-trace --profile-child-processes mpirun -np 2 ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 128 15560 8 8 ${BIGANN_DIR}/bigann_learn.bvecs 32381696 ${BIGANN_DIR}/bigann_base.bvecs 500000000 ${BIGANN_DIR}/bigann_query.bvecs 0 "" 11 12 6 7 6 7 0 48 1 1 0 1 28991029248 ${RESULT_DIR}/coarse/coarse_imipq15560_32M_m_replica_full.bin ${RESULT_DIR}/index/index_imipq15560_32M_m_replica_500M.bin 1 0 0 1 0 1 1 1 | tee ${RESULT_DIR}/outs/mpi_2_1gpu_imipq500M_15560_32M_m_27GB_1Mqueries.txt
