#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p ict_gpu       #Fila (partition) a ser utilizada
#SBATCH --time=6:00:00
#SBATCH --account=petrobrasiageo
#SBATCH --exclusive         #Utilização exclusiva dos nós
#SBATCH --job-name="a1test"

# module load boost/1.73_gnu cmake/3.17.3 gcc/10.2 gcc/7.4 
# #module nvhpc/22.3
# module load cuda/11.0
# module load /scratch/app/modulos/sequana/current openmpi/gnu/4.0.1_sequana
# set -x

BASE_DIR=/petrobr/parceirosbr/petrobrasiageo/willian.barreiros/git/g/pqnns-multi-stream/data
BIGANN_DIR=${BASE_DIR}/sift1bi
FAISS_DIR=/petrobr/parceirosbr/petrobrasiageo/willian.barreiros/git/a/faiss_imipq
DEMO_DIR=${FAISS_DIR}/build/faiss/gpu/demos
RESULT_DIR=${FAISS_DIR}/demo_out
PROF_SCRIPT_DIR=${FAISS_DIR}/faiss/gpu/demos/scripts/mpi/prof

N_THREADS=48
USE_SHARD=1
MAX_MEM=28991029248
USE_PRECOMP=1
USE_IMI=1
USE_GPU=1
PRINT_GPU_MEM=0

# IMIPQ single process
N_GPUS=1
## Indexing
SEARCH=0
mpirun -np 1 ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 128 15560 8 8 ${BIGANN_DIR}/bigann_learn.bvecs 32381696 ${BIGANN_DIR}/bigann_base.bvecs 1000000000 ${BIGANN_DIR}/bigann_query.bvecs 0 "" 3 4 6 7 6 7 0 ${N_THREADS} ${N_GPUS} ${USE_SHARD} 0 1 ${MAX_MEM} ${RESULT_DIR}/coarse/coarse_imipq15560_32M_m_replica_full.bin ${RESULT_DIR}/index/index_imipq15560_32M_m_replica.bin ${SEARCH} 0 0 1 0 ${USE_PRECOMP} ${USE_IMI} ${USE_GPU} ${PRINT_GPU_MEM} 0 | tee ${RESULT_DIR}/outs/mpi_1_1gpu_imipq1000M_15560_32M_m_27GB_10Kqueries_search0.txt
## Searching
SEARCH=1
mpirun -np 1 ${PROF_SCRIPT_DIR}/nvprof_wrapper.sh ${RESULT_DIR}/outs/mpi_1_1gpu_imipq1000M_15560_32M_m_27GB_10Kqueries_prof ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 128 15560 8 8 ${BIGANN_DIR}/bigann_learn.bvecs 32381696 ${BIGANN_DIR}/bigann_base.bvecs 1000000000 ${BIGANN_DIR}/bigann_query.bvecs 0 "" 3 4 6 7 6 7 0 ${N_THREADS} ${N_GPUS} ${USE_SHARD} 0 1 ${MAX_MEM} ${RESULT_DIR}/coarse/coarse_imipq15560_32M_m_replica_full.bin ${RESULT_DIR}/index/index_imipq15560_32M_m_replica.bin ${SEARCH} 0 0 1 0 ${USE_PRECOMP} ${USE_IMI} ${USE_GPU} ${PRINT_GPU_MEM} 0 | tee ${RESULT_DIR}/outs/mpi_1_1gpu_imipq1000M_15560_32M_m_27GB_10Kqueries_search1.txt

# IMIPQ multi process
N_GPUS=2
## Indexing
SEARCH=0
mpirun -np 2 ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 128 15560 8 8 ${BIGANN_DIR}/bigann_learn.bvecs 32381696 ${BIGANN_DIR}/bigann_base.bvecs 500000000 ${BIGANN_DIR}/bigann_query.bvecs 0 "" 11 12 6 7 6 7 0 ${N_THREADS} ${N_GPUS} ${USE_SHARD} 0 1 ${MAX_MEM} ${RESULT_DIR}/coarse/coarse_imipq15560_32M_m_replica_full.bin ${RESULT_DIR}/index/index_imipq15560_32M_m_replica_500M.bin ${SEARCH} 0 0 1 0 ${USE_PRECOMP} ${USE_IMI} ${USE_GPU} ${PRINT_GPU_MEM} 0 | tee ${RESULT_DIR}/outs/mpi_2_1gpu_imipq500M_15560_32M_m_27GB_1Mqueries_search0.txt
## Searching
SEARCH=1
mpirun -np 2 ${PROF_SCRIPT_DIR}/nvprof_wrapper.sh ${RESULT_DIR}/outs/mpi_2_1gpu_imipq500M_15560_32M_m_27GB_1Mqueries_prof ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 128 15560 8 8 ${BIGANN_DIR}/bigann_learn.bvecs 32381696 ${BIGANN_DIR}/bigann_base.bvecs 500000000 ${BIGANN_DIR}/bigann_query.bvecs 0 "" 11 12 6 7 6 7 0 ${N_THREADS} ${N_GPUS} ${USE_SHARD} 0 1 ${MAX_MEM} ${RESULT_DIR}/coarse/coarse_imipq15560_32M_m_replica_full.bin ${RESULT_DIR}/index/index_imipq15560_32M_m_replica_500M.bin ${SEARCH} 0 0 1 0 ${USE_PRECOMP} ${USE_IMI} ${USE_GPU} ${PRINT_GPU_MEM} 0 | tee ${RESULT_DIR}/outs/mpi_2_1gpu_imipq500M_15560_32M_m_27GB_1Mqueries_search1.txt

# NO PROF
SEARCH=1

N_GPUS=1
mpirun -np 1 ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 128 15560 8 8 ${BIGANN_DIR}/bigann_learn.bvecs 32381696 ${BIGANN_DIR}/bigann_base.bvecs 250000000 ${BIGANN_DIR}/bigann_query.bvecs 0 "" 11 12 6 7 6 7 0 ${N_THREADS} ${N_GPUS} ${USE_SHARD} 0 1 ${MAX_MEM} ${RESULT_DIR}/coarse/coarse_imipq15560_32M_m_replica_full.bin ${RESULT_DIR}/index/index_imipq15560_32M_m_replica_250M.bin ${SEARCH} 0 0 1 0 ${USE_PRECOMP} ${USE_IMI} ${USE_GPU} ${PRINT_GPU_MEM} 0 | tee ${RESULT_DIR}/outs/mpi_1_1gpu_imipq250M_15560_32M_m_27GB_1Mqueries_search1_noprof.txt

N_GPUS=2
mpirun -np 2 ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 128 15560 8 8 ${BIGANN_DIR}/bigann_learn.bvecs 32381696 ${BIGANN_DIR}/bigann_base.bvecs 500000000 ${BIGANN_DIR}/bigann_query.bvecs 0 "" 11 12 6 7 6 7 0 ${N_THREADS} ${N_GPUS} ${USE_SHARD} 0 1 ${MAX_MEM} ${RESULT_DIR}/coarse/coarse_imipq15560_32M_m_replica_full.bin ${RESULT_DIR}/index/index_imipq15560_32M_m_replica_500M.bin ${SEARCH} 0 0 1 0 ${USE_PRECOMP} ${USE_IMI} ${USE_GPU} ${PRINT_GPU_MEM} 0 | tee ${RESULT_DIR}/outs/mpi_2_1gpu_imipq500M_15560_32M_m_27GB_1Mqueries_search1_noprof.txt
