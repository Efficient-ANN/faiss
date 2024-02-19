#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p ict_gpu       #Fila (partition) a ser utilizada
#SBATCH --time=6:00:00
#SBATCH --account=petrobrasiageo
#SBATCH --exclusive         #Utilização exclusiva dos nós
#SBATCH --job-name="a1test"

module load boost/1.73_gnu cmake/3.17.3 gcc/10.2 gcc/7.4 
#module nvhpc/22.3
module load cuda/11.0
module load /scratch/app/modulos/sequana/current openmpi/gnu/4.0.1_sequana
set -x

FAISS_DIR=/petrobr/parceirosbr/petrobrasiageo/willian.barreiros/git/a/faiss_imipq
PROF_SCRIPT_DIR=${FAISS_DIR}/faiss/gpu/demos/scripts/mpi

${PROF_SCRIPT_DIR}/gist/gist.sh
${PROF_SCRIPT_DIR}/sift/sift.sh
