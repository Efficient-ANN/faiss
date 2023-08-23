module load boost/1.73_gnu cmake/3.17.3 gcc/10.2 cuda/11.8 gcc/7.4
module load /scratch/app/modulos/sequana/current openmpi/gnu/4.0.1_sequana

cd build
cmake --build .

cd faiss/gpu/demos
make demo_imipq_gpu_sift_m -j8

cd ../../../
make -j8

