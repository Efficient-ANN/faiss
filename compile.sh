module load boost/1.73_gnu cmake/3.17.3 gcc/10.2 cuda/11.4 gcc/7.4
module load /scratch/app/modulos/sequana/current openmpi/gnu/4.0.1_sequana

cd build
cmake -B build . -DFAISS_ENABLE_PYTHON=OFF -DCMAKE_CUDA_ARCHITECTURES="35;50;52;60;61;70;75"
make -C build -j8

cd faiss/gpu/demos
make demo_imipq_gpu_sift_m -j8
make demo_memory -j8
make demo_vector_residual -j8
make demo_imipq_gpu_sift_m_v2 -j8
make demo_memory_v2 -j8


