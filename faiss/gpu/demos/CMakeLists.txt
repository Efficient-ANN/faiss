# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

find_package(CUDAToolkit REQUIRED)

add_executable(demo_imipq_cpu_gpu_sift EXCLUDE_FROM_ALL demo_imipq_cpu_gpu_sift.cpp)
target_link_libraries(demo_imipq_cpu_gpu_sift PRIVATE faiss CUDA::cudart)

add_executable(demo_imipq_cpu_gpu_sift_m EXCLUDE_FROM_ALL demo_imipq_cpu_gpu_sift_m.cpp)
target_link_libraries(demo_imipq_cpu_gpu_sift_m PRIVATE faiss CUDA::cudart)

add_executable(demo_imipq_cpu_sift EXCLUDE_FROM_ALL demo_imipq_cpu_sift.cpp)
target_link_libraries(demo_imipq_cpu_sift PRIVATE faiss CUDA::cudart)

add_executable(demo_imipq_gpu_sift EXCLUDE_FROM_ALL demo_imipq_gpu_sift.cpp)
target_link_libraries(demo_imipq_gpu_sift PRIVATE faiss CUDA::cudart)

add_executable(demo_imipq_gpu_sift_m EXCLUDE_FROM_ALL demo_imipq_gpu_sift_m.cpp)
target_link_libraries(demo_imipq_gpu_sift_m PRIVATE faiss CUDA::cudart)

add_executable(demo_imipq_gpu_sift_m_v2 EXCLUDE_FROM_ALL demo_imipq_gpu_sift_m.cpp)
target_link_libraries(demo_imipq_gpu_sift_m_v2 PRIVATE CUDA::cudart faiss CUDA::cudart)

add_executable(demo_ivfpq_gpu_sift EXCLUDE_FROM_ALL demo_ivfpq_gpu_sift.cpp)
target_link_libraries(demo_ivfpq_gpu_sift PRIVATE faiss CUDA::cudart)

add_executable(demo_ivfpq_gpu_sift_m EXCLUDE_FROM_ALL demo_ivfpq_gpu_sift_m.cpp)
target_link_libraries(demo_ivfpq_gpu_sift_m PRIVATE faiss CUDA::cudart)

add_executable(demo_ivfpq_cpu_gpu_sift EXCLUDE_FROM_ALL demo_ivfpq_cpu_gpu_sift.cpp)
target_link_libraries(demo_ivfpq_cpu_gpu_sift PRIVATE faiss CUDA::cudart)

add_executable(demo_imipq_gpu_sift_m_rand EXCLUDE_FROM_ALL demo_imipq_gpu_sift_m_rand.cpp)
target_link_libraries(demo_imipq_gpu_sift_m_rand PRIVATE faiss CUDA::cudart)

add_executable(demo_ivfpq_cpu_gpu_sift_m EXCLUDE_FROM_ALL demo_ivfpq_cpu_gpu_sift_m.cpp)
target_link_libraries(demo_ivfpq_cpu_gpu_sift_m PRIVATE faiss CUDA::cudart)

add_executable(demo_knn_comp EXCLUDE_FROM_ALL demo_knn_comp.cpp)
target_link_libraries(demo_knn_comp PRIVATE faiss CUDA::cudart)

add_executable(demo_knn_merge EXCLUDE_FROM_ALL demo_knn_merge.cpp)
target_link_libraries(demo_knn_merge PRIVATE faiss CUDA::cudart)

add_executable(demo_knn EXCLUDE_FROM_ALL demo_knn.cpp)
target_link_libraries(demo_knn PRIVATE faiss CUDA::cudart)

add_executable(demo_vector_residual EXCLUDE_FROM_ALL demo_vector_residual.cu)
target_link_libraries(demo_vector_residual PRIVATE faiss CUDA::cudart)

add_executable(demo_memory EXCLUDE_FROM_ALL demo_memory.cu)
target_link_libraries(demo_memory PRIVATE CUDA::cudart)

add_executable(demo_memory_v2 EXCLUDE_FROM_ALL demo_memory.cu)
target_link_libraries(demo_memory_v2 PRIVATE faiss CUDA::cudart)
