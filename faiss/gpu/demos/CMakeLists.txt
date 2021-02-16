# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

find_package(CUDAToolkit REQUIRED)

add_executable(demo_imipq_cpu_sift EXCLUDE_FROM_ALL demo_imipq_cpu_sift.cpp)
target_link_libraries(demo_imipq_cpu_sift PRIVATE faiss CUDA::cudart)

add_executable(demo_imipq_gpu_sift EXCLUDE_FROM_ALL demo_imipq_gpu_sift.cpp)
target_link_libraries(demo_imipq_gpu_sift PRIVATE faiss CUDA::cudart)

add_executable(demo_ivfpq_gpu_sift EXCLUDE_FROM_ALL demo_ivfpq_gpu_sift.cpp)
target_link_libraries(demo_ivfpq_gpu_sift PRIVATE faiss CUDA::cudart)

add_executable(demo_knn_merge EXCLUDE_FROM_ALL demo_knn_merge.cpp)
target_link_libraries(demo_knn_merge PRIVATE faiss CUDA::cudart)

add_executable(demo_knn EXCLUDE_FROM_ALL demo_knn.cpp)
target_link_libraries(demo_knn PRIVATE faiss CUDA::cudart)

add_executable(demo_vecs_read EXCLUDE_FROM_ALL demo_vecs_read.cpp)
target_link_libraries(demo_vecs_read PRIVATE faiss)

add_executable(demo_vecs_write EXCLUDE_FROM_ALL demo_vecs_write.cpp)
target_link_libraries(demo_vecs_write PRIVATE faiss)

add_executable(demo_vector_residual EXCLUDE_FROM_ALL demo_vector_residual.cu)
target_link_libraries(demo_vector_residual PRIVATE faiss CUDA::cudart)
