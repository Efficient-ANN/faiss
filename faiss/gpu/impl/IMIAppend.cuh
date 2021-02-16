/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <faiss/gpu/utils/Tensor.cuh>
#include <thrust/device_vector.h>

namespace faiss {
namespace gpu {

/// Append user indices to IMI lists
void runIMIIndicesAppend(int codebookSize, Tensor<ushort2, 1, true> &listIds,
                         Tensor<int, 1, true> &listOffset,
                         Tensor<Index::idx_t, 1, true> &indices,
                         IndicesOptions opt,
                         thrust::device_vector<void *> &listIndices,
                         cudaStream_t stream);

/// Append PQ codes to IMI lists (non-interleaved format)
void runIMIPQAppend(int codebookSize, Tensor<ushort2, 1, true> &listIds,
                    Tensor<int, 1, true> &listOffset,
                    Tensor<uint8_t, 2, true> &encodings,
                    thrust::device_vector<void *> &listCodes,
                    cudaStream_t stream);

/// Append PQ codes to IMI lists (interleaved format)
void runIMIPQInterleavedAppend(int codebookSize,
                               Tensor<int, 1, true> &uniqueLists,
                               Tensor<int, 1, true> &vectorsByUniqueList,
                               Tensor<int, 1, true> &uniqueListVectorStart,
                               Tensor<int, 1, true> &uniqueListStartOffset,
                               int bitsPerCode,
                               Tensor<uint8_t, 2, true> &encodings,
                               thrust::device_vector<void *> &listCodes,
                               cudaStream_t stream);

} // namespace gpu
} // namespace faiss
