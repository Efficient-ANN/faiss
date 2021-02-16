/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
namespace gpu {

void runMultiSequence2(const int k, Tensor<float, 3, true> &inDistances,
                       Tensor<ushort, 3, true> &inIndices,
                       Tensor<float, 2, true> &outDistances,
                       Tensor<ushort2, 2, true> &outIndices, GpuResources *res);

void runMultiSequence2(const int k, Tensor<float, 3, true> &inDistances,
                       Tensor<int, 3, true> &inIndices,
                       Tensor<float, 2, true> &outDistances,
                       Tensor<int2, 2, true> &outIndices, GpuResources *res);

void runMultiSequence2(const int k, Tensor<float, 3, true> &inDistances,
                       Tensor<ushort, 3, true> &inIndices,
                       Tensor<float, 2, true> &outDistances, int codebookSize,
                       Tensor<Index::idx_t, 2, true> &outIndices,
                       GpuResources *res);

} // namespace gpu
} // namespace faiss
