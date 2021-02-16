/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/impl/VectorResidual.cuh>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/Tensor.cuh>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/gpu/utils/Limits.cuh>
#include <math_constants.h> // in CUDA SDK, for CUDART_NAN_F

namespace faiss { namespace gpu {

template <typename CentroidT, bool LargeDim>
__global__ void calcResidual(Tensor<float, 2, true> vecs,
                             Tensor<CentroidT, 2, true> centroids,
                             Tensor<int, 1, true> vecToCentroid,
                             Tensor<float, 2, true> residuals) {
  auto vec = vecs[blockIdx.x];
  auto residual = residuals[blockIdx.x];

  int centroidId = vecToCentroid[blockIdx.x];
  // Vector could be invalid (containing NaNs), so -1 was the
  // classified centroid
  if (centroidId == -1) {
    if (LargeDim) {
      for (int i = threadIdx.x; i < vecs.getSize(1); i += blockDim.x) {
        residual[i] = CUDART_NAN_F;
      }
    } else {
      residual[threadIdx.x] = CUDART_NAN_F;
    }

    return;
  }

  auto centroid = centroids[centroidId];

  if (LargeDim) {
    for (int i = threadIdx.x; i < vecs.getSize(1); i += blockDim.x) {
      residual[i] = vec[i] - ConvertTo<float>::to(centroid[i]);
    }
  } else {
    residual[threadIdx.x] = vec[threadIdx.x] -
      ConvertTo<float>::to(centroid[threadIdx.x]);
  }
}

template <typename CentroidT, typename IndexT, typename IndexTVec2,
          bool LargeDim>
__global__ void
calcResidualMultiIndex2(Tensor<float, 2, true> vecs,
                        Tensor<CentroidT, 2, true> centroids,
                        Tensor<IndexTVec2, 1, true> vecToCentroid,
                        Tensor<float, 2, true> residuals) {
  bool firstVecSubVec = blockIdx.x < residuals.getSize(0);
  auto subVec = vecs[blockIdx.x];
  IndexT centroidId;
  float *residual;
  if (firstVecSubVec) {
    IndexTVec2 centroidId2 = vecToCentroid[blockIdx.x];
    centroidId = centroidId2.x;
    residual = residuals[blockIdx.x].data();
  } else {
    int residualIdx = blockIdx.x - residuals.getSize(0);
    IndexTVec2 centroidId2 = vecToCentroid[residualIdx];
    centroidId = centroids.getSize(0) / 2 + centroidId2.y;
    residual = residuals[residualIdx].data() + vecs.getSize(1);
  }

  // Vector could be invalid (containing NaNs), so Limits<IndexT>::getMax() was
  // the classified centroid
  if (centroidId == Limits<IndexT>::getMax()) {
    if (LargeDim) {
      for (int i = threadIdx.x; i < vecs.getSize(1); i += blockDim.x) {
        residual[i] = CUDART_NAN_F;
      }
    } else {
      residual[threadIdx.x] = CUDART_NAN_F;
    }
    return;
  }

  auto centroid = centroids[centroidId];
  if (LargeDim) {
    for (int i = threadIdx.x; i < vecs.getSize(1); i += blockDim.x) {
      residual[i] = subVec[i] - ConvertTo<float>::to(centroid[i]);
    }
  } else {
    residual[threadIdx.x] =
        subVec[threadIdx.x] - ConvertTo<float>::to(centroid[threadIdx.x]);
  }
}

template <typename T>
__global__ void gatherReconstruct(Tensor<int, 1, true> listIds,
                                  Tensor<T, 2, true> vecs,
                                  Tensor<float, 2, true> out) {
  auto id = listIds[blockIdx.x];
  auto vec = vecs[id];
  auto outVec = out[blockIdx.x];

  Convert<T, float> conv;

  for (int i = threadIdx.x; i < vecs.getSize(1); i += blockDim.x) {
    outVec[i] = id == -1 ? 0.0f : conv(vec[i]);
  }
}

template <typename CentroidT>
void calcResidual(Tensor<float, 2, true>& vecs,
                  Tensor<CentroidT, 2, true>& centroids,
                  Tensor<int, 1, true>& vecToCentroid,
                  Tensor<float, 2, true>& residuals,
                  cudaStream_t stream) {
  FAISS_ASSERT(vecs.getSize(1) == centroids.getSize(1));
  FAISS_ASSERT(vecs.getSize(1) == residuals.getSize(1));
  FAISS_ASSERT(vecs.getSize(0) == vecToCentroid.getSize(0));
  FAISS_ASSERT(vecs.getSize(0) == residuals.getSize(0));

  dim3 grid(vecs.getSize(0));

  int maxThreads = getMaxThreadsCurrentDevice();
  bool largeDim = vecs.getSize(1) > maxThreads;
  dim3 block(std::min(vecs.getSize(1), maxThreads));

  if (largeDim) {
    calcResidual<CentroidT, true><<<grid, block, 0, stream>>>(
      vecs, centroids, vecToCentroid, residuals);
  } else {
    calcResidual<CentroidT, false><<<grid, block, 0, stream>>>(
      vecs, centroids, vecToCentroid, residuals);
  }

  CUDA_TEST_ERROR();
}

template <typename CentroidT, typename IndexT, typename IndexTVec2>
void calcResidualMultiIndex2(Tensor<float, 2, true> &vecs,
                             Tensor<CentroidT, 2, true> &centroids,
                             Tensor<IndexTVec2, 1, true> &vecToCentroid,
                             Tensor<float, 2, true> &residuals,
                             cudaStream_t stream) {
  FAISS_ASSERT(vecs.getSize(1) == centroids.getSize(1));
  FAISS_ASSERT(vecs.getSize(1) * 2 == residuals.getSize(1));
  FAISS_ASSERT(vecs.getSize(0) % 2 == 0);
  FAISS_ASSERT(vecs.getSize(0) == vecToCentroid.getSize(0) * 2);
  FAISS_ASSERT(vecs.getSize(0) == residuals.getSize(0) * 2);

  dim3 grid(vecs.getSize(0));

  int maxThreads = getMaxThreadsCurrentDevice();
  bool largeDim = vecs.getSize(1) > maxThreads;
  dim3 block(std::min(vecs.getSize(1), maxThreads));

  if (largeDim) {
    calcResidualMultiIndex2<CentroidT, IndexT, IndexTVec2, true>
        <<<grid, block, 0, stream>>>(vecs, centroids, vecToCentroid, residuals);
  } else {
    calcResidualMultiIndex2<CentroidT, IndexT, IndexTVec2, false>
        <<<grid, block, 0, stream>>>(vecs, centroids, vecToCentroid, residuals);
  }

  CUDA_TEST_ERROR();
}

template <typename T>
void gatherReconstruct(Tensor<int, 1, true>& listIds,
                       Tensor<T, 2, true>& vecs,
                       Tensor<float, 2, true>& out,
                       cudaStream_t stream) {
  FAISS_ASSERT(listIds.getSize(0) == out.getSize(0));
  FAISS_ASSERT(vecs.getSize(1) == out.getSize(1));

  dim3 grid(listIds.getSize(0));

  int maxThreads = getMaxThreadsCurrentDevice();
  dim3 block(std::min(vecs.getSize(1), maxThreads));

  gatherReconstruct<T><<<grid, block, 0, stream>>>(listIds, vecs, out);

  CUDA_TEST_ERROR();
}

void runCalcResidual(Tensor<float, 2, true>& vecs,
                     Tensor<float, 2, true>& centroids,
                     Tensor<int, 1, true>& vecToCentroid,
                     Tensor<float, 2, true>& residuals,
                     cudaStream_t stream) {
  calcResidual<float>(vecs, centroids, vecToCentroid, residuals, stream);
}

void runCalcResidual(Tensor<float, 2, true>& vecs,
                     Tensor<half, 2, true>& centroids,
                     Tensor<int, 1, true>& vecToCentroid,
                     Tensor<float, 2, true>& residuals,
                     cudaStream_t stream) {
  calcResidual<half>(vecs, centroids, vecToCentroid, residuals, stream);
}

void runCalcResidual(Tensor<float, 2, true> &vecs,
                     Tensor<float, 2, true> &centroids,
                     Tensor<ushort2, 1, true> &vecToCentroid,
                     Tensor<float, 2, true> &residuals, cudaStream_t stream) {
  calcResidualMultiIndex2<float, unsigned short, ushort2>(
      vecs, centroids, vecToCentroid, residuals, stream);
}

void runCalcResidual(Tensor<float, 2, true> &vecs,
                     Tensor<half, 2, true> &centroids,
                     Tensor<ushort2, 1, true> &vecToCentroid,
                     Tensor<float, 2, true> &residuals, cudaStream_t stream) {
  calcResidualMultiIndex2<half, unsigned short, ushort2>(
      vecs, centroids, vecToCentroid, residuals, stream);
}

void runCalcResidual(Tensor<float, 2, true> &vecs,
                     Tensor<float, 2, true> &centroids,
                     Tensor<int2, 1, true> &vecToCentroid,
                     Tensor<float, 2, true> &residuals, cudaStream_t stream) {
  calcResidualMultiIndex2<float, int, int2>(vecs, centroids, vecToCentroid,
                                            residuals, stream);
}

void runCalcResidual(Tensor<float, 2, true> &vecs,
                     Tensor<half, 2, true> &centroids,
                     Tensor<int2, 1, true> &vecToCentroid,
                     Tensor<float, 2, true> &residuals, cudaStream_t stream) {
  calcResidualMultiIndex2<half, int, int2>(vecs, centroids, vecToCentroid,
                                           residuals, stream);
}

void runReconstruct(Tensor<int, 1, true>& listIds,
                    Tensor<float, 2, true>& vecs,
                    Tensor<float, 2, true>& out,
                    cudaStream_t stream) {
  gatherReconstruct<float>(listIds, vecs, out, stream);
}

void runReconstruct(Tensor<int, 1, true>& listIds,
                    Tensor<half, 2, true>& vecs,
                    Tensor<float, 2, true>& out,
                    cudaStream_t stream) {
  gatherReconstruct<half>(listIds, vecs, out, stream);
}

} } // namespace
