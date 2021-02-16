/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <faiss/MetricType.h>
#include <faiss/gpu/impl/Distance.cuh>
#include <faiss/gpu/impl/L2Norm.cuh>
#include <faiss/gpu/impl/MultiIndex2.cuh>
#include <faiss/gpu/impl/VectorResidual.cuh>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/MultiSequence.cuh>
#include <faiss/gpu/utils/Transpose.cuh>

namespace faiss {
namespace gpu {

MultiIndex2::MultiIndex2(GpuResources *res, int dim, MemorySpace space)
    : resources_(res), space_(space), numCodebooks_(2),
      dimPerCodebook_(dim / numCodebooks_), numCentroidsPerCodebook_(0),
      rawData_(res, AllocInfo(AllocType::FlatData, getCurrentDevice(), space,
                              res->getDefaultStreamCurrentDevice())) {
  FAISS_ASSERT(dim % numCodebooks_ == 0);
}

bool MultiIndex2::getUseFloat16() const { return false; }

int MultiIndex2::getSize() const {
  return numCentroidsPerCodebook_ * numCentroidsPerCodebook_;
}

int MultiIndex2::getCodebookSize() const { return numCentroidsPerCodebook_; }

int MultiIndex2::toMultiIndex(ushort2 indexPair) const {
  return indexPair.x + numCentroidsPerCodebook_ * indexPair.y;
}

int MultiIndex2::getDim() const { return numCodebooks_ * dimPerCodebook_; }

int MultiIndex2::getSubDim() const { return dimPerCodebook_; }

int MultiIndex2::getNumCodebooks() const { return numCodebooks_; }

void MultiIndex2::reserve(int numVecsTotal, cudaStream_t stream) {
  rawData_.reserve((unsigned)numVecsTotal * dimPerCodebook_ * sizeof(float),
                   stream);
}

Tensor<float, 2, true> &MultiIndex2::getVectorsFloat32Ref() { return vectors_; }

template <typename IndexT, typename IndexTVec2>
void MultiIndex2::queryImpl(Tensor<float, 2, true> &subQueries, int k,
                            Tensor<float, 2, true> &outDistances,
                            Tensor<IndexTVec2, 2, true> &outIndices,
                            bool exactDistance) {
  FAISS_ASSERT(subQueries.getSize(0) % numCodebooks_ == 0);
  FAISS_ASSERT(subQueries.getSize(1) == dimPerCodebook_);
  auto stream = resources_->getDefaultStreamCurrentDevice();

  int numSubQueries = subQueries.getSize(0);
  int numSubQueriesPerCodebook = numSubQueries / numCodebooks_;
  int subK = std::min(k, numCentroidsPerCodebook_);
  DeviceTensor<float, 3, true> outSubDistances(
      resources_, makeTempAlloc(AllocType::Other, stream),
      {numCodebooks_, numSubQueriesPerCodebook, subK});
  DeviceTensor<IndexT, 3, true> outSubIndices(
      resources_, makeTempAlloc(AllocType::Other, stream),
      {numCodebooks_, numSubQueriesPerCodebook, subK});

  for (int i = 0; i < numCodebooks_; i++) {
    auto subQueriesView = subQueries.narrowOutermost(
        i * numSubQueriesPerCodebook, numSubQueriesPerCodebook);
    auto vectorsView = vectors_.narrowOutermost(i * numCentroidsPerCodebook_,
                                                numCentroidsPerCodebook_);
    auto normsView = norms_.narrowOutermost(i * numCentroidsPerCodebook_,
                                            numCentroidsPerCodebook_);
    auto outSubDistancesView = outSubDistances[i].view();
    auto outSubIndicesView = outSubIndices[i].view();
    runL2Distance(resources_, vectorsView,
                  true, // vectors is row major
                  &normsView, subQueriesView,
                  true, // input is row major
                  subK, outSubDistancesView, outSubIndicesView, !exactDistance);
  }
  runMultiSequence2(k, outSubDistances, outSubIndices, outDistances, outIndices,
                    resources_);
}

template <typename IndexT>
void MultiIndex2::queryImpl(Tensor<float, 2, true> &subQueries, int k,
                            Tensor<float, 2, true> &outDistances,
                            Tensor<Index::idx_t, 2, true> &outIndices,
                            bool exactDistance) {
  FAISS_ASSERT(subQueries.getSize(0) % numCodebooks_ == 0);
  FAISS_ASSERT(subQueries.getSize(1) == dimPerCodebook_);
  auto stream = resources_->getDefaultStreamCurrentDevice();

  int numSubQueries = subQueries.getSize(0);
  int numSubQueriesPerCodebook = numSubQueries / numCodebooks_;
  int subK = std::min(k, numCentroidsPerCodebook_);
  DeviceTensor<float, 3, true> outSubDistances(
      resources_, makeTempAlloc(AllocType::Other, stream),
      {numCodebooks_, numSubQueriesPerCodebook, subK});
  DeviceTensor<IndexT, 3, true> outSubIndices(
      resources_, makeTempAlloc(AllocType::Other, stream),
      {numCodebooks_, numSubQueriesPerCodebook, subK});

  for (int i = 0; i < numCodebooks_; i++) {
    auto subQueriesView = subQueries.narrowOutermost(
        i * numSubQueriesPerCodebook, numSubQueriesPerCodebook);
    auto vectorsView = vectors_.narrowOutermost(i * numCentroidsPerCodebook_,
                                                numCentroidsPerCodebook_);
    auto normsView = norms_.narrowOutermost(i * numCentroidsPerCodebook_,
                                            numCentroidsPerCodebook_);
    auto outSubDistancesView = outSubDistances[i].view();
    auto outSubIndicesView = outSubIndices[i].view();
    runL2Distance(resources_, vectorsView,
                  true, // vectors is row major
                  &normsView, subQueriesView,
                  true, // input is row major
                  subK, outSubDistancesView, outSubIndicesView, !exactDistance);
  }
  runMultiSequence2(k, outSubDistances, outSubIndices, outDistances,
                    numCentroidsPerCodebook_, outIndices, resources_);
}

void MultiIndex2::query(Tensor<float, 2, true> &subQueries, int k,
                        Tensor<float, 2, true> &outDistances,
                        Tensor<int2, 2, true> &outIndices, bool exactDistance) {
  queryImpl<int, int2>(subQueries, k, outDistances, outIndices, exactDistance);
}

void MultiIndex2::query(Tensor<float, 2, true> &subQueries, int k,
                        Tensor<float, 2, true> &outDistances,
                        Tensor<ushort2, 2, true> &outIndices,
                        bool exactDistance) {
  queryImpl<unsigned short, ushort2>(subQueries, k, outDistances, outIndices,
                                     exactDistance);
}

void MultiIndex2::query(Tensor<float, 2, true> &subQueries, int k,
                        Tensor<float, 2, true> &outDistances,
                        Tensor<Index::idx_t, 2, true> &outIndices,
                        bool exactDistance) {
  queryImpl<unsigned short>(subQueries, k, outDistances, outIndices,
                            exactDistance);
}

void MultiIndex2::computeResidual(Tensor<float, 2, true> &vecs,
                                  Tensor<ushort2, 1, true> &listIds,
                                  Tensor<float, 2, true> &residuals) {
  runCalcResidual(vecs, vectors_, listIds, residuals,
                  resources_->getDefaultStreamCurrentDevice());
}

void MultiIndex2::computeResidual(Tensor<float, 2, true> &vecs,
                                  Tensor<int2, 1, true> &listIds,
                                  Tensor<float, 2, true> &residuals) {
  runCalcResidual(vecs, vectors_, listIds, residuals,
                  resources_->getDefaultStreamCurrentDevice());
}

void MultiIndex2::add(const float *data, int numVecsTotal,
                      cudaStream_t stream) {
  FAISS_ASSERT(numVecsTotal % numCodebooks_ == 0);
  FAISS_ASSERT(numCentroidsPerCodebook_ == 0);
  if (numVecsTotal == 0) {
    return;
  }

  rawData_.append((char *)data,
                  (unsigned)numVecsTotal * dimPerCodebook_ * sizeof(float),
                  stream, true /* reserve exactly */);

  numCentroidsPerCodebook_ += numVecsTotal / numCodebooks_;

  DeviceTensor<float, 2, true> vectors((float *)rawData_.data(),
                                       {numVecsTotal, dimPerCodebook_});
  vectors_ = std::move(vectors);

  DeviceTensor<float, 1, true> norms(
      resources_, makeSpaceAlloc(AllocType::FlatData, space_, stream),
      {numVecsTotal});
  runL2Norm(vectors_, true, norms, true, stream);
  norms_ = std::move(norms);
}

void MultiIndex2::reset() {
  rawData_.clear();
  vectors_ = DeviceTensor<float, 2, true>();
  norms_ = DeviceTensor<float, 1, true>();
  numCentroidsPerCodebook_ = 0;
}

} // namespace gpu
} // namespace faiss
