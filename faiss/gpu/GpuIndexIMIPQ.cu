/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cstring>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIMIPQ.h>
#include <faiss/gpu/impl/IMIPQ.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/utils/utils.h>
#include <utility>

namespace faiss {
namespace gpu {

GpuIndexIMIPQ::GpuIndexIMIPQ(GpuResourcesProvider *provider, int dims,
                             int coarseCodebookSize, int subQuantizers,
                             int bitsPerCode, GpuIndexIMIPQConfig config)
    : GpuIndexIMI(provider, dims, coarseCodebookSize, config),
      pq(dims, subQuantizers, bitsPerCode), imipqConfig_(config),
      usePrecomputedTables_(config.usePrecomputedTables),
      subQuantizers_(subQuantizers), bitsPerCode_(bitsPerCode),
      reserveMemoryVecs_(0), expectedNumAddsPerList(nullptr), index_(nullptr) {
  verifySettings_();
  // We haven't trained ourselves, so don't construct the PQ index yet
  this->is_trained = false;
}

GpuIndexIMIPQ::GpuIndexIMIPQ(std::shared_ptr<GpuResources> resources, int dims,
                             int coarseCodebookSize, int subQuantizers,
                             int bitsPerCode, GpuIndexIMIPQConfig config)
    : GpuIndexIMI(resources, dims, coarseCodebookSize, config),
      pq(dims, subQuantizers, bitsPerCode), imipqConfig_(config),
      usePrecomputedTables_(config.usePrecomputedTables),
      subQuantizers_(subQuantizers), bitsPerCode_(bitsPerCode),
      reserveMemoryVecs_(0), expectedNumAddsPerList(nullptr), index_(nullptr) {
  verifySettings_();
  // We haven't trained ourselves, so don't construct the PQ index yet
  this->is_trained = false;
}

GpuIndexIMIPQ::~GpuIndexIMIPQ() {}

size_t GpuIndexIMIPQ::calcInvListsMemorySpaceSize(int numVecs,
                                                  int numSubQuantizers,
                                                  int bitsPerSubQuantizer,
                                                  bool interleavedLayout,
                                                  IndicesOptions options) {
  return IMIPQ::calcMemorySpaceSize(numVecs, numSubQuantizers,
                                    bitsPerSubQuantizer, interleavedLayout,
                                    options);
}

size_t GpuIndexIMIPQ::calcMemorySpaceSize(int numTotalVecsCoarseQuantizer,
                                          int dimPerCodebook, bool useFloat16,
                                          int numVecs, int numSubQuantizers,
                                          int bitsPerSubQuantizer,
                                          bool interleavedLayout,
                                          IndicesOptions options) {
  return GpuIndexIMI::calcMemorySpaceSizeCoarseQuantizer(
             numTotalVecsCoarseQuantizer, dimPerCodebook, useFloat16) +
         GpuIndexIMIPQ::calcInvListsMemorySpaceSize(numVecs, numSubQuantizers,
                                                    bitsPerSubQuantizer,
                                                    interleavedLayout, options);
}

void GpuIndexIMIPQ::updateExpectedNumAddsPerList(Index::idx_t n,
                                                 const float *x) {
  if (!expectedNumAddsPerList) {
    expectedNumAddsPerList.reset(new std::unordered_map<int, int>());
  }

  std::vector<faiss::Index::idx_t> outLabels(n);

  quantizer->assign(n, x, outLabels.data());

  std::unordered_map<int, int>::iterator entry;
  for (auto &label : outLabels) {
    entry = expectedNumAddsPerList->find((int)label);
    if (entry == expectedNumAddsPerList->end()) {
      expectedNumAddsPerList->operator[]((int)label) = 1;
    } else {
      expectedNumAddsPerList->operator[]((int)label)++;
    }
  }
}

void GpuIndexIMIPQ::applyExpectedNumAddsPerList() {
  if (expectedNumAddsPerList && index_) {
    size_t numExpectedVecs = 0;
    for (auto &expectedNumAdds : *expectedNumAddsPerList) {
      numExpectedVecs += expectedNumAdds.second;
    }
    index_->reserveMemory(expectedNumAddsPerList.get());
  }
}

void GpuIndexIMIPQ::resetExpectedNumAddsPerList() {
  expectedNumAddsPerList.reset(nullptr);
}

void GpuIndexIMIPQ::reserveMemory(size_t numVecs) {
  reserveMemoryVecs_ = numVecs;
  if (index_) {
    DeviceScope scope(config_.device);
    index_->reserveMemory(numVecs);
  }
}

void GpuIndexIMIPQ::setPrecomputedCodes(bool enable) {
  imipqConfig_.usePrecomputedTables = enable;
  if (index_) {
    DeviceScope scope(config_.device);
    index_->setPrecomputedCodes(enable);
  }

  verifySettings_();
}

bool GpuIndexIMIPQ::getPrecomputedCodes() const {
  return usePrecomputedTables_;
}

int GpuIndexIMIPQ::getMaxListLength() const {
  return index_->getMaxListLength();
}

int GpuIndexIMIPQ::getNumSubQuantizers() const { return subQuantizers_; }

int GpuIndexIMIPQ::getBitsPerCode() const { return bitsPerCode_; }

int GpuIndexIMIPQ::getCentroidsPerSubQuantizer() const {
  return utils::pow2(bitsPerCode_);
}

size_t GpuIndexIMIPQ::reclaimMemory() {
  if (index_) {
    DeviceScope scope(config_.device);
    return index_->reclaimMemory();
  }

  return 0;
}

void GpuIndexIMIPQ::reset() {
  if (index_) {
    DeviceScope scope(config_.device);

    index_->reset();
    this->ntotal = 0;
  } else {
    FAISS_ASSERT(this->ntotal == 0);
  }
  resetExpectedNumAddsPerList();
}

void GpuIndexIMIPQ::trainResidualQuantizer_(Index::idx_t n, const float *x) {
  // Code largely copied from faiss::IndexIVFPQ
  auto x_in = x;

  x = fvecs_maybe_subsample(d, (size_t *)&n,
                            pq.cp.max_points_per_centroid * pq.ksub, x, verbose,
                            pq.cp.seed);

  ScopeDeleter<float> del_x(x_in == x ? nullptr : x);

  if (this->verbose) {
    printf("computing residuals\n");
  }

  std::vector<float> residuals(n * d);
  quantizer->compute_nearest_residual_n(n, x, residuals.data());

  if (this->verbose) {
    printf("training %d x %d product quantizer on %ld vectors in %dD\n",
           subQuantizers_, getCentroidsPerSubQuantizer(), n, this->d);
  }

  // For PQ training purposes, accelerate it by using a GPU clustering index if
  // a clustering index has not already been assigned
  if (!pq.assign_index) {
    try {
      GpuIndexFlatConfig config;
      config.device = imipqConfig_.device;
      GpuIndexFlatL2 pqIndex(resources_, pq.dsub, config);

      pq.assign_index = &pqIndex;
      pq.train(n, residuals.data());
    } catch (...) {
      pq.assign_index = nullptr;
      throw;
    }

    pq.assign_index = nullptr;
  } else {
    // use the currently assigned clustering index
    pq.train(n, residuals.data());
  }

  index_.reset(new IMIPQ(
      resources_.get(), quantizer->getGpuData(), subQuantizers_, bitsPerCode_,
      imipqConfig_.useMMCodeDistance, imipqConfig_.interleavedLayout,
      pq.centroids.data(), imipqConfig_.indicesOptions, config_.memorySpace));

  FAISS_ASSERT(this->nlist == index_->getNumLists());

  if (reserveMemoryVecs_) {
    index_->reserveMemory(reserveMemoryVecs_);
  }

  index_->setPrecomputedCodes(imipqConfig_.usePrecomputedTables);
}

void GpuIndexIMIPQ::train(Index::idx_t n, const float *x) {
  DeviceScope scope(config_.device);

  if (this->is_trained) {
    FAISS_ASSERT(quantizer->is_trained);
    FAISS_ASSERT(quantizer->ntotal == nlist);
    FAISS_ASSERT(index_);
    return;
  }

  FAISS_ASSERT(!index_);

  // FIXME: GPUize more of this
  // First, make sure that the data is resident on the CPU, if it is not on the
  // CPU, as we depend upon parts of the CPU code
  auto hostData =
      toHost<float, 2>((float *)x, resources_->getDefaultStream(config_.device),
                       {(int)n, (int)this->d});

  trainQuantizer_(n, hostData.data());
  trainResidualQuantizer_(n, hostData.data());

  FAISS_ASSERT(index_);

  this->is_trained = true;
}

void GpuIndexIMIPQ::addImpl_(int n, const float *x, const Index::idx_t *xids) {
  // Device is already set in GpuIndex::add
  FAISS_ASSERT(index_);
  FAISS_ASSERT(n > 0);

  // Data is already resident on the GPU
  Tensor<float, 2, true> data(const_cast<float *>(x),
                              {this->quantizer->getNumCodebooks() * (int)n,
                               this->quantizer->getSubDim()});
  Tensor<Index::idx_t, 1, true> labels(const_cast<Index::idx_t *>(xids), {n});

  // Not all vectors may be able to be added (some may contain NaNs etc)
  index_->addVectors(data, labels);

  // but keep the ntotal based on the total number of vectors that we attempted
  // to add
  ntotal += n;
}

void GpuIndexIMIPQ::searchImpl_(int n, const float *x, int k, float *distances,
                                Index::idx_t *labels) const {
  // Device is already set in GpuIndex::search
  FAISS_ASSERT(index_);
  FAISS_ASSERT(n > 0);

  // Data is already resident on the GPU
  Tensor<float, 2, true> queries(const_cast<float *>(x),
                                 {this->quantizer->getNumCodebooks() * (int)n,
                                  this->quantizer->getSubDim()});
  Tensor<float, 2, true> outDistances(distances, {n, k});
  Tensor<Index::idx_t, 2, true> outLabels(const_cast<Index::idx_t *>(labels),
                                          {n, k});

  index_->query(queries, nprobe, k, outDistances, outLabels);
}

int GpuIndexIMIPQ::getListLength(int listId) const {
  FAISS_ASSERT(index_);
  DeviceScope scope(config_.device);

  return index_->getListLength(listId);
}

int GpuIndexIMIPQ::getAllListsLength() const {
  FAISS_ASSERT(index_);
  DeviceScope scope(config_.device);

  return index_->getAllListsLength();
}

std::vector<uint8_t> GpuIndexIMIPQ::getListVectorData(int listId,
                                                      bool gpuFormat) const {
  FAISS_ASSERT(index_);
  DeviceScope scope(config_.device);

  return index_->getListVectorData(listId, gpuFormat);
}

std::vector<Index::idx_t> GpuIndexIMIPQ::getListIndices(int listId) const {
  FAISS_ASSERT(index_);
  DeviceScope scope(config_.device);

  return index_->getListIndices(listId);
}

std::vector<float> GpuIndexIMIPQ::getPQCentroids() const {
  DeviceScope scope(config_.device);
  auto subCentroidsDevice = index_->getPQCentroids();
  std::vector<float> subCentroids(subCentroidsDevice.numElements());
  fromDevice<float, 3>(subCentroidsDevice, subCentroids.data(),
                       resources_->getDefaultStream(config_.device));
  return subCentroids;
}

std::vector<float> GpuIndexIMIPQ::getPrecomputedCodesVec() const {
  DeviceScope scope(config_.device);
  auto precomputedCodesDevice = index_->getPrecomputedCodesVecFloat32();
  std::vector<float> precomputedCodes(precomputedCodesDevice.numElements());
  fromDevice<float, 3>(precomputedCodesDevice, precomputedCodes.data(),
                       resources_->getDefaultStream(config_.device));
  return precomputedCodes;
}

std::vector<float> GpuIndexIMIPQ::calcTerm3(int n, const float *x) {
  DeviceScope scope(config_.device);
  auto stream = resources_->getDefaultStream(config_.device);

  float *subQueries = new float[n * this->d];
  ScopeDeleter<float> delSubQueries(subQueries);
  fvec_split(subQueries, this->quantizer->getNumCodebooks(), x, (size_t)n,
             this->quantizer->getSubDim());

  // Make sure arguments are on the device we desire; use temporary
  // memory allocations to move it if necessary
  auto vecs = toDeviceTemporary<float, 2>(
      resources_.get(), imipqConfig_.device, const_cast<float *>(subQueries),
      stream,
      {this->quantizer->getNumCodebooks() * n, this->quantizer->getSubDim()});

  CudaEvent copyEnd(resources_->getDefaultStream(config_.device));

  int numSubQuantizerCodes = utils::pow2(bitsPerCode_);
  DeviceTensor<float, 3, true> term3Device(
      resources_.get(), makeTempAlloc(AllocType::Other, stream),
      {n, subQuantizers_, numSubQuantizerCodes});

  index_->runCalcTerm3(vecs, term3Device);

  copyEnd.cpuWaitOnEvent();

  std::vector<float> term3(n * subQuantizers_ * numSubQuantizerCodes);
  fromDevice<float, 3>(term3Device, term3.data(), stream);
  return term3;
}

void GpuIndexIMIPQ::verifySettings_() const {
  // Our implementation has these restrictions:

  // Must have some number of lists
  FAISS_THROW_IF_NOT_MSG(nlist > 0, "nlist must be >0");

  // up to a single byte per code
  if (imipqConfig_.interleavedLayout) {
    FAISS_THROW_IF_NOT_FMT(
        bitsPerCode_ == 4 || bitsPerCode_ == 5 || bitsPerCode_ == 6 ||
            bitsPerCode_ == 8,
        "Bits per code must be between 4, 5, 6 or 8 (passed %d)", bitsPerCode_);

  } else {
    FAISS_THROW_IF_NOT_FMT(bitsPerCode_ == 8,
                           "Bits per code must be 8 (passed %d)", bitsPerCode_);
  }

  // Sub-quantizers must evenly divide dimensions available
  FAISS_THROW_IF_NOT_FMT(this->d % subQuantizers_ == 0,
                         "Number of sub-quantizers (%d) must be an "
                         "even divisor of the number of dimensions (%d)",
                         subQuantizers_, this->d);

  // The number of bytes per encoded vector must be one we support
  FAISS_THROW_IF_NOT_FMT(
      imipqConfig_.interleavedLayout ||
          IMIPQ::isSupportedPQCodeLength(subQuantizers_),
      "Number of bytes per encoded vector / sub-quantizers (%d) "
      "is not supported",
      subQuantizers_);

  // We must have enough shared memory on the current device to store
  // our lookup distances
  int lookupTableSize = sizeof(float);
  if (imipqConfig_.useFloat16LookupTables) {
    lookupTableSize = sizeof(half);
  }

  // 64 bytes per code is only supported with usage of float16, at 2^8
  // codes per subquantizer
  size_t requiredSmemSize =
      lookupTableSize * subQuantizers_ * utils::pow2(bitsPerCode_);
  size_t smemPerBlock = getMaxSharedMemPerBlock(config_.device);

  FAISS_THROW_IF_NOT_FMT(requiredSmemSize <=
                             getMaxSharedMemPerBlock(config_.device),
                         "Device %d has %zu bytes of shared memory, while "
                         "%d bits per code and %d sub-quantizers requires %zu "
                         "bytes. Consider useFloat16LookupTables and/or "
                         "reduce parameters",
                         config_.device, smemPerBlock, bitsPerCode_,
                         subQuantizers_, requiredSmemSize);
}

} // namespace gpu
} // namespace faiss
