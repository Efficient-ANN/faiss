/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/GpuIndex.h>
#include <memory>
#include <utility>
#include <vector>

namespace faiss {
namespace gpu {

struct MultiIndex2;

struct GpuMultiIndex2Config : GpuIndexConfig {
  inline GpuMultiIndex2Config() {}
};

class GpuMultiIndex2 : public GpuIndex {
public:
  GpuMultiIndex2(GpuResourcesProvider *provider, int dims,
                 int numCentroidsPerCodebook,
                 GpuMultiIndex2Config config = GpuMultiIndex2Config());

  GpuMultiIndex2(std::shared_ptr<GpuResources> resources, int dims,
                 int numCentroidsPerCodebook,
                 GpuMultiIndex2Config config = GpuMultiIndex2Config());

  ~GpuMultiIndex2() override;

  static size_t calcMemorySpaceSize(int numVecsTotal, int dimPerCodebook,
                                    bool useFloat16);

  int toMultiIndex(std::pair<ushort, ushort> indexPair) const;

  int getCodebookSize();

  int getNumCodebooks();

  /// Returns the number of vectors we address
  int getNumVecs();

  int getSubDim();

  /// Returns centrois (numCodebooks * codebookSize * subDim) for debugging
  /// purpose
  std::vector<float> getCentroids();

  // centroids (numCodebooks * codebookSize, subDim)
  void load(int codebookSize, const float *centroids);

  /// Clears all vectors from this index
  void reset() override;

  void train(Index::idx_t n, const float *x) override;

  void add(faiss::Index::idx_t, const float *x) override;

  void add_with_ids(Index::idx_t n, const float *x,
                    const Index::idx_t *ids) override;

  /// `x` and `labels` can be resident on the CPU or any GPU; copies are
  /// performed as needed
  void assign(Index::idx_t n, const float *x, Index::idx_t *labels,
              Index::idx_t k = 1) const override;

  void assign_pair(Index::idx_t n, const float *x,
                   std::pair<ushort, ushort> *labels, Index::idx_t k = 1) const;

  void search(Index::idx_t n, const float *x, Index::idx_t k, float *distances,
              Index::idx_t *labels) const override;

  void search_pair(Index::idx_t n, const float *x, Index::idx_t k,
                   float *distances, std::pair<ushort, ushort> *labels) const;

  void compute_residual_pair(const float *xs, float *residuals,
                             std::pair<ushort, ushort> key) const;

  void compute_residual_n_pair(faiss::Index::idx_t n, const float *xs,
                               float *residuals,
                               const std::pair<ushort, ushort> *keys) const;

  void compute_nearest_residual_n(faiss::Index::idx_t n, const float *xs,
                                  float *residuals) const;

  /// For internal access
  inline MultiIndex2 *getGpuData() { return data_.get(); }

  static const int NUM_CODEBOOKS;

protected:
  bool addImplRequiresIDs_() const override;

  void addImpl_(int n, const float *x, const Index::idx_t *ids) override;

  void searchImpl_(int n, const float *x, int k, float *distances,
                   Index::idx_t *labels) const override;

  void searchPairImpl_(int n, const float *x, int k, float *distances,
                       std::pair<ushort, ushort> *labels) const;

  int numVecsPerCodebook_, subDim_;

  /// Our configuration options
  const GpuMultiIndex2Config config_;

  /// Holds our GPU data containing the list of vectors
  std::unique_ptr<MultiIndex2> data_;

private:
  /// Calls searchImpl_ for a single page of GPU-resident data
  void searchNonPaged_(int n, const float *x, int k, float *outDistancesData,
                       Index::idx_t *outIndicesData) const;

  /// Calls searchImpl_ for a single page of GPU-resident data,
  /// handling paging of the data and copies from the CPU
  void searchFromCpuPaged_(int n, const float *x, int k,
                           float *outDistancesData,
                           Index::idx_t *outIndicesData) const;

  /// Calls searchImpl_ for a single page of GPU-resident data
  void searchNonPaged_(int n, const float *x, int k, float *outDistancesData,
                       std::pair<ushort, ushort> *outIndicesData) const;

  /// Calls searchImpl_ for a single page of GPU-resident data,
  /// handling paging of the data and copies from the CPU
  void searchFromCpuPaged_(int n, const float *x, int k,
                           float *outDistancesData,
                           std::pair<ushort, ushort> *outIndicesData) const;
};

} // namespace gpu
} // namespace faiss
