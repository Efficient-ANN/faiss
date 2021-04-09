/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/gpu/GpuIndexIMI.h>
#include <faiss/gpu/GpuIndexIMIPQ.h>
#include <faiss/impl/ProductQuantizer.h>
#include <memory>
#include <unordered_map>
#include <vector>

namespace faiss {
struct IndexIVFPQ;
}

namespace faiss {
namespace gpu {

class IMIPQv2;

class GpuIndexIMIPQv2 : public GpuIndexIMI {
public:
  /// Construct an empty index
  GpuIndexIMIPQv2(GpuResourcesProvider *provider,
                  const faiss::IndexIVFPQ *index,
                  GpuIndexIMIPQConfig config = GpuIndexIMIPQConfig());

  /// Construct an empty index
  GpuIndexIMIPQv2(std::shared_ptr<GpuResources> resources,
                  const faiss::IndexIVFPQ *index,
                  GpuIndexIMIPQConfig config = GpuIndexIMIPQConfig());

  /// Construct an empty index
  GpuIndexIMIPQv2(GpuResourcesProvider *provider, int dims,
                  int coarseCodebookSize, int subQuantizers, int bitsPerCode,
                  GpuIndexIMIPQConfig config = GpuIndexIMIPQConfig());

  /// Construct an empty index
  GpuIndexIMIPQv2(std::shared_ptr<GpuResources> resources, int dims,
                  int coarseCodebookSize, int subQuantizers, int bitsPerCode,
                  GpuIndexIMIPQConfig config = GpuIndexIMIPQConfig());

  ~GpuIndexIMIPQv2() override;

  static size_t calcInvListsMemorySpaceSize(int numVecs, int numSubQuantizers,
                                            int bitsPerSubQuantizer,
                                            bool interleavedLayout,
                                            IndicesOptions options);

  static size_t calcMemorySpaceSize(int numTotalVecsCoarseQuantizer,
                                    int dimPerCodebook, bool useFloat16,
                                    int numVecs, int numSubQuantizers,
                                    int bitsPerSubQuantizer,
                                    bool interleavedLayout,
                                    IndicesOptions options);

  void updateExpectedNumAddsPerList(Index::idx_t n, const float *x);

  void applyExpectedNumAddsPerList();

  void resetExpectedNumAddsPerList();

  void copyPrecomputedCodesFrom(const float *precomputedCodes);

  /// Initialize ourselves from the given CPU index; will overwrite
  /// all data in ourselves
  void copyFrom(const faiss::IndexIVFPQ *index);

  /// Copy ourselves to the given CPU index; will overwrite all data
  /// in the index instance
  void copyTo(faiss::IndexIVFPQ *index) const;

  /// Enable or disable pre-computed codes
  void setPrecomputedCodes(bool enable);

  /// Are pre-computed codes enabled?
  bool getPrecomputedCodes() const;

  int getMaxListLength() const;

  /// Return the number of sub-quantizers we are using
  int getNumSubQuantizers() const;

  /// Return the number of bits per PQ code
  int getBitsPerCode() const;

  /// Return the number of centroids per PQ code (2^bits per code)
  int getCentroidsPerSubQuantizer() const;

  /// Clears out all inverted lists, but retains the coarse and
  /// product centroid information
  void reset() override;

  void train(Index::idx_t n, const float *x) override;

  /// For debugging purposes, return the list length of a particular
  /// list
  int getListLength(int listId) const;

  /// For debugging purposes, return the length of all lists
  int getAllListsLength() const;

  /// Return the encoded vector data contained in a particular inverted list,
  /// for debugging purposes.
  /// If gpuFormat is true, the data is returned as it is encoded in the
  /// GPU-side representation.
  /// Otherwise, it is converted to the CPU format.
  /// compliant format, while the native GPU format may differ.
  std::vector<uint8_t> getListVectorData(int listId,
                                         bool gpuFormat = false) const override;

  /// Return the vector indices contained in a particular inverted list, for
  /// debugging purposes.
  std::vector<Index::idx_t> getListIndices(int listId) const override;

  // returns subQuantizerCentroids (sub q)(code id)(sub dim)
  std::vector<float> getPQCentroids() const;

  // returns precomputedCodesVec (centroid id)(sub q)(code id)
  std::vector<float> getPrecomputedCodesVec() const;

  std::vector<float> calcTerm3(int n, const float *x);

  /// Like the CPU version, we expose a publically-visible ProductQuantizer for
  /// manipulation
  ProductQuantizer pq;

protected:
  /// Called from GpuIndex for add/add_with_ids
  void addImpl_(int n, const float *x, const Index::idx_t *ids) override;

  /// Called from GpuIndex for search
  void searchImpl_(int n, const float *x, int k, float *distances,
                   Index::idx_t *labels) const override;

private:
  void verifySettings_() const;

  void trainResidualQuantizer_(Index::idx_t n, const float *x);

private:
  GpuIndexIMIPQConfig imipqConfig_;

  /// Runtime override: whether or not we use precomputed tables
  bool usePrecomputedTables_;

  /// Number of sub-quantizers per encoded vector
  int subQuantizers_;

  /// Bits per sub-quantizer code
  int bitsPerCode_;

  /// Desired inverted list memory reservation
  size_t reserveMemoryVecs_;

  std::unique_ptr<std::unordered_map<int, int>> expectedNumAddsPerList;

  /// The product quantizer instance that we own; contains the
  /// inverted lists
  std::unique_ptr<IMIPQv2> index_;
};

} // namespace gpu
} // namespace faiss
