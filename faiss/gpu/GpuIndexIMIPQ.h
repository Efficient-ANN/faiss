/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/gpu/GpuIndexIMI.h>
#include <faiss/impl/ProductQuantizer.h>
#include <memory>
#include <unordered_map>
#include <vector>

namespace faiss {
namespace gpu {

class IMIPQ;

struct GpuIndexIMIPQConfig : public GpuIndexIMIConfig {
  inline GpuIndexIMIPQConfig()
      : useFloat16LookupTables(false), usePrecomputedTables(true),
        interleavedLayout(false), useMMCodeDistance(false),
        precomputeCodesOnCpu(false) {}

  /// Whether or not float16 residual distance tables are used in the
  /// list scanning kernels.
  bool useFloat16LookupTables;

  /// Whether or not we enable the precomputed table option for
  /// search, which can substantially increase the memory requirement.
  bool usePrecomputedTables;

  /// Use the alternative memory layout for the IVF lists
  /// WARNING: this is a feature under development, do not use!
  bool interleavedLayout;

  /// Use GEMM-backed computation of PQ code distances for the no precomputed
  /// table version of IMIPQ.
  /// This is for debugging purposes, it should not substantially affect the
  /// results one way for another.
  ///
  /// Note that MM code distance is enabled automatically if one uses a number
  /// of dimensions per sub-quantizer that is not natively specialized (an odd
  /// number like 7 or so).
  bool useMMCodeDistance;
  bool precomputeCodesOnCpu;
};

class GpuIndexIMIPQ : public GpuIndexIMI {
public:
  /// Construct an empty index
  GpuIndexIMIPQ(GpuResourcesProvider *provider, int dims,
                int coarseCodebookSize, int subQuantizers, int bitsPerCode,
                GpuIndexIMIPQConfig config = GpuIndexIMIPQConfig());

  /// Construct an empty index
  GpuIndexIMIPQ(std::shared_ptr<GpuResources> resources, int dims,
                int coarseCodebookSize, int subQuantizers, int bitsPerCode,
                GpuIndexIMIPQConfig config = GpuIndexIMIPQConfig());

  ~GpuIndexIMIPQ() override;

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

  /// Reserve GPU memory in our inverted lists for this number of vectors
  void reserveMemory(size_t numVecs);

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

  /// After adding vectors, one can call this to reclaim device memory
  /// to exactly the amount needed. Returns space reclaimed in bytes
  size_t reclaimMemory();

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
  std::unique_ptr<IMIPQ> index_;
};

} // namespace gpu
} // namespace faiss
