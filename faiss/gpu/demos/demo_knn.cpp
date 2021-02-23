/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <ctime>
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/utils/vecs_storage.h>
#include <iostream>
#include <omp.h>
#include <string>
#include <sys/types.h>

void knnOutOfMemory(int d, std::string fileNameIndexing, int numIndexingVecs,
                    std::string fileNameQueries, int numQueriesVecs, int offset,
                    int k, std::string outDirectory, bool isVecFloat,
                    bool isGpu) {
  faiss::Index *index;
  clock_t tStart, tEnd;
  double duration;
  int batchSize, currentBatch, dRead;

  size_t devFree = 0;
  size_t devTotal = 0;

  CUDA_VERIFY(cudaMemGetInfo(&devFree, &devTotal));
  std::cout << "-------Memory-------" << std::endl;
  std::cout << "Free: " << devFree << std::endl;
  std::cout << "Total: " << devTotal << std::endl;

  float *queries;
  if (isVecFloat) {
    queries = faiss::fvecs_read(fileNameQueries.c_str(), numQueriesVecs,
                                (size_t)offset, &dRead);
  } else {
    queries = faiss::bvecs_read(fileNameQueries.c_str(), numQueriesVecs,
                                (size_t)offset, &dRead);
  }
  assert(d == dRead);

  batchSize = 5000000;
  currentBatch = 0;
  for (int i = 0; i < numIndexingVecs; i += batchSize) {
    int currentNumVecsTile = std::min(batchSize, numIndexingVecs - i);
    faiss::gpu::StandardGpuResources *res;

    if (isGpu) {
      res = new faiss::gpu::StandardGpuResources();
      res->setTempMemory((size_t)512 * 1024 * 1024);
      index = new faiss::gpu::GpuIndexFlatL2(res, d);
    } else {
      res = NULL;
      index = new faiss::IndexFlat(d);
    }

    { // add
      float *indexingVecs;
      if (isVecFloat) {
        indexingVecs = faiss::fvecs_read(fileNameIndexing.c_str(),
                                         currentNumVecsTile, (size_t)i, &dRead);
      } else {
        indexingVecs = faiss::bvecs_read(fileNameIndexing.c_str(),
                                         currentNumVecsTile, (size_t)i, &dRead);
      }
      assert(d == dRead);
      tStart = clock();
      index->add(currentNumVecsTile, indexingVecs);
      tEnd = clock();
      duration = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
      std::cout << "IMIPQ add time on GPU: " << duration << std::endl;
      delete[] indexingVecs;
    }

    std::vector<float> outDistances(numQueriesVecs * k);
    std::vector<faiss::Index::idx_t> outLabels(numQueriesVecs * k);

    { // search
      tStart = clock();
      index->search(numQueriesVecs, queries, k, outDistances.data(),
                    outLabels.data());
      faiss::gpu::CudaEvent copyEnd(
          res->getResources()->getDefaultStreamCurrentDevice());
      copyEnd.cpuWaitOnEvent();
      tEnd = clock();
      duration = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
      std::cout << "Flat search time on GPU: " << duration << std::endl;
    }

    { // store
      tStart = clock();
      std::string outDistancesDirectory =
          outDirectory + "distances" + std::to_string(currentBatch) + ".fvecs";
      faiss::fvecs_write(outDistancesDirectory.c_str(), numQueriesVecs, k,
                         outDistances.data());

      std::vector<int> outLabelsInt(numQueriesVecs * k);

      for (int j = 0; j < outLabels.size(); j++) {
        outLabelsInt[j] = outLabels[j];
      }

      std::string outLabelsDirectory =
          outDirectory + "labels" + std::to_string(currentBatch) + ".ivecs";
      faiss::ivecs_write(outLabelsDirectory.c_str(), numQueriesVecs, k,
                         outLabelsInt.data());
      tEnd = clock();
      duration = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
      std::cout << "Store time: " << duration << std::endl;
    }

    currentBatch++;
    delete index;
    delete res;
    res = NULL;
  }

  delete[] queries;
}

int main(int argc, char **argv) {
  if (argc <= 8) {
    std::cout << "There must be 7 or more parameters" << std::endl;
    return 1;
  }

  std::string fileNameIndexing, fileNameQueries, outDirectory;
  int d, numIndexingVecs, numQueriesVecs, offset, k, isVecFloat, isGpu,
      numThreads;

  d = std::stoi(argv[1]);
  fileNameIndexing = argv[2];
  numIndexingVecs = std::stoi(argv[3]);
  fileNameQueries = argv[4];
  numQueriesVecs = std::stoi(argv[5]);
  offset = std::stoi(argv[6]);
  k = std::stoi(argv[7]);
  outDirectory = argv[8];
  isVecFloat = argc > 9 ? std::stoi(argv[9]) : 1;
  isGpu = argc > 10 ? std::stoi(argv[10]) : 1;
  numThreads = argc > 11 ? std::stoi(argv[11]) : 1;

  omp_set_num_threads(numThreads);

  knnOutOfMemory(d, fileNameIndexing, numIndexingVecs, fileNameQueries,
                 numQueriesVecs, offset, k, outDirectory, isVecFloat == 1,
                 isGpu == 1);

  return 0;
}
