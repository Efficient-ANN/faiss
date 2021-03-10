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
#include <faiss/IndexPQ.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexIMIPQ.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/index_io.h>
#include <faiss/utils/vecs_storage.h>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <string>
#include <sys/types.h>

void search(faiss::gpu::StandardGpuResources *res, faiss::Index *index,
            float *queries, int *groundTruth, size_t numQueries, int kBegin,
            int kEnd, int groundTruthK) {
  std::vector<int> kList = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
  clock_t tStart, tEnd;
  double tGpu;

  for (int i = kBegin > 0 ? kBegin : 0; i < kEnd && i < kList.size(); i++) {
    int k = kList[i];
    std::cout << "k: " << k << std::endl;

    std::vector<float> outDistances(numQueries * k);
    std::vector<faiss::Index::idx_t> outLabels(numQueries * k);

    tStart = clock();
    index->search(numQueries, queries, k, outDistances.data(),
                  outLabels.data());
    faiss::gpu::CudaEvent copyEnd(
        res->getResources()->getDefaultStreamCurrentDevice());
    copyEnd.cpuWaitOnEvent();
    tEnd = clock();
    tGpu = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
    std::cout << "IMIPQ search time on GPU: " << tGpu << std::endl;

    int n_1 = 0, n_10 = 0, n_100 = 0, n_1000 = 0;
    for (int a = 0; a < numQueries; a++) {
      faiss::Index::idx_t firstGrounTruthId = groundTruth[a * groundTruthK];
      for (int b = 0; b < k; b++) {
        if (outLabels[a * k + b] == firstGrounTruthId) {
          if (b < 1) {
            n_1++;
          }
          if (b < 10) {
            n_10++;
          }
          if (b < 100) {
            n_100++;
          }
          if (b < 1000) {
            n_1000++;
          }
          break;
        }
      }
    }
    std::cout << "R@1 = " << n_1 / double(numQueries) << std::endl;
    std::cout << "R@10 = " << n_10 / double(numQueries) << std::endl;
    std::cout << "R@100 = " << n_100 / double(numQueries) << std::endl;
    std::cout << "R@1000 = " << n_1000 / double(numQueries) << std::endl;
  }
}

size_t calcImiStructureMemSize(size_t d, size_t coarseCodebookSize,
                               size_t numSubQuantizers,
                               size_t nbitsSubQuantizer) {
  size_t subCodebookSize = 1 << nbitsSubQuantizer;
  size_t coarseQuantizerMemSize = faiss::gpu::utils::roundUp(
      d * coarseCodebookSize * sizeof(float), (size_t)256);
  size_t normMemSize = faiss::gpu::utils::roundUp(
      2 * coarseCodebookSize * sizeof(float), (size_t)256);
  size_t productQuantizerMemSize = faiss::gpu::utils::roundUp(
      d * subCodebookSize * numSubQuantizers * sizeof(float), (size_t)256);
  size_t precomputedMemSize = faiss::gpu::utils::roundUp(
      coarseCodebookSize * subCodebookSize * numSubQuantizers * sizeof(float),
      (size_t)256);
  size_t codesPointersMemSize =
      coarseCodebookSize * coarseCodebookSize * sizeof(void *);
  size_t idsPointersMemSize =
      coarseCodebookSize * coarseCodebookSize * sizeof(void *);
  size_t listsLengthsMemSize =
      coarseCodebookSize * coarseCodebookSize * sizeof(int);
  return subCodebookSize + coarseQuantizerMemSize + normMemSize +
         productQuantizerMemSize + precomputedMemSize + codesPointersMemSize +
         idsPointersMemSize + listsLengthsMemSize;
}

template <bool isVecFloat>
void demo_imipq(int d, int coarseCodebookSize, int numSubQuantizers,
                int nbitsSubQuantizer, std::string fileNameTraining,
                size_t numTrainingVecs, std::string fileNameIndexing,
                size_t numIndexingVecs, std::string fileNameQueries,
                size_t queriesOffset, std::string fileNameGroundTruth,
                int numQueriesBegin, int numQueriesEnd, int nprobeBegin,
                int nprobeEnd, int kBegin, int kEnd,
                std::string fileNameCoarseQuantizer) {
  size_t devFree = 0;
  size_t devTotal = 0;

  CUDA_VERIFY(cudaMemGetInfo(&devFree, &devTotal));
  std::cout << "-------Memory-------" << std::endl;
  std::cout << "Free: " << devFree << std::endl;
  std::cout << "Total: " << devTotal << std::endl;

  faiss::gpu::IndicesOptions indiceOptions = faiss::gpu::INDICES_32_BIT;
  /*
  size_t fixedMemSize = faiss::gpu::GpuIndexIMIPQ::calcMemorySpaceSize(
      coarseCodebookSize * 2, d / 2, false, numIndexingVecs, numSubQuantizers,
      nbitsSubQuantizer, false, indiceOptions);
  */
  size_t fixedMemSize = faiss::gpu::GpuIndexIMIPQ::calcInvListsMemorySpaceSize(
      numIndexingVecs, numSubQuantizers, nbitsSubQuantizer, false,
      indiceOptions);
  std::cout << "fixedMemSize: " << fixedMemSize << std::endl;

  size_t imiStructureMemSize = calcImiStructureMemSize(
      d, coarseCodebookSize, numSubQuantizers, nbitsSubQuantizer);
  std::cout << "imiStructureMemSize: " << imiStructureMemSize << std::endl;

  faiss::gpu::StandardGpuResources res(fixedMemSize);
  constexpr size_t safeMargin = 300 * 1024 * 1024; // 300MB
  size_t tempMemory =
      devFree - faiss::gpu::utils::roundUp(fixedMemSize + 256, (size_t)256) -
      imiStructureMemSize - safeMargin;
  // res.noTempMemory();
  res.setTempMemory(tempMemory);
  std::cout << "tempMemory: " << tempMemory << std::endl;

  faiss::gpu::GpuIndexIMIPQConfig config;
  config.memorySpace = faiss::gpu::MemorySpace::Fixed;
  // config.multiIndexConfig.memorySpace = faiss::gpu::MemorySpace::Fixed;
  config.indicesOptions = indiceOptions;
  config.usePrecomputedTables = true;

  faiss::gpu::GpuIndexIMIPQ imipqGpu(
      &res, d, coarseCodebookSize, numSubQuantizers, nbitsSubQuantizer, config);
  clock_t tStart, tEnd;
  double tGpu;
  int dRead;

  { // train
    bool storeCoarseQuantizer = true;
    if (!fileNameCoarseQuantizer.empty()) {
      FILE *f = fopen(fileNameCoarseQuantizer.c_str(), "rb");
      if (f) {
        fclose(f);
        faiss::MultiIndexQuantizer *cpu_index =
            dynamic_cast<faiss::MultiIndexQuantizer *>(
                faiss::read_index(fileNameCoarseQuantizer.c_str()));
        imipqGpu.quantizer->copyFrom(cpu_index);
        delete cpu_index;
        storeCoarseQuantizer = false;
      }
    }

    float *trainingVecs;
    if (isVecFloat) {
      trainingVecs = faiss::fvecs_read(fileNameTraining.c_str(),
                                       numTrainingVecs, 0, &dRead);
    } else {
      trainingVecs = faiss::bvecs_read(fileNameTraining.c_str(),
                                       numTrainingVecs, 0, &dRead);
    }
    assert(d == dRead);
    tStart = clock();
    imipqGpu.train(numTrainingVecs, trainingVecs);
    tEnd = clock();
    tGpu = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
    std::cout << "IMIPQ train time on GPU: " << tGpu << std::endl;
    delete trainingVecs;

    if (storeCoarseQuantizer) {
      faiss::Index *cpu_index =
          faiss::gpu::index_gpu_to_cpu(imipqGpu.quantizer);
      faiss::write_index(cpu_index, fileNameCoarseQuantizer.c_str());
      delete cpu_index;
    }
  }

  CUDA_VERIFY(cudaMemGetInfo(&devFree, &devTotal));
  std::cout << "-------Memory-------" << std::endl;
  std::cout << "Free: " << devFree << std::endl;
  std::cout << "Total: " << devTotal << std::endl;

  { // reserve
    size_t maxAddTileSize = (size_t)8 * 1024 * 1024 * 1024;
    size_t numVecsTile = maxAddTileSize / (d * sizeof(float));
    numVecsTile = std::min(numVecsTile, numIndexingVecs);
    numVecsTile = std::max(numVecsTile, (size_t)1);
    tStart = clock();
    for (size_t i = 0; i < numIndexingVecs; i += numVecsTile) {
      size_t currentNumVecsTile = std::min(numVecsTile, numIndexingVecs - i);
      float *indexingVecs;
      if (isVecFloat) {
        indexingVecs = faiss::fvecs_read(fileNameIndexing.c_str(),
                                         currentNumVecsTile, i, &dRead);
      } else {
        indexingVecs = faiss::bvecs_read(fileNameIndexing.c_str(),
                                         currentNumVecsTile, i, &dRead);
      }
      assert(d == dRead);
      imipqGpu.updateExpectedNumAddsPerList(currentNumVecsTile, indexingVecs);
      faiss::gpu::CudaEvent updateEnd(
          res.getResources()->getDefaultStreamCurrentDevice());
      updateEnd.cpuWaitOnEvent();
      delete indexingVecs;
    }

    imipqGpu.applyExpectedNumAddsPerList();
    faiss::gpu::CudaEvent applyEnd(
        res.getResources()->getDefaultStreamCurrentDevice());
    applyEnd.cpuWaitOnEvent();
    tEnd = clock();
    tGpu = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
    std::cout << "IMIPQ reserve time on GPU: " << tGpu << std::endl;
    imipqGpu.resetExpectedNumAddsPerList();
  }

  CUDA_VERIFY(cudaMemGetInfo(&devFree, &devTotal));
  std::cout << "-------Memory-------" << std::endl;
  std::cout << "Free: " << devFree << std::endl;
  std::cout << "Total: " << devTotal << std::endl;

  { // add
    size_t maxAddTileSize = (size_t)8 * 1024 * 1024 * 1024;
    size_t numVecsTile = maxAddTileSize / (d * sizeof(float));
    numVecsTile = std::min(numVecsTile, numIndexingVecs);
    numVecsTile = std::max(numVecsTile, (size_t)1);
    for (size_t i = 0; i < numIndexingVecs; i += numVecsTile) {
      size_t currentNumVecsTile = std::min(numVecsTile, numIndexingVecs - i);
      float *indexingVecs;
      if (isVecFloat) {
        indexingVecs = faiss::fvecs_read(fileNameIndexing.c_str(),
                                         currentNumVecsTile, i, &dRead);
      } else {
        indexingVecs = faiss::bvecs_read(fileNameIndexing.c_str(),
                                         currentNumVecsTile, i, &dRead);
      }
      assert(d == dRead);
      tStart = clock();
      imipqGpu.add(currentNumVecsTile, indexingVecs);
      tEnd = clock();
      tGpu = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
      std::cout << "IMIPQ add time on GPU: " << tGpu << std::endl;
      delete indexingVecs;
    }
  }

  std::vector<int> numQueriesList = {1, 1000, 8192, 10000};
  std::vector<int> nprobeList = {1,  2,   4,   8,   16,   32,
                                 64, 128, 256, 512, 1024, 2048};

  float *queries;
  if (isVecFloat) {
    queries = faiss::fvecs_read(fileNameQueries.c_str(),
                                (size_t)numQueriesList[numQueriesEnd - 1],
                                queriesOffset, &dRead);
  } else {
    queries = faiss::bvecs_read(fileNameQueries.c_str(),
                                (size_t)numQueriesList[numQueriesEnd - 1],
                                queriesOffset, &dRead);
  }
  assert(d == dRead);
  int *groundTruth =
      faiss::ivecs_read(fileNameGroundTruth.c_str(),
                        numQueriesList[numQueriesEnd - 1], 0, &dRead);

  CUDA_VERIFY(cudaMemGetInfo(&devFree, &devTotal));
  std::cout << "-------Memory-------" << std::endl;
  std::cout << "Free: " << devFree << std::endl;
  std::cout << "Total: " << devTotal << std::endl;

  for (int i = numQueriesBegin > 0 ? numQueriesBegin : 0;
       i < numQueriesEnd && i < numQueriesList.size(); i++) {
    int numQueries = numQueriesList[i];
    std::cout << "numOfQueries: " << numQueries
              << " ===============" << std::endl;
    for (int j = nprobeBegin > 0 ? nprobeBegin : 0;
         j < nprobeEnd && j < nprobeList.size(); j++) {
      int nprobe = nprobeList[j];
      std::cout << "nprobe: " << nprobe << "---------" << std::endl;
      imipqGpu.setNumProbes(nprobe);
      search(&res, &imipqGpu, queries, groundTruth, numQueries, kBegin, kEnd,
             dRead);
    }
  }
  delete queries;
  delete groundTruth;
}

int main(int argc, char **argv) {
  if (argc <= 18) {
    std::cout << "There must be 18 or more parameters" << std::endl;
    return 1;
  }

  int d, coarseCodebookSize, numSubQuantizers, nbitsSubQuantizer, queriesOffset,
      numQueriesBegin, numQueriesEnd, kBegin, kEnd, nprobeBegin, nprobeEnd,
      isFloat, numThreads;
  size_t numTrainingVecs, numIndexingVecs;
  std::string fileNameTraining, fileNameIndexing, fileNameQueries,
      fileNameGroundTruth, fileNameCoarseQuantizer;

  d = std::stoi(argv[1]);
  coarseCodebookSize = std::stoi(argv[2]);
  numSubQuantizers = std::stoi(argv[3]);
  nbitsSubQuantizer = std::stoi(argv[4]);
  fileNameTraining = argv[5];
  numTrainingVecs = std::stoul(argv[6]);
  fileNameIndexing = argv[7];
  numIndexingVecs = std::stoul(argv[8]);
  fileNameQueries = argv[9];
  queriesOffset = std::stoul(argv[10]);
  fileNameGroundTruth = argv[11];
  numQueriesBegin = std::stoi(argv[12]);
  numQueriesEnd = std::stoi(argv[13]);
  nprobeBegin = std::stoi(argv[14]);
  nprobeEnd = std::stoi(argv[15]);
  kBegin = std::stoi(argv[16]);
  kEnd = std::stoi(argv[17]);
  isFloat = std::stoi(argv[18]);
  numThreads = argc > 19 ? std::stoi(argv[19]) : 1;
  fileNameCoarseQuantizer = argc > 20 ? argv[20] : "";

  omp_set_num_threads(numThreads);

  std::cout << std::setprecision(6) << std::fixed;

  if (isFloat == 1) {
    demo_imipq<true>(d, coarseCodebookSize, numSubQuantizers, nbitsSubQuantizer,
                     fileNameTraining, numTrainingVecs, fileNameIndexing,
                     numIndexingVecs, fileNameQueries, queriesOffset,
                     fileNameGroundTruth, numQueriesBegin, numQueriesEnd,
                     nprobeBegin, nprobeEnd, kBegin, kEnd,
                     fileNameCoarseQuantizer);
  } else {
    demo_imipq<false>(d, coarseCodebookSize, numSubQuantizers,
                      nbitsSubQuantizer, fileNameTraining, numTrainingVecs,
                      fileNameIndexing, numIndexingVecs, fileNameQueries,
                      queriesOffset, fileNameGroundTruth, numQueriesBegin,
                      numQueriesEnd, nprobeBegin, nprobeEnd, kBegin, kEnd,
                      fileNameCoarseQuantizer);
  }
  return 0;
}
