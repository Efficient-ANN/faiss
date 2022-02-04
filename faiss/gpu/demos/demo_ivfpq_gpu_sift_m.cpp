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
#include <faiss/IndexIVFPQ.h>
#include <faiss/MetricType.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/ThreadedIndex.h>
#include <faiss/index_io.h>
#include <faiss/utils/vecs_storage.h>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <string>
#include <sys/types.h>

void search(std::vector<faiss::gpu::GpuResourcesProvider *> &resVector,
            faiss::Index *index, float *queries, int *groundTruth,
            size_t numQueries, int kBegin, int kEnd, int groundTruthK) {
  std::vector<int> kList = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
  clock_t tStart, tEnd;
  double tGpu;

  for (int i = kBegin > 0 ? kBegin : 0; i < kEnd && i < kList.size(); i++) {
    int k = kList[i];
    std::cout << "k: " << k << std::endl;

    std::vector<float> outDistances(numQueries * k);
    std::vector<faiss::Index::idx_t> outLabels(numQueries * k);

    tGpu = 0;
    constexpr int nRuns = 5;
    for (int j = 0; j < nRuns; j++) {
      tStart = clock();

      index->search(numQueries, queries, k, outDistances.data(),
                    outLabels.data());

      for (auto &&res : resVector) {
        faiss::gpu::CudaEvent copyEnd(
            res->getResources()->getDefaultStreamCurrentDevice());
        copyEnd.cpuWaitOnEvent();
      }

      tEnd = clock();
      tGpu += (double)(tEnd - tStart) / CLOCKS_PER_SEC;
    }

    std::cout << "IVFPQ search time on GPU: " << tGpu / nRuns << std::endl;

    int n_1 = 0, n_10 = 0, n_100 = 0, n_1024 = 0;
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
          if (b < 1024) {
            n_1024++;
          }
          break;
        }
      }
    }
    std::cout << "R@1 = " << n_1 / double(numQueries) << std::endl;
    std::cout << "R@10 = " << n_10 / double(numQueries) << std::endl;
    std::cout << "R@100 = " << n_100 / double(numQueries) << std::endl;
    std::cout << "R@1024 = " << n_1024 / double(numQueries) << std::endl;
  }
}

size_t calcIvfStructureMemSize(size_t d, size_t coarseCodebookSize,
                               size_t numSubQuantizers,
                               size_t nbitsSubQuantizer, int maxPageSize) {
  size_t subCodebookSize = 1 << nbitsSubQuantizer;
  size_t coarseQuantizerMemSize = faiss::gpu::utils::roundUp(
      d * coarseCodebookSize * sizeof(float), (size_t)maxPageSize);
  size_t normMemSize = faiss::gpu::utils::roundUp(
      coarseCodebookSize * sizeof(float), (size_t)maxPageSize);
  size_t productQuantizerMemSize =
      2 * faiss::gpu::utils::roundUp(d * subCodebookSize * sizeof(float),
                                     (size_t)maxPageSize);
  size_t precomputedMemSize = faiss::gpu::utils::roundUp(
      coarseCodebookSize * subCodebookSize * numSubQuantizers * sizeof(float),
      (size_t)maxPageSize);
  size_t codesPointersMemSize = faiss::gpu::utils::roundUp(
      coarseCodebookSize * sizeof(void *), (size_t)maxPageSize);
  size_t idsPointersMemSize = faiss::gpu::utils::roundUp(
      coarseCodebookSize * sizeof(void *), (size_t)maxPageSize);
  size_t listsLengthsMemSize = faiss::gpu::utils::roundUp(
      coarseCodebookSize * sizeof(int), (size_t)maxPageSize);
  return subCodebookSize + coarseQuantizerMemSize + normMemSize +
         productQuantizerMemSize + precomputedMemSize + codesPointersMemSize +
         idsPointersMemSize + listsLengthsMemSize;
}

void initResourcesMultiGpu(
    int ngpus,
    std::unordered_map<faiss::gpu::AllocType, size_t>
        &allocSizePerTypeMapPerGpu,
    size_t tempMemory,
    std::vector<faiss::gpu::GpuResourcesProvider *> &resVector,
    std::vector<int> &devs, bool allocLogging) {
  for (int i = 0; i < ngpus; i++) {
    faiss::gpu::StandardGpuResources *res;
    res = new faiss::gpu::StandardGpuResources(allocSizePerTypeMapPerGpu);
    res->setLogMemoryAllocations(allocLogging);
    res->setTempMemory(tempMemory);
    resVector.push_back(res);
    devs.push_back(i);
  }
}

template <bool isVecFloat>
void demo_ivfpq(int d, int coarseCodebookSize, int numSubQuantizers,
                int nbitsSubQuantizer, std::string fileNameTraining,
                size_t numTrainingVecs, std::string fileNameIndexing,
                size_t numIndexingVecs, std::string fileNameQueries,
                size_t queriesOffset, std::string fileNameGroundTruth,
                int numQueriesBegin, int numQueriesEnd, int nprobeBegin,
                int nprobeEnd, int kBegin, int kEnd, bool usePrecomputed,
                int ngpus, bool useShards, size_t safeMemMargin,
                std::string fileNameCoarseQuantizer, std::string fileNameIndex,
                bool profile, bool allocLogging) {
  size_t devFree = 0;
  size_t devTotal = 0;
  constexpr int maxPageSize = 2 * 1024 * 1024; // 2MB

  size_t numIndexingVecsPerGpu;

  if (useShards) {
    numIndexingVecsPerGpu = numIndexingVecs / ngpus + numIndexingVecs % ngpus;
  } else {
    numIndexingVecsPerGpu = numIndexingVecs;
  }

  CUDA_VERIFY(cudaMemGetInfo(&devFree, &devTotal));
  std::cout << "-------Memory-------" << std::endl;
  std::cout << "Free: " << devFree << std::endl;
  std::cout << "Total: " << devTotal << std::endl;

  faiss::gpu::IndicesOptions indiceOptions = faiss::gpu::INDICES_32_BIT;

  size_t fixedMemSize = 0;
  auto allocSizePerTypeMap =
      faiss::gpu::GpuIndexIVFPQ::getInvListsAllocSizePerTypeInfo(
          numIndexingVecs, numSubQuantizers, nbitsSubQuantizer, false,
          indiceOptions);

  for (auto &&allocSizePerType : allocSizePerTypeMap) {
    size_t allocSize =
        faiss::gpu::utils::roundUp(allocSizePerType.second, (size_t)256);
    fixedMemSize += faiss::gpu::utils::roundUp(allocSize, (size_t)maxPageSize);
  }

  size_t fixedMemSizePerGpu = 0;
  auto allocSizePerTypeMapPerGpu =
      faiss::gpu::GpuIndexIVFPQ::getInvListsAllocSizePerTypeInfo(
          numIndexingVecsPerGpu, numSubQuantizers, nbitsSubQuantizer, false,
          indiceOptions);

  for (auto &&allocSizePerType : allocSizePerTypeMapPerGpu) {
    size_t allocSize =
        faiss::gpu::utils::roundUp(allocSizePerType.second, (size_t)256);
    fixedMemSizePerGpu +=
        faiss::gpu::utils::roundUp(allocSize, (size_t)maxPageSize);
  }

  size_t ivfStructureMemSize = calcIvfStructureMemSize(
      d, coarseCodebookSize, numSubQuantizers, nbitsSubQuantizer, maxPageSize);

  size_t devFreeLimit = std::min(devFree, safeMemMargin);
  size_t tempMemory = devFreeLimit - fixedMemSize - ivfStructureMemSize;
  tempMemory = tempMemory / 256 * 256;

  size_t tempMemoryPerGpu =
      devFreeLimit - fixedMemSizePerGpu - ivfStructureMemSize;
  tempMemoryPerGpu = tempMemoryPerGpu / 256 * 256;

  std::cout << "tempMemoryPerGpu: " << tempMemoryPerGpu << std::endl;
  std::cout << "fixedMemSize: " << fixedMemSize << std::endl;
  std::cout << "fixedMemSizePerGpu: " << fixedMemSizePerGpu << std::endl;
  std::cout << "ivfStructureMemSize: " << ivfStructureMemSize << std::endl;
  std::cout << "safeMemMargin: " << safeMemMargin << std::endl;
  std::cout << "devFreeLimit: " << devFreeLimit << std::endl;
  std::cout << "tempMemory: " << tempMemory << std::endl;

  faiss::Index *indexMultiGpu = nullptr;
  faiss::gpu::GpuIndexIVFPQConfig config;
  std::vector<faiss::gpu::GpuResourcesProvider *> resVector;
  std::vector<int> devs;
  clock_t tStart, tEnd;
  double tGpu;
  int dRead;

  config.memorySpace = faiss::gpu::MemorySpace::Fixed;
  // config.flatConfig.memorySpace = faiss::gpu::MemorySpace::Fixed;
  config.indicesOptions = indiceOptions;
  config.usePrecomputedTables = usePrecomputed;
  int nlist = coarseCodebookSize;

  faiss::Index *indexCpu = nullptr;

  if (!fileNameIndex.empty()) {
    FILE *f = fopen(fileNameIndex.c_str(), "rb");
    if (f) {
      fclose(f);
      tStart = clock();
      indexCpu = dynamic_cast<faiss::IndexIVFPQ *>(
          faiss::read_index(fileNameIndex.c_str()));
      tEnd = clock();
      tGpu = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
      std::cout << "Time to load index from memory: " << tGpu << std::endl;
    }
  }

  if (!indexCpu) {
    { // indexing
      faiss::gpu::StandardGpuResources res(allocSizePerTypeMap);
      res.setLogMemoryAllocations(allocLogging);
      res.setTempMemory(tempMemory);
      faiss::gpu::GpuIndexIVFPQ *ivfpq;
      ivfpq = new faiss::gpu::GpuIndexIVFPQ(&res, d, nlist, numSubQuantizers,
                                            nbitsSubQuantizer, faiss::METRIC_L2,
                                            config);

      { // train
        bool storeCoarseQuantizer = true;
        if (!fileNameCoarseQuantizer.empty()) {
          FILE *f = fopen(fileNameCoarseQuantizer.c_str(), "rb");
          if (f) {
            fclose(f);
            faiss::IndexFlat *preBuildCoarseIndexCpu =
                dynamic_cast<faiss::IndexFlat *>(
                    faiss::read_index(fileNameCoarseQuantizer.c_str()));
            ivfpq->quantizer->copyFrom(preBuildCoarseIndexCpu);
            delete preBuildCoarseIndexCpu;
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
        ivfpq->train(numTrainingVecs, trainingVecs);
        tEnd = clock();
        tGpu = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
        std::cout << "IVFPQ train time on GPU: " << tGpu << std::endl;
        delete trainingVecs;

        if (storeCoarseQuantizer) {
          faiss::Index *coarseIndexCpu =
              faiss::gpu::index_gpu_to_cpu(ivfpq->quantizer);
          faiss::gpu::CudaEvent cloneEnd(
              res.getResources()->getDefaultStreamCurrentDevice());
          cloneEnd.cpuWaitOnEvent();
          faiss::write_index(coarseIndexCpu, fileNameCoarseQuantizer.c_str());
          delete coarseIndexCpu;
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
        numVecsTile = std::min(numVecsTile, (size_t)10000);
        numVecsTile = std::max(numVecsTile, (size_t)1);
        tStart = clock();
        for (size_t i = 0; i < numIndexingVecs; i += numVecsTile) {
          size_t currentNumVecsTile =
              std::min(numVecsTile, numIndexingVecs - i);
          float *indexingVecs;
          if (isVecFloat) {
            indexingVecs = faiss::fvecs_read(fileNameIndexing.c_str(),
                                             currentNumVecsTile, i, &dRead);
          } else {
            indexingVecs = faiss::bvecs_read(fileNameIndexing.c_str(),
                                             currentNumVecsTile, i, &dRead);
          }
          assert(d == dRead);
          ivfpq->updateExpectedNumAddsPerList(currentNumVecsTile, indexingVecs);
          faiss::gpu::CudaEvent updateEnd(
              res.getResources()->getDefaultStreamCurrentDevice());
          updateEnd.cpuWaitOnEvent();
          delete indexingVecs;
        }

        ivfpq->applyExpectedNumAddsPerList();
        faiss::gpu::CudaEvent applyEnd(
            res.getResources()->getDefaultStreamCurrentDevice());
        applyEnd.cpuWaitOnEvent();
        tEnd = clock();
        tGpu = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
        std::cout << "IVFPQ reserve time on GPU: " << tGpu << std::endl;
        ivfpq->resetExpectedNumAddsPerList();
      }

      CUDA_VERIFY(cudaMemGetInfo(&devFree, &devTotal));
      std::cout << "-------Memory-------" << std::endl;
      std::cout << "Free: " << devFree << std::endl;
      std::cout << "Total: " << devTotal << std::endl;

      { // add
        size_t maxAddTileSize = (size_t)8 * 1024 * 1024 * 1024;
        size_t numVecsTile = maxAddTileSize / (d * sizeof(float));
        numVecsTile = std::min(numVecsTile, numIndexingVecs);
        numVecsTile = std::min(numVecsTile, (size_t)10000);
        numVecsTile = std::max(numVecsTile, (size_t)1);
        tStart = clock();
        for (size_t i = 0; i < numIndexingVecs; i += numVecsTile) {
          size_t currentNumVecsTile =
              std::min(numVecsTile, numIndexingVecs - i);
          float *indexingVecs;
          if (isVecFloat) {
            indexingVecs = faiss::fvecs_read(fileNameIndexing.c_str(),
                                             currentNumVecsTile, i, &dRead);
          } else {
            indexingVecs = faiss::bvecs_read(fileNameIndexing.c_str(),
                                             currentNumVecsTile, i, &dRead);
          }
          assert(d == dRead);
          ivfpq->add(currentNumVecsTile, indexingVecs);
          delete indexingVecs;
        }
        tEnd = clock();
        tGpu = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
        std::cout << "IVFPQ add time on GPU: " << tGpu << std::endl;
      }

      std::cout << "maxListLength: " << ivfpq->getMaxListLength() << std::endl;

      indexCpu = faiss::gpu::index_gpu_to_cpu(ivfpq);
      faiss::gpu::CudaEvent cloneEnd(
          res.getResources()->getDefaultStreamCurrentDevice());
      cloneEnd.cpuWaitOnEvent();

      delete ivfpq;
    }

    if (!fileNameIndex.empty()) {
      std::cout << "writing: " << fileNameIndex << "...";
      faiss::write_index(indexCpu, fileNameIndex.c_str());
      std::cout << "done" << std::endl;
    }
  }

  if (profile) {
    faiss::gpu::GpuMultipleClonerOptions options;
    options.memorySpace = config.memorySpace;
    options.indicesOptions = config.indicesOptions;
    options.usePrecomputed = config.usePrecomputedTables;
    options.precomputeCodesOnCpu = config.precomputeCodesOnCpu;
    options.shard = useShards;
    options.shard_type = 1;

    std::cout << "Ininting resource for multiple GPUs" << std::endl;
    initResourcesMultiGpu(ngpus, allocSizePerTypeMapPerGpu, tempMemoryPerGpu,
                          resVector, devs, allocLogging);

    std::cout << "Moving index from cpu to multiple GPUs: " << std::endl;
    tStart = clock();
    indexMultiGpu = faiss::gpu::index_cpu_to_gpu_multiple(resVector, devs,
                                                          indexCpu, &options);
    tEnd = clock();
    tGpu = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
    std::cout << "Index moved in " << tGpu << std::endl;
  }

  delete indexCpu;

  CUDA_VERIFY(cudaMemGetInfo(&devFree, &devTotal));
  std::cout << "-------Memory-------" << std::endl;
  std::cout << "Free: " << devFree << std::endl;
  std::cout << "Total: " << devTotal << std::endl;

  if (profile) {
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

        faiss::ThreadedIndex<faiss::Index> *threadedIndex =
            dynamic_cast<faiss::ThreadedIndex<faiss::Index> *>(indexMultiGpu);

        if (threadedIndex) {
          // multi GPU
          for (int k = 0; k < threadedIndex->count(); k++) {
            faiss::gpu::GpuIndexIVFPQ *ivfpq =
                dynamic_cast<faiss::gpu::GpuIndexIVFPQ *>(threadedIndex->at(k));
            ivfpq->setNumProbes(nprobe);
          }
        } else {
          // single GPU
          faiss::gpu::GpuIndexIVFPQ *ivfpq =
              dynamic_cast<faiss::gpu::GpuIndexIVFPQ *>(indexMultiGpu);
          ivfpq->setNumProbes(nprobe);
        }

        search(resVector, indexMultiGpu, queries, groundTruth, numQueries,
               kBegin, kEnd, dRead);
      }
    }

    delete queries;
    delete groundTruth;
  }

  delete indexMultiGpu;

  for (int i = 0; i < resVector.size(); i++) {
    delete resVector[i];
  }
}

int main(int argc, char **argv) {
  if (argc <= 18) {
    std::cout << "There must be 18 or more parameters" << std::endl;
    return 1;
  }

  int d, coarseCodebookSize, numSubQuantizers, nbitsSubQuantizer, queriesOffset,
      numQueriesBegin, numQueriesEnd, kBegin, kEnd, nprobeBegin, nprobeEnd,
      isFloat, usePrecomputed, numThreads, ngpus, useShards, profile,
      allocLogging;
  size_t numTrainingVecs, numIndexingVecs;
  std::string fileNameTraining, fileNameIndexing, fileNameQueries,
      fileNameGroundTruth, fileNameCoarseQuantizer, fileNameIndex;
  size_t safeMemMargin;

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
  usePrecomputed = argc > 19 ? std::stoi(argv[19]) : 1;
  numThreads = argc > 20 ? std::stoi(argv[20]) : 1;
  ngpus = argc > 21 ? std::stoi(argv[21]) : 2;
  useShards = argc > 22 ? std::stoi(argv[22]) : 0;
  safeMemMargin = argc > 23 ? std::stoul(argv[23]) : 0;
  fileNameCoarseQuantizer = argc > 24 ? argv[24] : "";
  fileNameIndex = argc > 25 ? argv[25] : "";
  profile = argc > 26 ? std::stoi(argv[26]) : 1;
  allocLogging = argc > 27 ? std::stoi(argv[27]) : 0;

  omp_set_num_threads(numThreads);

  std::cout << std::setprecision(6) << std::fixed;

  if (isFloat == 1) {
    demo_ivfpq<true>(d, coarseCodebookSize, numSubQuantizers, nbitsSubQuantizer,
                     fileNameTraining, numTrainingVecs, fileNameIndexing,
                     numIndexingVecs, fileNameQueries, queriesOffset,
                     fileNameGroundTruth, numQueriesBegin, numQueriesEnd,
                     nprobeBegin, nprobeEnd, kBegin, kEnd, usePrecomputed == 1,
                     ngpus, useShards, safeMemMargin, fileNameCoarseQuantizer,
                     fileNameIndex, profile, allocLogging);
  } else {
    demo_ivfpq<false>(
        d, coarseCodebookSize, numSubQuantizers, nbitsSubQuantizer,
        fileNameTraining, numTrainingVecs, fileNameIndexing, numIndexingVecs,
        fileNameQueries, queriesOffset, fileNameGroundTruth, numQueriesBegin,
        numQueriesEnd, nprobeBegin, nprobeEnd, kBegin, kEnd,
        usePrecomputed == 1, ngpus, useShards, safeMemMargin,
        fileNameCoarseQuantizer, fileNameIndex, profile, allocLogging);
  }
  return 0;
}
