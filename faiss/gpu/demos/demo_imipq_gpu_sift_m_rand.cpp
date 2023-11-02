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
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexPQ.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexIMIPQv2.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/gpu/utils/Timer.h>
#include <faiss/impl/ThreadedIndex.h>
#include <faiss/index_io.h>
#include <faiss/utils/random.h>
#include <faiss/utils/vecs_storage.h>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <memory>
#include <vector>
#include <omp.h>
#include <string>
#include <sys/types.h>

void search(const faiss::Index *index, float *queries, int *groundTruth,
            size_t numQueries, int kBegin, int kEnd, int groundTruthK, int nRuns) {
  std::vector<int> kList = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
  clock_t tStart, tEnd;
  double tGpu;

  if (nRuns <= 0) {
    nRuns = 1;
  }

  for (int i = kBegin > 0 ? kBegin : 0; i < kEnd && i < kList.size(); i++) {
    int k = kList[i];
    std::cout << "k: " << k << std::endl;

    try {
      std::vector<float> outDistances(numQueries * k);
      std::vector<faiss::Index::idx_t> outLabels(numQueries * k);

      tGpu = 0;
      for (int j = 0; j < nRuns; j++) {
        faiss::gpu::CpuTimer timer;

        tStart = clock();

        index->search(numQueries, queries, k, outDistances.data(),
                      outLabels.data());

        tEnd = clock();
        std::cout << "Time before sync: " << (double)(tEnd - tStart) / CLOCKS_PER_SEC << std::endl;

        faiss::gpu::synchronizeAllDevices();

        tEnd = clock();
        tGpu += (double)(tEnd - tStart) / CLOCKS_PER_SEC;
        std::cout << "Timer after sync (millis)" << timer.elapsedMilliseconds() << std::endl;
      }

      std::cout << "IMIPQ search time on GPU: " << tGpu / nRuns << std::endl;

      if (groundTruth != nullptr) {
        int n_1 = 0, n_10 = 0, n_100 = 0, n_1000, n_1024 = 0;
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
        std::cout << "R@1000 = " << n_1000 / double(numQueries) << std::endl;
        std::cout << "R@1024 = " << n_1024 / double(numQueries) << std::endl;
      } else {
        std::cout << "R@1 = NOT COMPUTED" << std::endl;
        std::cout << "R@10 = NOT COMPUTED" << std::endl;
        std::cout << "R@100 = NOT COMPUTED" << std::endl;
        std::cout << "R@1000 = NOT COMPUTED" << std::endl;
        std::cout << "R@1024 = NOT COMPUTED" << std::endl;
      }
    } catch (const std::exception &e) {
      faiss::gpu::synchronizeAllDevices();
      std::cout << "K EXCEPTION: " << e.what() << std::endl;
      if (i == 0 || i == kBegin) {
        throw;
      }
    } catch (...) {
      faiss::gpu::synchronizeAllDevices();
      std::cout << "K UNKNOWN EXCEPTION" << std::endl;
      if (i == 0 || i == kBegin) {
        throw;
      }
    }
  }
}

struct RandomContext {
  int64_t seed = 0;
};

float *fvecs_rand(size_t num, size_t dim, RandomContext &randomContext) {
  size_t size = num * dim;
  float *vecs = new float[size];
  faiss::float_rand(vecs, size, randomContext.seed);
  ++randomContext.seed;
  return vecs;
}

float *vecs_load(bool isVecFloat, std::string fileName, size_t num, int d, RandomContext &randomContext, size_t numOffset = 0) {
  int dRead;
  if (fileName.empty()) {
    return fvecs_rand(num, d, randomContext);
  }
  float *vecs;
  if (isVecFloat) {
    vecs = faiss::fvecs_read(fileName.c_str(), num, numOffset, &dRead);
  } else {
    vecs = faiss::bvecs_read(fileName.c_str(), num, numOffset, &dRead);
  }
  assert(d == dRead);
  return vecs;
}

size_t calcImiStructureMemSize(size_t d, size_t coarseCodebookSize,
                               size_t numSubQuantizers,
                               size_t nbitsSubQuantizer, int roundSize) {
  size_t subCodebookSize = 1 << nbitsSubQuantizer;
  size_t coarseQuantizerMemSize = faiss::gpu::utils::roundUp(
      d * coarseCodebookSize * sizeof(float), (size_t)roundSize);
  size_t normMemSize = faiss::gpu::utils::roundUp(
      2 * coarseCodebookSize * sizeof(float), (size_t)roundSize);
  size_t productQuantizerMemSize =
      2 * faiss::gpu::utils::roundUp(d * subCodebookSize * sizeof(float),
                                     (size_t)roundSize);
  size_t precomputedMemSize = faiss::gpu::utils::roundUp(
      coarseCodebookSize * subCodebookSize * numSubQuantizers * sizeof(float),
      (size_t)roundSize);
  size_t listOffsetMemSize = faiss::gpu::utils::roundUp(
      coarseCodebookSize * coarseCodebookSize * sizeof(unsigned int),
      (size_t)roundSize);
  return subCodebookSize + coarseQuantizerMemSize + normMemSize +
         productQuantizerMemSize + precomputedMemSize + listOffsetMemSize;
}

void initResourcesMultiGpu(
    int ngpus,
    std::unordered_map<faiss::gpu::AllocType, size_t>
        &allocSizePerTypeMapPerGpu,
    size_t tempMemory,
    std::vector<faiss::gpu::GpuResourcesProvider *> &resVector,
    std::vector<int> &devs, bool allocLogging, int pinnedMemoryMode) {
  for (int i = 0; i < ngpus; i++) {
    faiss::gpu::StandardGpuResources *res;
    res = new faiss::gpu::StandardGpuResources(allocSizePerTypeMapPerGpu);
    res->setLogMemoryAllocations(allocLogging);
    res->setTempMemory(tempMemory);
    if (pinnedMemoryMode == 0) {
      res->setPinnedMemory(0);
    }
    resVector.push_back(res);
    devs.push_back(i);
  }
}

void printDeviceMemory(size_t devFree, size_t devTotal, int deviceId = 0) {
  std::cout << "-------Memory-------" << std::endl;
  std::cout << "Device: " << deviceId << std::endl;
  std::cout << "Free: " << devFree << std::endl;
  std::cout << "Total: " << devTotal << std::endl;
}

void printDeviceMemory(int deviceId = 0) {
  size_t devFree = 0;
  size_t devTotal = 0;
  faiss::gpu::setCurrentDevice(deviceId);
  CUDA_VERIFY(cudaMemGetInfo(&devFree, &devTotal));
  printDeviceMemory(devFree, devTotal, deviceId);
}

void getAvailableMemoryPerDevice(size_t &devFree, size_t &devTotal, int nProcessPerDevice = 1) {
  size_t currDevFree = 0;
  size_t currDevTotal = 0;
  bool devFreeIsSet = false;

  devFree = 0;
  devTotal = 0;
  int numDevices = faiss::gpu::getNumDevices();
  for (int i = 0; i < numDevices; i++) {
    if (!devFreeIsSet) {
      faiss::gpu::setCurrentDevice(i);
      CUDA_VERIFY(cudaMemGetInfo(&currDevFree, &currDevTotal));
      devFreeIsSet = true;
      devFree = currDevFree;
      devTotal = currDevTotal;
    } else {
      devFree = std::min(devFree, currDevFree);
      devTotal = std::min(devTotal, currDevTotal);
    }
  }
  devFree /= nProcessPerDevice;
  devTotal /= nProcessPerDevice;
}

template <class IndexT>
IndexT * loadIndexToCpu(std::string fileName) {
  IndexT *indexCpu = nullptr;
  if (!fileName.empty()) {
    FILE *f = fopen(fileName.c_str(), "rb");
    if (f) {
      clock_t tStart, tEnd, tGpu;
      fclose(f);
      tStart = clock();
      indexCpu = dynamic_cast<IndexT *>(faiss::read_index(fileName.c_str()));
      tEnd = clock();
      tGpu = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
      std::cout << "Time to load index from memory: " << tGpu << std::endl;
    }
  }
  return indexCpu;
}

void demo_imipq(bool isVecFloat, int d, int coarseCodebookSize, int numSubQuantizers,
                int nbitsSubQuantizer, std::string fileNameTraining,
                size_t numTrainingVecs, std::string fileNameIndexing,
                size_t numIndexingVecs, std::string fileNameQueries,
                size_t queriesOffset, std::string fileNameGroundTruth,
                int numQueriesBegin, int numQueriesEnd, int nprobeBegin,
                int nprobeEnd, int kBegin, int kEnd, int ngpus, bool useShards,
                size_t safeMemMargin, std::string fileNameCoarseQuantizer,
                std::string fileNameIndex, bool profile, bool allocLogging, bool verbose, int nRuns, int pinnedMemoryMode) {
  RandomContext randomContext;
  
  size_t devFree = 0;
  size_t devTotal = 0;
  constexpr int roundSize = 256;

  size_t numIndexingVecsPerGpu;

  if (useShards) {
    numIndexingVecsPerGpu = numIndexingVecs / ngpus + numIndexingVecs % ngpus;
  } else {
    numIndexingVecsPerGpu = numIndexingVecs;
  }

  getAvailableMemoryPerDevice(devFree, devTotal);
  printDeviceMemory(devFree, devFree);

  faiss::gpu::IndicesOptions indiceOptions = faiss::gpu::INDICES_32_BIT;

  size_t fixedMemSize = 0;
  auto allocSizePerTypeMap =
      faiss::gpu::GpuIndexIMIPQv2::getInvListsAllocSizePerTypeInfo(
          numIndexingVecs, numSubQuantizers, nbitsSubQuantizer, false,
          indiceOptions);

  for (auto &&allocSizePerType : allocSizePerTypeMap) {
    size_t allocSize =
        faiss::gpu::utils::roundUp(allocSizePerType.second, (size_t)256);
    fixedMemSize += faiss::gpu::utils::roundUp(allocSize, (size_t)roundSize);
  }

  size_t fixedMemSizePerGpu = 0;
  auto allocSizePerTypeMapPerGpu =
      faiss::gpu::GpuIndexIMIPQv2::getInvListsAllocSizePerTypeInfo(
          numIndexingVecsPerGpu, numSubQuantizers, nbitsSubQuantizer, false,
          indiceOptions);

  for (auto &&allocSizePerType : allocSizePerTypeMapPerGpu) {
    size_t allocSize =
        faiss::gpu::utils::roundUp(allocSizePerType.second, (size_t)256);
    fixedMemSizePerGpu +=
        faiss::gpu::utils::roundUp(allocSize, (size_t)roundSize);
  }

  size_t imiStructureMemSize = calcImiStructureMemSize(
      d, coarseCodebookSize, numSubQuantizers, nbitsSubQuantizer, roundSize);

  size_t devFreeLimit = std::min(devFree, safeMemMargin);

  size_t tempMemory = devFreeLimit - fixedMemSize - imiStructureMemSize;
  tempMemory = tempMemory / 256 * 256;

  size_t tempMemoryPerGpu =
      devFreeLimit - fixedMemSizePerGpu - imiStructureMemSize;
  tempMemoryPerGpu = tempMemoryPerGpu / 256 * 256;

  std::cout << "tempMemoryPerGpu: " << tempMemoryPerGpu << std::endl;
  std::cout << "fixedMemSize: " << fixedMemSize << std::endl;
  std::cout << "fixedMemSizePerGpu: " << fixedMemSizePerGpu << std::endl;
  std::cout << "imiStructureMemSize: " << imiStructureMemSize << std::endl;
  std::cout << "safeMemMargin: " << safeMemMargin << std::endl;
  std::cout << "devFreeLimit: " << devFreeLimit << std::endl;
  std::cout << "tempMemory: " << tempMemory << std::endl;

  faiss::gpu::GpuIndexIMIPQConfig config;
  std::vector<faiss::gpu::GpuResourcesProvider *> resVector;
  std::vector<int> devs;
  clock_t tStart, tEnd;
  double tGpu;
  int dRead;

  config.memorySpace = faiss::gpu::MemorySpace::Fixed;
  // config.multiIndexConfig.memorySpace = faiss::gpu::MemorySpace::Fixed;
  config.indicesOptions = indiceOptions;
  config.usePrecomputedTables = true;

  if (pinnedMemoryMode == 2) {
    config.forcePinnedMemory = true;
  }

  std::unique_ptr<faiss::Index> indexCpu(loadIndexToCpu<faiss::IndexIVFPQ>(fileNameIndex));
  if (!indexCpu) {
    { // indexing
      faiss::gpu::StandardGpuResources res(allocSizePerTypeMap);
      res.setLogMemoryAllocations(allocLogging);
      res.setTempMemory(tempMemory);
      std::unique_ptr<faiss::gpu::GpuIndexIMIPQv2> imipqGpu(
        new faiss::gpu::GpuIndexIMIPQv2(&res, d, coarseCodebookSize, numSubQuantizers, nbitsSubQuantizer, config));
      
      // set verbose according to provided parameter
      imipqGpu->verbose = verbose;

      { // build coarse quantizer
        std::unique_ptr<faiss::MultiIndexQuantizer> preBuildCoarseIndexCpu(loadIndexToCpu<faiss::MultiIndexQuantizer>(fileNameCoarseQuantizer));
        if (preBuildCoarseIndexCpu) {
            imipqGpu->quantizer->copyFrom(preBuildCoarseIndexCpu.get());
        } else {
          // train
          std::unique_ptr<float> trainingVecs(vecs_load(isVecFloat, fileNameTraining, numTrainingVecs, d, randomContext, 0));
          tStart = clock();
          imipqGpu->train(numTrainingVecs, trainingVecs.get());
          tEnd = clock();
          tGpu = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
          std::cout << "IMIPQ train time on GPU: " << tGpu << std::endl;

          // save coase quantizer
          if (!fileNameCoarseQuantizer.empty()) {
            std::unique_ptr<faiss::Index> coarseIndexCpu(faiss::gpu::index_gpu_to_cpu(imipqGpu->quantizer));
            faiss::gpu::CudaEvent cloneEnd(res.getResources()->getDefaultStreamCurrentDevice());
            cloneEnd.cpuWaitOnEvent();
            faiss::write_index(coarseIndexCpu.get(), fileNameCoarseQuantizer.c_str());
          }
        }
      }

      printDeviceMemory();

      int64_t initSeed = 0;
      int64_t endSeed = 0;
      
      // set maximum available memory for tiling over vectors while adding them to the GPU
      size_t maxAddTileSize = (size_t)8 * 1024 * 1024 * 1024;
      size_t numVecsTile = maxAddTileSize / (d * sizeof(float));
      numVecsTile = std::min(numVecsTile, numIndexingVecs);
      numVecsTile = std::min(numVecsTile, (size_t)10000);
      numVecsTile = std::max(numVecsTile, (size_t)1);
      
      // save current initial seed for using it again while adding the vectors
      initSeed = randomContext.seed;

      { // reserve space for indexing        
        tStart = clock();
        for (size_t i = 0; i < numIndexingVecs; i += numVecsTile) {
          size_t currentNumVecsTile = std::min(numVecsTile, numIndexingVecs - i);
          
          std::unique_ptr<float> indexingVecs(vecs_load(isVecFloat, fileNameIndexing, currentNumVecsTile, d, randomContext, i));

          imipqGpu->updateExpectedNumAddsPerList(currentNumVecsTile, indexingVecs.get());
          faiss::gpu::CudaEvent updateEnd(res.getResources()->getDefaultStreamCurrentDevice());
          updateEnd.cpuWaitOnEvent();
        }

        imipqGpu->applyExpectedNumAddsPerList();
        faiss::gpu::CudaEvent applyEnd(res.getResources()->getDefaultStreamCurrentDevice());
        applyEnd.cpuWaitOnEvent();

        tEnd = clock();
        tGpu = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
        std::cout << "IMIPQ reserve time on GPU: " << tGpu << std::endl;
        imipqGpu->resetExpectedNumAddsPerList();
      }

      // save it for assertion
      endSeed = randomContext.seed;

      printDeviceMemory();

      randomContext.seed = initSeed;

      { // add
        tStart = clock();
        for (size_t i = 0; i < numIndexingVecs; i += numVecsTile) {
          size_t currentNumVecsTile = std::min(numVecsTile, numIndexingVecs - i);
          std::unique_ptr<float> indexingVecs(vecs_load(isVecFloat, fileNameIndexing, currentNumVecsTile, d, randomContext, i));
          imipqGpu->add(currentNumVecsTile, indexingVecs.get());
        }
        tEnd = clock();
        tGpu = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
        std::cout << "IMIPQ add time on GPU: " << tGpu << std::endl;
      }

      assert(randomContext.seed == endSeed);

      std::cout << "maxListLength: " << imipqGpu->getMaxListLength()
                << std::endl;

      indexCpu.reset(faiss::gpu::index_gpu_to_cpu(imipqGpu.get()));
      faiss::gpu::CudaEvent cloneEnd(
          res.getResources()->getDefaultStreamCurrentDevice());
      cloneEnd.cpuWaitOnEvent();
    }

    if (!fileNameIndex.empty()) {
      std::cout << "writing: " << fileNameIndex << "...";
      faiss::write_index(indexCpu.get(), fileNameIndex.c_str());
      std::cout << "done" << std::endl;
    }
  }

  if (profile) {
    faiss::gpu::GpuMultipleClonerOptions options;
    options.memorySpace = config.memorySpace;
    options.indicesOptions = config.indicesOptions;
    options.usePrecomputed = config.usePrecomputedTables;
    options.precomputeCodesOnCpu = config.precomputeCodesOnCpu;
    options.forcePinnedMemory = config.forcePinnedMemory;
    options.shard = useShards;
    options.shard_type = 1;
    options.verbose = verbose;

    std::cout << "Ininting resource for multiple GPUs" << std::endl;
    initResourcesMultiGpu(ngpus, allocSizePerTypeMapPerGpu, tempMemoryPerGpu,
                          resVector, devs, allocLogging, pinnedMemoryMode);

    std::cout << "Moving index from cpu to multiple GPUs: " << std::endl;
    tStart = clock();
    std::unique_ptr<faiss::Index> indexMultiGpu(faiss::gpu::index_cpu_to_gpu_multiple(resVector, devs, indexCpu.get(), &options));
    faiss::gpu::synchronizeAllDevices();
    tEnd = clock();
    tGpu = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
    std::cout << "Index moved in " << tGpu << std::endl;

    indexMultiGpu->verbose = verbose;

    getAvailableMemoryPerDevice(devFree, devTotal);
    printDeviceMemory(devFree, devFree);
    
    std::vector<int> numQueriesList = {
        1,      1000,   8192,   10000,   16384,   32768,   65536,   100000,
        131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216};
    numQueriesEnd = std::min(numQueriesEnd, (int)numQueriesList.size());
    std::vector<int> nprobeList = {
        1,    2,    4,    8,    16,   32,   64,   128,  256,   512,   1024,
        2048, 2194, 2352, 2521, 2702, 2896, 4096, 8192, 16384, 32768, 65536};

    std::unique_ptr<float> queries(vecs_load(isVecFloat, fileNameQueries, (size_t)numQueriesList[numQueriesEnd - 1], d, randomContext, queriesOffset));

    std::unique_ptr<int> groundTruth;
    if (!fileNameGroundTruth.empty()) {
      groundTruth.reset(faiss::ivecs_read(fileNameGroundTruth.c_str(), numQueriesList[numQueriesEnd - 1], 0, &dRead));
    }

    getAvailableMemoryPerDevice(devFree, devTotal);
    printDeviceMemory(devFree, devFree);

    for (int i = numQueriesBegin > 0 ? numQueriesBegin : 0; i < numQueriesEnd;
         i++) {
      int numQueries = numQueriesList[i];
      std::cout << "numOfQueries: " << numQueries
                << " ===============" << std::endl;
      try {
        for (int j = nprobeBegin > 0 ? nprobeBegin : 0;
             j < nprobeEnd && j < nprobeList.size(); j++) {
          int nprobe = nprobeList[j];
          std::cout << "nprobe: " << nprobe << "---------" << std::endl;

          try {
            faiss::ThreadedIndex<faiss::Index> *threadedIndex =
                dynamic_cast<faiss::ThreadedIndex<faiss::Index> *>(
                    indexMultiGpu.get());

            if (threadedIndex) {
              // multi GPU
              for (int k = 0; k < threadedIndex->count(); k++) {
                faiss::gpu::GpuIndexIMIPQv2 *imipqGpu =
                    dynamic_cast<faiss::gpu::GpuIndexIMIPQv2 *>(
                        threadedIndex->at(k));
                imipqGpu->setNumProbes(nprobe);
                imipqGpu->verbose = verbose;
                std::cout << "Gpu: " << imipqGpu->getDevice()
                          << ", maxListLength: " << imipqGpu->getMaxListLength()
                          << ", nlist: " << imipqGpu->nlist
                          << ", ntotal: " << imipqGpu->ntotal
                          << std::endl;
              }
            } else {
              // single GPU
              faiss::gpu::GpuIndexIMIPQv2 *imipqGpu =
                  dynamic_cast<faiss::gpu::GpuIndexIMIPQv2 *>(indexMultiGpu.get());
              imipqGpu->setNumProbes(nprobe);
              imipqGpu->verbose = verbose;
              std::cout << "Gpu: " << imipqGpu->getDevice()
                        << ", maxListLength: " << imipqGpu->getMaxListLength()
                        << ", nlist: " << imipqGpu->nlist
                        << ", ntotal: " << imipqGpu->ntotal
                        << std::endl;
            }

            search(indexMultiGpu.get(), queries.get(), groundTruth.get(), numQueries, kBegin,
                   kEnd, dRead, nRuns);

          } catch (...) {
            std::cout << "NPROBE UNKNOWN EXCEPTION" << std::endl;
            if (j == 0 || j == nprobeBegin) {
              throw;
            }
          }
        }
      } catch (...) {
        std::cout << "QUERY UNKNOWN EXCEPTION" << std::endl;
      }
    }
  }

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
      isFloat, numThreads, ngpus, useShards, profile, allocLogging, verbose, nRuns, pinnedMemoryMode;
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
  numThreads = argc > 19 ? std::stoi(argv[19]) : 1;
  ngpus = argc > 20 ? std::stoi(argv[20]) : 2;
  useShards = argc > 21 ? std::stoi(argv[21]) : 0;
  safeMemMargin = argc > 22 ? std::stoul(argv[22]) : 0;
  fileNameCoarseQuantizer = argc > 23 ? argv[23] : "";
  fileNameIndex = argc > 24 ? argv[24] : "";
  profile = argc > 25 ? std::stoi(argv[25]) : 1;
  allocLogging = argc > 26 ? std::stoi(argv[26]) : 0;
  verbose = argc > 27 ? std::stoi(argv[27]) : 0;
  nRuns = argc > 28 ? std::stoi(argv[28]) : 5;
  pinnedMemoryMode = argc > 29 ? std::stoi(argv[29]) : 1;

  omp_set_num_threads(numThreads);

  std::cout << std::setprecision(6) << std::fixed;

  demo_imipq(isFloat, d, coarseCodebookSize, numSubQuantizers, nbitsSubQuantizer,
              fileNameTraining, numTrainingVecs, fileNameIndexing,
              numIndexingVecs, fileNameQueries, queriesOffset,
              fileNameGroundTruth, numQueriesBegin, numQueriesEnd,
              nprobeBegin, nprobeEnd, kBegin, kEnd, ngpus, useShards,
              safeMemMargin, fileNameCoarseQuantizer, fileNameIndex,
              profile, allocLogging, verbose, nRuns, pinnedMemoryMode);
  return 0;
}
