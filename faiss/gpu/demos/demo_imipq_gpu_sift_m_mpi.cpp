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
#include <faiss/utils/Heap.h>
#include <faiss/utils/random.h>
#include <faiss/utils/vecs_storage.h>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <memory>
#include <vector>
#include <omp.h>
#include <string>
#include <sstream>
#include <sys/types.h>
#include <omp.h>
#include <mpi.h>

void processPrint(int processRank, std::string str) {
  std::stringstream out;
  out << "Process " << processRank << " << " <<  str << std::endl;
  std::cout << out.str();
}

void processPrint(int processRank, std::stringstream &str) {
  processPrint(processRank, str.str());
}

template <class C>
void
merge_tables(long n, long k, int nProcesses,
             float *distances,
             faiss::Index::idx_t *labels,
             const std::vector<float>& all_distances,
             const std::vector<faiss::Index::idx_t>& all_labels,
             const std::vector<long>& translations) {
  if (k == 0) {
    return;
  }

  long stride = n * k;
#pragma omp parallel
  {
    std::vector<int> buf (2 * nProcesses);
    int * pointer = buf.data();
    int * processIds = pointer + nProcesses;
    std::vector<float> buf2 (nProcesses);
    float * heap_vals = buf2.data();
#pragma omp for
    for (long i = 0; i < n; i++) {
      // the heap maps values to the process where they are
      // produced.
      const float *D_in = all_distances.data() + i * k;
      const faiss::Index::idx_t *I_in = all_labels.data() + i * k;
      int heap_size = 0;

      for (long currRank = 0; currRank < nProcesses; currRank++) {
        pointer[currRank] = 0;
        if (I_in[stride * currRank] >= 0) {
          faiss::heap_push<C> (++heap_size, heap_vals, processIds,
                        D_in[stride * currRank], currRank);
        }
      }

      float *D = distances + i * k;
      faiss::Index::idx_t *I = labels + i * k;

      for (int j = 0; j < k; j++) {
        if (heap_size == 0) {
          I[j] = -1;
          D[j] = C::neutral();
        } else {
          // pop best element
          int currRank = processIds[0];
          int & p = pointer[currRank];
          D[j] = heap_vals[0];
          I[j] = I_in[stride * currRank + p] + translations[currRank];

          faiss::heap_pop<C> (heap_size--, heap_vals, processIds);
          p++;
          if (p < k && I_in[stride * currRank + p] >= 0) {
            faiss::heap_push<C> (++heap_size, heap_vals, processIds,
                          D_in[stride * currRank + p], currRank);
          }
        }
      }
    }
  }
}

void search(int processRank, int nProcesses, bool shardPerProcess, int numIndexingVecs, int remainingIndexingVecs,
            const faiss::Index *index, float *queries, int *groundTruth,
            size_t numQueries, int kBegin, int kEnd, int groundTruthK, int nRuns, bool verbose = false) {
  std::vector<int> kList = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
  clock_t tStart, tEnd, tMid;
  double tGpu, tGpuStd, tCpu, tCpuStd, tTotal, tTotalStd;

  if (nRuns <= 0) {
    nRuns = 1;
  }
  
  int totalNumQueries = numQueries;
  int remainingNumQueries = totalNumQueries % nProcesses;
  if (!shardPerProcess) {
    if (nProcesses > 0) {
      numQueries = totalNumQueries / nProcesses;
      if (processRank == 0) {
        numQueries += remainingNumQueries;
      } else {
        queries += remainingNumQueries + numQueries * processRank;
      }
    }
  }

  std::stringstream queriesOut;
  queriesOut << "total # queries: " << totalNumQueries << ", ";
  queriesOut << "# queries per proccess: " << numQueries;
  processPrint(processRank, queriesOut);
  MPI_Barrier(MPI_COMM_WORLD);

  for (int i = kBegin > 0 ? kBegin : 0; i < kEnd && i < kList.size(); i++) {
    int k = kList[i];
    std::stringstream kOut;
    kOut << "k: " << k;
    processPrint(processRank, kOut);
    MPI_Barrier(MPI_COMM_WORLD);

    try {
      std::vector<float> heapDistances, outDistances;
      std::vector<faiss::Index::idx_t> heapLabels, outLabels;
      int outNum = numQueries * k;
      
      if (processRank == 0) {
        outDistances.resize(nProcesses * outNum);
        outLabels.resize(nProcesses * outNum);
      } else {
        outDistances.resize(outNum);
        outLabels.resize(outNum);
      }

      tGpu = 0;
      tGpuStd = 0;
      tCpu = 0;
      tCpuStd = 0;
      tTotal = 0;
      tTotalStd = 0;
      for (int j = 0; j < nRuns; j++) {
        faiss::gpu::CpuTimer timer;

        tStart = clock();

        index->search(numQueries, queries, k, outDistances.data(),
                      outLabels.data());

        faiss::gpu::synchronizeAllDevices();

        tEnd = clock();
        tGpu += (double)(tEnd - tStart) / CLOCKS_PER_SEC;
        tGpuStd += timer.elapsedMilliseconds();

        MPI_Barrier(MPI_COMM_WORLD);

        tMid = clock();

        if (nProcesses > 1) {
          if (processRank != 0) {
            // send distances & labels
            const int rankToReceive = 0;
            const int messageTag = 0;
            MPI_Send(outDistances.data(), outDistances.size(), MPI_FLOAT, rankToReceive, messageTag, MPI_COMM_WORLD);
            MPI_Send(outLabels.data(), outLabels.size(), MPI_LONG_LONG, rankToReceive, messageTag, MPI_COMM_WORLD);
            if (verbose) {
              processPrint(processRank, "Data sent");
            }
          } else {
            heapDistances.resize(outNum);
            heapLabels.resize(outNum);

            const int messageTag = 0;
            float *distanceAddress = outDistances.data();
            faiss::Index::idx_t *labelsAddress = outLabels.data();
            for (int sendingRank = 1; sendingRank < nProcesses; sendingRank++) {
              // receive the distances + labels right next the previous ones
              distanceAddress += outNum;
              labelsAddress += outNum;
              MPI_Recv(distanceAddress, outNum, MPI_FLOAT, sendingRank, messageTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              MPI_Recv(labelsAddress, outNum, MPI_LONG_LONG, sendingRank, messageTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              if (verbose) {
                processPrint(processRank, "Data received");
              }
            }

            std::vector<long> translations(nProcesses, 0);

            // In case we split the index per processes, we must shift the received labels
            if (shardPerProcess) {
              translations[0] = 0;
              translations[1] = remainingIndexingVecs + numIndexingVecs;
              for (int currRank = 1; currRank + 1 < nProcesses; currRank++) {
                translations[currRank + 1] = translations[currRank] + numIndexingVecs;
              }
            }
            merge_tables<faiss::CMin<float, int>>(numQueries, k, nProcesses, 
              heapDistances.data(), heapLabels.data(), outDistances, outLabels, translations);

          }
        }
        tEnd = clock();
        
        tCpu += (double)(tEnd - tMid) / CLOCKS_PER_SEC;
        tCpuStd += timer.elapsedMilliseconds() - tGpuStd;
        tTotal = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
        tTotalStd += timer.elapsedMilliseconds();
      }

      std::stringstream nRunsOut;
      nRunsOut << "Average number of runs: " << nRuns;
      processPrint(processRank, nRunsOut);

      std::stringstream timeGpuOut;
      timeGpuOut << "IMIPQ search time on Device Only (seconds): " << tGpu / nRuns << std::endl;
      timeGpuOut << "std timer (millis): " << tGpuStd / nRuns;
      processPrint(processRank, timeGpuOut);

      std::stringstream timeCpuOut;
      timeCpuOut << "IMIPQ search time on CPU (seconds): " << tCpu / nRuns << std::endl;
      timeCpuOut << "std timer (millis): " << tCpuStd / nRuns;
      processPrint(processRank, timeCpuOut);

      if (processRank == 0) {
        std::stringstream timeOut;
        timeOut << "IMIPQ search time on GPU (seconds): " << tTotal / nRuns << std::endl;
        timeOut << "std timer (millis): " << tTotalStd / nRuns;
        processPrint(processRank, timeOut);

        float *outDistancesData;
        faiss::Index::idx_t *outLabelsData;
        if (nProcesses == 1) {
          outDistancesData = outDistances.data();
          outLabelsData = outLabels.data();
        } else {
          // receive + merge distances & labels
          outDistancesData = heapDistances.data();
          outLabelsData = heapLabels.data();
        }

        if (verbose) {
          std::stringstream resultOut;
          resultOut << "# Result:" << std::endl;
          for (int a = 0; a < numQueries; a++) {
            resultOut << "  ## Query " << a << " :" << std::endl;
            for (int b = 0; b < k; b++) {
              int resPos = a * k + b;
              resultOut << "    [" << b << "]" << outDistancesData[resPos] << ", " 
                        << outLabelsData[resPos] << std::endl;
            }
          }
          processPrint(processRank, resultOut);
        }

        std::stringstream recallOut;
        if (groundTruth != nullptr) {
          int n_1 = 0, n_10 = 0, n_100 = 0, n_1000, n_1024 = 0;
          for (int a = 0; a < numQueries; a++) {
            faiss::Index::idx_t firstGrounTruthId = groundTruth[a * groundTruthK];
            for (int b = 0; b < k; b++) {
              if (outLabelsData[a * k + b] == firstGrounTruthId) {
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
          recallOut << "R@1 = " << n_1 / double(numQueries) << std::endl;
          recallOut << "R@10 = " << n_10 / double(numQueries) << std::endl;
          recallOut << "R@100 = " << n_100 / double(numQueries) << std::endl;
          recallOut << "R@1000 = " << n_1000 / double(numQueries) << std::endl;
          recallOut << "R@1024 = " << n_1024 / double(numQueries);
        } else {
          recallOut << "R@1 = NOT COMPUTED" << std::endl;
          recallOut << "R@10 = NOT COMPUTED" << std::endl;
          recallOut << "R@100 = NOT COMPUTED" << std::endl;
          recallOut << "R@1000 = NOT COMPUTED" << std::endl;
          recallOut << "R@1024 = NOT COMPUTED";
        }
        processPrint(processRank, recallOut);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    } catch (const std::exception &e) {
      std::stringstream eOut;
      MPI_Barrier(MPI_COMM_WORLD);
      faiss::gpu::synchronizeAllDevices();
      eOut << "K EXCEPTION: " << e.what() << std::endl;
      processPrint(processRank, eOut);
      if (i == 0 || i == kBegin) {
        throw;
      }
    } catch (...) {
      MPI_Barrier(MPI_COMM_WORLD);
      faiss::gpu::synchronizeAllDevices();
      processPrint(processRank, "K UNKNOWN EXCEPTION");
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

size_t roundMemAllocUp(size_t size) {
    return faiss::gpu::utils::roundUp(size, (size_t)256);
}

size_t roundMemAllocDown(size_t size) {
    return size / 256 * 256;
}

size_t calcFixedMemSize(std::unordered_map<faiss::gpu::AllocType, size_t> allocSizePerTypeMap) {
    size_t fixedMemSize = 0;
    for (auto &&allocSizePerType : allocSizePerTypeMap) {
        size_t allocSize = roundMemAllocUp(allocSizePerType.second);
        fixedMemSize += roundMemAllocUp(allocSize);
    }
    return fixedMemSize;
}

size_t calcImiStructureMemSize(size_t d, size_t coarseCodebookSize,
                               size_t numSubQuantizers,
                               size_t nbitsSubQuantizer) {
  size_t subCodebookSize = 1 << nbitsSubQuantizer;
  size_t coarseQuantizerMemSize = roundMemAllocUp(d * coarseCodebookSize * sizeof(float));
  size_t normMemSize = roundMemAllocUp(2 * coarseCodebookSize * sizeof(float));
  size_t productQuantizerMemSize = 2 * roundMemAllocUp(d * subCodebookSize * sizeof(float));
  size_t precomputedMemSize = roundMemAllocUp(coarseCodebookSize * subCodebookSize * numSubQuantizers * sizeof(float));
  size_t listOffsetMemSize = roundMemAllocUp(coarseCodebookSize * coarseCodebookSize * sizeof(unsigned int));
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
    std::cout << "i:" << i << std::endl;
    devs.push_back(i);
  }
}

void printDeviceMemory(size_t devFree, size_t devTotal, int deviceId, int processRank) {
  std::stringstream out;
  out << std::endl;
  out << "-------Memory-------" << std::endl;
  out << "Device: " << deviceId << std::endl;
  out << "Free: " << devFree << std::endl;
  out << "Total: " << devTotal << std::endl;
  processPrint(processRank, out);
}

void printDeviceMemory(int deviceId = 0, int processRank = 0) {
  size_t devFree = 0;
  size_t devTotal = 0;
  faiss::gpu::setCurrentDevice(deviceId);
  CUDA_VERIFY(cudaMemGetInfo(&devFree, &devTotal));
  printDeviceMemory(devFree, devTotal, deviceId, processRank);
  faiss::gpu::setCurrentDevice(0);
}

void printAllDevicesMemory(int deviceIdInit = 0, int ngpus = 1, int processRank = 0) {
  std::stringstream out;
  for (int deviceId = deviceIdInit; deviceId < ngpus; deviceId++) {
    size_t devFree, devTotal;
    CUDA_VERIFY(cudaMemGetInfo(&devFree, &devTotal));
    out << std::endl;
    out << "-------Memory-------" << std::endl;
    out << "Device: " << deviceId << std::endl;
    out << "Free: " << devFree << std::endl;
    out << "Total: " << devTotal << std::endl;
  }
  processPrint(processRank, out);
}

void getAvailableMemoryPerDevice(size_t &devFree, size_t &devTotal, int deviceIdInit = 0, int ngpus = 1, int nProcessesPerGpu = 1) {
  size_t currDevFree = 0;
  size_t currDevTotal = 0;
  bool devFreeIsSet = false;

  devFree = 0;
  devTotal = 0;
  for (int deviceId = deviceIdInit; deviceId < ngpus; deviceId++) {
    if (!devFreeIsSet) {
      faiss::gpu::setCurrentDevice(deviceId);
      CUDA_VERIFY(cudaMemGetInfo(&currDevFree, &currDevTotal));
      devFreeIsSet = true;
      devFree = currDevFree;
      devTotal = currDevTotal;
    } else {
      devFree = std::min(devFree, currDevFree);
      devTotal = std::min(devTotal, currDevTotal);
    }
  }
  faiss::gpu::setCurrentDevice(0);
  devFree /= nProcessesPerGpu;
  devTotal /= nProcessesPerGpu;
}

template <class IndexT>
IndexT * loadIndexToCpu(int processRank, std::string fileName) {
  IndexT *indexCpu = nullptr;
  if (!fileName.empty()) {
    FILE *f = fopen(fileName.c_str(), "rb");
    if (f) {
      clock_t tStart, tEnd;
      double tGpu;
      fclose(f);
      tStart = clock();
      indexCpu = dynamic_cast<IndexT *>(faiss::read_index(fileName.c_str()));
      tEnd = clock();
      tGpu = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
      std::stringstream out;
      out << "Time to load index from memory: " << tGpu;
      processPrint(processRank, out);
    }
  }
  return indexCpu;
}

int buildCoarseQuantizer(int processRank, faiss::gpu::GpuIndexIMIPQv2 *imipqGpu, std::string fileNameCoarseQuantizer, bool isVecFloat, std::string fileNameTraining,
  int numTrainingVecs, int d, RandomContext &randomContext, size_t readOffset = 0) {
  std::unique_ptr<faiss::Index> indexCpuTrainedOnly(loadIndexToCpu<faiss::IndexIVFPQ>(processRank, fileNameCoarseQuantizer));
  if (indexCpuTrainedOnly) {
    imipqGpu->copyFrom(dynamic_cast<faiss::IndexIVFPQ*>(indexCpuTrainedOnly.get()));
    return 0;
  }
  // train
  clock_t tStart, tEnd;
  double tGpu;
  std::unique_ptr<float> trainingVecs(vecs_load(isVecFloat, fileNameTraining, numTrainingVecs, d, randomContext, readOffset));
  tStart = clock();
  imipqGpu->train(numTrainingVecs, trainingVecs.get());
  tEnd = clock();
  tGpu = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
  std::stringstream outTrain;
  outTrain << "IMIPQ train time on GPU: " << tGpu;
  processPrint(processRank, outTrain);

  // save coase quantizer
  if (processRank == 0 && !fileNameCoarseQuantizer.empty()) {
    tStart = clock();
    indexCpuTrainedOnly.reset(faiss::gpu::index_gpu_to_cpu(imipqGpu));
    faiss::gpu::CudaEvent cloneEnd(imipqGpu->getResources()->getDefaultStreamCurrentDevice());
    cloneEnd.cpuWaitOnEvent();
    faiss::write_index(indexCpuTrainedOnly.get(), fileNameCoarseQuantizer.c_str());
    tEnd = clock();
    tGpu = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
    std::stringstream outSave;
    outSave << "IMIPQ writting coarse quantizer: " << tGpu;
    processPrint(processRank, outSave);
  }
  return 1;
}

void reserveIndexingSpace(int processRank, faiss::gpu::GpuIndexIMIPQv2 *imipqGpu, bool isVecFloat, std::string fileNameIndexing,
  int numIndexingVecs, int d, size_t numVecsTile, RandomContext &randomContext, size_t readOffset = 0) {     
  clock_t tStart, tEnd;
  double tGpu;
  tStart = clock();
  for (size_t i = 0; i < numIndexingVecs; i += numVecsTile) {
    size_t currentNumVecsTile = std::min(numVecsTile, numIndexingVecs - i);
    
    std::unique_ptr<float> indexingVecs(vecs_load(isVecFloat, fileNameIndexing, currentNumVecsTile, d, randomContext, readOffset + i));

    imipqGpu->updateExpectedNumAddsPerList(currentNumVecsTile, indexingVecs.get());
    faiss::gpu::CudaEvent updateEnd(imipqGpu->getResources()->getDefaultStreamCurrentDevice());
    updateEnd.cpuWaitOnEvent();
  }

  imipqGpu->applyExpectedNumAddsPerList();
  faiss::gpu::CudaEvent applyEnd(imipqGpu->getResources()->getDefaultStreamCurrentDevice());
  applyEnd.cpuWaitOnEvent();

  imipqGpu->resetExpectedNumAddsPerList();

  tEnd = clock();
  tGpu = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
  std::stringstream out;
  out << "IMIPQ reserve time on GPU: " << tGpu;
  processPrint(processRank, out);
}

void addToIndex(int processRank, faiss::gpu::GpuIndexIMIPQv2 *imipqGpu, bool isVecFloat, std::string fileNameIndexing,
  int numIndexingVecs, int d, size_t numVecsTile, RandomContext &randomContext, size_t readOffset = 0) {
  clock_t tStart, tEnd;
  double tGpu;
  tStart = clock();
  for (size_t i = 0; i < numIndexingVecs; i += numVecsTile) {
    size_t currentNumVecsTile = std::min(numVecsTile, numIndexingVecs - i);
    std::unique_ptr<float> indexingVecs(vecs_load(isVecFloat, fileNameIndexing, currentNumVecsTile, d, randomContext, i));
    imipqGpu->add(currentNumVecsTile, indexingVecs.get());
  }
  tEnd = clock();
  tGpu = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
  std::stringstream out;
  out << "IMIPQ add time on GPU: " << tGpu;
  processPrint(processRank, out);
}

void demo_imipq(bool isVecFloat, int d, int coarseCodebookSize, int numSubQuantizers,
                int nbitsSubQuantizer, std::string fileNameTraining,
                size_t numTrainingVecs, std::string fileNameIndexing,
                size_t numIndexingVecs, std::string fileNameQueries,
                size_t queriesOffset, std::string fileNameGroundTruth,
                int numQueriesBegin, int numQueriesEnd, int nprobeBegin,
                int nprobeEnd, int kBegin, int kEnd, int ngpus, bool useShards, 
                int nProcesses, int processRank, bool sharedGpuProcess,
                size_t safeMemMargin, std::string fileNameCoarseQuantizer,
                std::string fileNameIndex, bool profile, bool allocLogging, bool verbose, int nRuns, int pinnedMemoryMode) {
  RandomContext randomContext;
  
  int numDevices = faiss::gpu::getNumDevices();
  
  int totalGpus = ngpus;
  int totalNumIndexingVecs = numIndexingVecs;
  int remainingIndexingVecs = totalNumIndexingVecs % nProcesses;
  int deviceIdInit = 0;
  int nProcessesPerGpu = 1;
  
  if (sharedGpuProcess) {
    nProcessesPerGpu = nProcesses;
  } else {
    assert(ngpus % nProcesses == 0);
    ngpus /= nProcesses;
    deviceIdInit = processRank * ngpus;
  }

  bool shardPerProcess = true;
  if (shardPerProcess) {
    numIndexingVecs /= nProcesses;
    // the first process manages the remaining number of vecs
    if (processRank == 0) {
      numIndexingVecs += remainingIndexingVecs;
    }
  }
  
  size_t devFree = 0;
  size_t devTotal = 0;

  if (processRank == 0) {
    std::stringstream deviceStatusStr;
    deviceStatusStr << "# Available Devices: " << numDevices << std::endl;
    deviceStatusStr << "# Devices: " << totalGpus << std::endl;
    deviceStatusStr << "# Processes: " << nProcesses << std::endl;
    deviceStatusStr << "# Devices per process: " << ngpus << std::endl;
    processPrint(processRank, deviceStatusStr);
    printAllDevicesMemory(0, ngpus);
  }

  assert(ngpus <= numDevices);

  MPI_Barrier(MPI_COMM_WORLD);
  std::stringstream outDev;
  outDev << "(first device ID) " << deviceIdInit;
  processPrint(processRank, outDev);

  getAvailableMemoryPerDevice(devFree, devTotal, deviceIdInit, ngpus, nProcessesPerGpu);

  // Let's ensure the memory isn't change
  MPI_Barrier(MPI_COMM_WORLD);

  size_t numIndexingVecsPerGpu;
  if (useShards) {
    numIndexingVecsPerGpu = numIndexingVecs / ngpus + numIndexingVecs % ngpus;
  } else {
    numIndexingVecsPerGpu = numIndexingVecs;
  }

  faiss::gpu::IndicesOptions indiceOptions = faiss::gpu::INDICES_32_BIT;

  size_t devFreeLimit = std::min(devFree, safeMemMargin);

  size_t imiStructureMemSize = calcImiStructureMemSize(
      d, coarseCodebookSize, numSubQuantizers, nbitsSubQuantizer);
  
  auto allocSizePerTypeMap = faiss::gpu::GpuIndexIMIPQv2::getInvListsAllocSizePerTypeInfo(
        numIndexingVecs, numSubQuantizers, nbitsSubQuantizer, false, indiceOptions);
  auto allocSizePerTypeMapPerGpu = faiss::gpu::GpuIndexIMIPQv2::getInvListsAllocSizePerTypeInfo(
        numIndexingVecsPerGpu, numSubQuantizers, nbitsSubQuantizer, false, indiceOptions);

  size_t fixedMemSize = calcFixedMemSize(allocSizePerTypeMap);
  size_t fixedMemSizePerGpu = calcFixedMemSize(allocSizePerTypeMapPerGpu);

  assert(devFreeLimit > fixedMemSize + imiStructureMemSize);
  assert(devFreeLimit > fixedMemSizePerGpu + imiStructureMemSize);

  size_t tempMemory = roundMemAllocDown(devFreeLimit - fixedMemSize - imiStructureMemSize);
  size_t tempMemoryPerGpu = roundMemAllocDown(devFreeLimit - fixedMemSizePerGpu - imiStructureMemSize);

  std::stringstream memoryInfoStr;
  memoryInfoStr << std::endl;
  memoryInfoStr << "tempMemoryPerGpu: " << tempMemoryPerGpu << std::endl;
  memoryInfoStr << "fixedMemSize: " << fixedMemSize << std::endl;
  memoryInfoStr << "fixedMemSizePerGpu: " << fixedMemSizePerGpu << std::endl;
  memoryInfoStr << "imiStructureMemSize: " << imiStructureMemSize << std::endl;
  memoryInfoStr << "devFree: " << devFree << std::endl;
  memoryInfoStr << "safeMemMargin: " << safeMemMargin << std::endl;
  memoryInfoStr << "devFreeLimit: " << devFreeLimit << std::endl;
  memoryInfoStr << "tempMemory: " << tempMemory << std::endl;
  processPrint(processRank, memoryInfoStr);

  // set maximum available memory for tiling over vectors while adding them to the GPU
  size_t maxAddTileSize = (size_t)8 * 1024 * 1024 * 1024;
  size_t numVecsTile = maxAddTileSize / (d * sizeof(float));
  numVecsTile = std::min(numVecsTile, numIndexingVecs);
  numVecsTile = std::min(numVecsTile, (size_t)10000);
  numVecsTile = std::max(numVecsTile, (size_t)1);

  faiss::gpu::GpuIndexIMIPQConfig config;

  config.memorySpace = faiss::gpu::MemorySpace::Fixed;
  // config.multiIndexConfig.memorySpace = faiss::gpu::MemorySpace::Fixed;
  config.indicesOptions = indiceOptions;
  config.usePrecomputedTables = true;

  if (pinnedMemoryMode == 2) {
    config.forcePinnedMemory = true;
  }

  std::vector<faiss::gpu::GpuResourcesProvider *> resVector;
  std::vector<int> devs;
  std::unique_ptr<faiss::Index> indexMultiGpu;
  

  { // Build Index
    bool fileNamIndexIsEmpty = fileNameIndex.empty();

    int storedRank = processRank;
    if (!shardPerProcess) {
      storedRank = 0;
    }
    std::stringstream finaNameIndexPostfix;
    finaNameIndexPostfix << "_rank" << storedRank << "_numVecs" << numIndexingVecs;
    fileNameIndex.append(finaNameIndexPostfix.str());

    std::stringstream loadIndexStart;
    loadIndexStart << "Index - loading: " << fileNameIndex;
    processPrint(processRank, loadIndexStart);
    std::unique_ptr<faiss::Index> indexCpu(loadIndexToCpu<faiss::IndexIVFPQ>(processRank, fileNameIndex));
    if (!indexCpu) {
      { // indexing
        faiss::gpu::StandardGpuResources res(allocSizePerTypeMap);
        res.setLogMemoryAllocations(allocLogging);
        res.setTempMemory(tempMemory);
        std::unique_ptr<faiss::gpu::GpuIndexIMIPQv2> imipqGpu(
          new faiss::gpu::GpuIndexIMIPQv2(&res, d, coarseCodebookSize, numSubQuantizers, nbitsSubQuantizer, config));
        imipqGpu->verbose = verbose;

        size_t indexToAddOffset = 0;

        if (processRank == 0) {
          std::stringstream outTrainStart, outEnd;
          outTrainStart << "Coarse quantizer - loading: " << fileNameCoarseQuantizer;
          processPrint(processRank, outTrainStart);
          // train or load the coarse quantizer
          buildCoarseQuantizer(processRank, imipqGpu.get(), fileNameCoarseQuantizer, isVecFloat, fileNameTraining, numTrainingVecs, d, randomContext);
          outEnd << "Coarse quantizer - loaded: " << fileNameCoarseQuantizer;
          processPrint(processRank, outEnd);
          // Just ensure the seed is correct in case the coarse quantizer was loaded
          randomContext.seed = numTrainingVecs;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (processRank != 0) {
          // train or load the coarse quantizer
          std::stringstream readStart, readEnd;
          readStart << "Coarse quantizer - reading: " << fileNameCoarseQuantizer;
          processPrint(processRank, readStart);

          std::unique_ptr<faiss::IndexIVFPQ> indexCpuTrainedOnly(loadIndexToCpu<faiss::IndexIVFPQ>(processRank, fileNameCoarseQuantizer));
          assert(indexCpuTrainedOnly);
          imipqGpu->copyFrom(indexCpuTrainedOnly.get());
          readEnd << "Coarse quantizer - loaded";
          processPrint(processRank, readEnd);

          // Let's ensure every vector if different for everyProccess process
          if (shardPerProcess) {
            indexToAddOffset = remainingIndexingVecs + numIndexingVecs * processRank;
          } else {
            indexToAddOffset = 0;
          }
          randomContext.seed = numTrainingVecs + indexToAddOffset;
        }

        if (processRank == 0) {
          printAllDevicesMemory(0, ngpus);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        std::stringstream addToIndexStart;
        addToIndexStart << "Adding " << numIndexingVecs << " vectors from " << totalNumIndexingVecs 
                        << " to index with " << indexToAddOffset << " offset";
        processPrint(processRank, addToIndexStart);
        
        int64_t initSeed = 0;
        int64_t endSeed = 0;
        
        // save current initial seed for using it again while adding the vectors
        initSeed = randomContext.seed;

        reserveIndexingSpace(processRank, imipqGpu.get(), isVecFloat, fileNameIndexing, numIndexingVecs, d, numVecsTile, randomContext, indexToAddOffset);

        // save it for assertion
        endSeed = randomContext.seed;

        if (processRank == 0) {
          printAllDevicesMemory(0, ngpus);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        randomContext.seed = initSeed;

        addToIndex(processRank, imipqGpu.get(), isVecFloat, fileNameIndexing, numIndexingVecs, d, numVecsTile, randomContext, indexToAddOffset);

        assert(randomContext.seed == endSeed);

        if (processRank == 0) {
          printAllDevicesMemory(0, ngpus);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        indexCpu.reset(faiss::gpu::index_gpu_to_cpu(imipqGpu.get()));
        faiss::gpu::CudaEvent cloneEnd(imipqGpu->getResources()->getDefaultStreamCurrentDevice());
        cloneEnd.cpuWaitOnEvent();
      }
      
      if (!fileNamIndexIsEmpty) {
        std::stringstream writeIndexStart;
        writeIndexStart << "writing: " << fileNameIndex << "...";
        processPrint(processRank, writeIndexStart);
        faiss::write_index(indexCpu.get(), fileNameIndex.c_str());
        processPrint(processRank, "done");
      }
    }

    std::stringstream loadIndexEnd;
    loadIndexEnd << "Index - built: " << fileNameIndex;
    MPI_Barrier(MPI_COMM_WORLD);

    if (profile) {
      clock_t tStart, tEnd;
      double tGpu;

      faiss::gpu::GpuMultipleClonerOptions options;
      options.memorySpace = config.memorySpace;
      options.indicesOptions = config.indicesOptions;
      options.usePrecomputed = config.usePrecomputedTables;
      options.precomputeCodesOnCpu = config.precomputeCodesOnCpu;
      options.forcePinnedMemory = config.forcePinnedMemory;
      options.shard = useShards;
      options.shard_type = 1;
      options.verbose = verbose;
      
      processPrint(processRank, "Ininting resource for multiple GPUs");
      initResourcesMultiGpu(ngpus, allocSizePerTypeMapPerGpu, tempMemoryPerGpu,
                            resVector, devs, allocLogging, pinnedMemoryMode);

      processPrint(processRank, "Moving index from cpu to multiple GPUs: ");
      tStart = clock();
      indexMultiGpu.reset(faiss::gpu::index_cpu_to_gpu_multiple(resVector, devs, indexCpu.get(), &options));
      faiss::gpu::synchronizeAllDevices();
      tEnd = clock();
      tGpu = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
      std::stringstream indexMovedOut;
      indexMovedOut << "Index moved in " << tGpu;
      processPrint(processRank, indexMovedOut);
    }      
                
  }

  if (profile) {
    if (processRank == 0) {
      printAllDevicesMemory(0, ngpus);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    indexMultiGpu->verbose = verbose;
    
    std::vector<int> numQueriesList = {
        1,      1000,   8192,   10000,   16384,   32768,   65536,   100000,
        131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216};
    numQueriesEnd = std::min(numQueriesEnd, (int)numQueriesList.size());
    std::vector<int> nprobeList = {
        1,    2,    4,    8,    16,   32,   64,   128,  256,   512,   1024,
        2048, 2194, 2352, 2521, 2702, 2896, 4096, 8192, 16384, 32768, 65536};

    std::unique_ptr<float> queries(vecs_load(isVecFloat, fileNameQueries, (size_t)numQueriesList[numQueriesEnd - 1], d, randomContext, queriesOffset));

    std::unique_ptr<int> groundTruth;
    int dRead;

    if (!fileNameGroundTruth.empty()) {
      groundTruth.reset(faiss::ivecs_read(fileNameGroundTruth.c_str(), numQueriesList[numQueriesEnd - 1], 0, &dRead));
    }

    for (int i = numQueriesBegin > 0 ? numQueriesBegin : 0; i < numQueriesEnd;
         i++) {
      int numQueries = numQueriesList[i];
      
      std::stringstream numQueriesOut;
      numQueriesOut << "numOfQueries: " << numQueries << " ===============";
      processPrint(processRank, numQueriesOut);
      
      try {
        for (int j = nprobeBegin > 0 ? nprobeBegin : 0;
             j < nprobeEnd && j < nprobeList.size(); j++) {
          int nprobe = nprobeList[j];
          std::stringstream numProbeOut;
          numProbeOut << "nprobe: " << nprobe << "---------";
          processPrint(processRank, numProbeOut);

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
                std::stringstream searchInfoOut;
                searchInfoOut << "Gpu: " << imipqGpu->getDevice()
                          << ", maxListLength: " << imipqGpu->getMaxListLength()
                          << ", nlist: " << imipqGpu->nlist
                          << ", ntotal: " << imipqGpu->ntotal;
                processPrint(processRank, searchInfoOut);
              }
            } else {
              // single GPU
              faiss::gpu::GpuIndexIMIPQv2 *imipqGpu =
                  dynamic_cast<faiss::gpu::GpuIndexIMIPQv2 *>(indexMultiGpu.get());
              imipqGpu->setNumProbes(nprobe);
              imipqGpu->verbose = verbose;
              std::stringstream searchInfoOut;
              searchInfoOut << "Gpu: " << imipqGpu->getDevice()
                        << ", maxListLength: " << imipqGpu->getMaxListLength()
                        << ", nlist: " << imipqGpu->nlist
                        << ", ntotal: " << imipqGpu->ntotal;
              processPrint(processRank, searchInfoOut);
            }

            search(processRank, nProcesses, 1, numIndexingVecs, remainingIndexingVecs,
                   indexMultiGpu.get(), queries.get(), groundTruth.get(), numQueries, kBegin,
                   kEnd, dRead, nRuns);

          } catch (...) {
            MPI_Barrier(MPI_COMM_WORLD);
            processPrint(processRank, "NPROBE UNKNOWN EXCEPTION");
            if (j == 0 || j == nprobeBegin) {
              throw;
            }
          }
        }
      } catch (...) {
        MPI_Barrier(MPI_COMM_WORLD);
        processPrint(processRank, "QUERY UNKNOWN EXCEPTION");
      }
    }
    
    for (int i = 0; i < resVector.size(); i++) {
      delete resVector[i];
    }
  }
}

int main(int argc, char **argv) {
  if (argc <= 18) {
    std::cout << "There must be 18 or more parameters" << std::endl;
    return 1;
  }

  int d, coarseCodebookSize, numSubQuantizers, nbitsSubQuantizer, queriesOffset,
      numQueriesBegin, numQueriesEnd, kBegin, kEnd, nprobeBegin, nprobeEnd,
      isFloat, numThreads, ngpus, useShards, sharedGpuProcess, profile, allocLogging, verbose, nRuns, pinnedMemoryMode;
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
  sharedGpuProcess = argc > 22 ? std::stoi(argv[22]) : 0;
  safeMemMargin = argc > 23 ? std::stoul(argv[23]) : 0;
  fileNameCoarseQuantizer = argc > 24 ? argv[24] : "";
  fileNameIndex = argc > 25 ? argv[25] : "";
  profile = argc > 26 ? std::stoi(argv[26]) : 1;
  allocLogging = argc > 27 ? std::stoi(argv[27]) : 0;
  verbose = argc > 28 ? std::stoi(argv[28]) : 0;
  nRuns = argc > 29 ? std::stoi(argv[29]) : 5;
  pinnedMemoryMode = argc > 30 ? std::stoi(argv[30]) : 1;

  int nProcesses, processRank;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);
  MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
  
  std::cout << "numThreads: " << numThreads << std::endl;

  omp_set_num_threads(numThreads);

  std::cout << std::setprecision(6) << std::fixed;

  demo_imipq(isFloat, d, coarseCodebookSize, numSubQuantizers, nbitsSubQuantizer,
              fileNameTraining, numTrainingVecs, fileNameIndexing,
              numIndexingVecs, fileNameQueries, queriesOffset,
              fileNameGroundTruth, numQueriesBegin, numQueriesEnd,
              nprobeBegin, nprobeEnd, kBegin, kEnd, ngpus, useShards, 
              nProcesses, processRank, sharedGpuProcess,
              safeMemMargin, fileNameCoarseQuantizer, fileNameIndex,
              profile, allocLogging, verbose, nRuns, pinnedMemoryMode);

  MPI_Barrier(MPI_COMM_WORLD);
  return 0;
}
