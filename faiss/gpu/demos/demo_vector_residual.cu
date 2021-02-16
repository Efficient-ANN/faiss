/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <cstdlib>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/impl/VectorResidual.cuh>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <faiss/gpu/utils/StaticUtils.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

void fillWithRandom(std::mt19937 &rng,
                    std::uniform_real_distribution<> &distrib, float *array,
                    int size) {
  for (int i = 0; i < size; i++) {
    array[i] = distrib(rng) * 1000.;
  }
}

void demoVectorResidual(int numOfQueries, int d, int multiIndexCodebookSize) {
  FAISS_ASSERT(d > 0 && d % 2 == 0);
  FAISS_ASSERT(multiIndexCodebookSize > 0);

  std::mt19937 rng;
  std::uniform_real_distribution<> distrib;

  constexpr int NUM_CODEBOOKS = 2;
  faiss::gpu::StandardGpuResources provider;

  // provider.getResources()->noTempMemory();

  int device = 0;
  cudaStream_t stream =
      provider.getResources()->getDefaultStreamCurrentDevice();
  faiss::gpu::DeviceTensor<float, 2, true> outResiduals(
      provider.getResources().get(),
      faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream),
      {numOfQueries, d});
  std::vector<float> residuals(numOfQueries * d);
  std::vector<float> queries(numOfQueries * d);

  fillWithRandom(rng, distrib, queries.data(), queries.size());

  std::chrono::steady_clock::time_point start, end;
  std::chrono::duration<double> duration;

  std::cout << std::setprecision(6) << std::fixed;
  { // computing residual flat-index

    auto inQueries = faiss::gpu::toDeviceTemporary<float, 2>(
        provider.getResources().get(), device,
        const_cast<float *>(queries.data()), stream, {numOfQueries, d});
    int flatIndexCodebookSize = multiIndexCodebookSize * multiIndexCodebookSize;
    std::vector<float> centroidsFlat(flatIndexCodebookSize * d);

    fillWithRandom(rng, distrib, centroidsFlat.data(), centroidsFlat.size());

    auto inCentroidsFlat = faiss::gpu::toDeviceTemporary<float, 2>(
        provider.getResources().get(), device,
        const_cast<float *>(centroidsFlat.data()), stream,
        {flatIndexCodebookSize, d});
    std::vector<int> keys(numOfQueries);

    for (int i = 0; i < keys.size(); i++) {
      keys[i] = rand() % flatIndexCodebookSize;
    }

    auto inKeys = faiss::gpu::toDeviceTemporary<int, 1>(
        provider.getResources().get(), device, const_cast<int *>(keys.data()),
        stream, {(int)keys.size()});

    start = std::chrono::steady_clock::now();
    faiss::gpu::runCalcResidual(inQueries, inCentroidsFlat, inKeys,
                                outResiduals, stream);
    faiss::gpu::fromDevice<float, 2>(outResiduals, residuals.data(), stream);

    faiss::gpu::CudaEvent copyEnd(stream);

    copyEnd.cpuWaitOnEvent();
    end = std::chrono::steady_clock::now();
    duration =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Time on flat-index: " << duration.count() << std::endl;
  }

  { // computing residual multi-index

    auto inQueries = faiss::gpu::toDeviceTemporary<float, 2>(
        provider.getResources().get(), device,
        const_cast<float *>(queries.data()), stream,
        {NUM_CODEBOOKS * numOfQueries, d / NUM_CODEBOOKS});
    std::vector<float> centroidsMulti(NUM_CODEBOOKS * multiIndexCodebookSize *
                                      d / NUM_CODEBOOKS);

    fillWithRandom(rng, distrib, centroidsMulti.data(), centroidsMulti.size());

    auto inCentroidsMulti = faiss::gpu::toDeviceTemporary<float, 2>(
        provider.getResources().get(), device,
        const_cast<float *>(centroidsMulti.data()), stream,
        {NUM_CODEBOOKS * multiIndexCodebookSize, d / NUM_CODEBOOKS});

    std::vector<std::pair<ushort, ushort>> keyPairs(numOfQueries);

    for (int i = 0; i < keyPairs.size(); i++) {
      keyPairs[i].first = rand() % multiIndexCodebookSize;
      keyPairs[i].second = rand() % multiIndexCodebookSize;
    }

    auto inKeyPairs = faiss::gpu::toDeviceTemporary<ushort2, 1>(
        provider.getResources().get(), device, (ushort2 *)(keyPairs.data()),
        stream, {(int)keyPairs.size()});

    start = std::chrono::steady_clock::now();
    faiss::gpu::runCalcResidual(inQueries, inCentroidsMulti, inKeyPairs,
                                outResiduals, stream);
    faiss::gpu::fromDevice<float, 2>(outResiduals, residuals.data(), stream);

    faiss::gpu::CudaEvent copyEnd(stream);

    copyEnd.cpuWaitOnEvent();
    end = std::chrono::steady_clock::now();
    duration =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Time on multi-index: " << duration.count() << std::endl;
  }
}

int main(int argc, char **argv) {
  int numOfQueries, d, multiIndexCodebookSize;

  numOfQueries = argc > 1 ? std::stoi(argv[1]) : 1024;
  d = argc > 2 ? std::stoi(argv[2]) : 64;
  multiIndexCodebookSize = argc > 3 ? std::stoi(argv[3]) : 2048;
  demoVectorResidual(numOfQueries, d, multiIndexCodebookSize);
  return 0;
}
