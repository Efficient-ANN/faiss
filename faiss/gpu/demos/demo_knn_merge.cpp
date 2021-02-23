/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <faiss/utils/vecs_storage.h>
#include <iostream>
#include <omp.h>
#include <string>

void merge(int k, float *firstDistances, int *firstLabels,
           float *secondDistances, int *secondLabels, float *mergedDistances,
           int *mergedLabels, int labelsOffset) {
  int i = 0;
  int j = 0;
  for (int currentK = 0; currentK < k; currentK++) {
    float distance1 = firstDistances[i];
    float distance2 = secondDistances[j];
    if (distance1 <= distance2) {
      mergedDistances[currentK] = distance1;
      mergedLabels[currentK] = firstLabels[i];
      i++;
    } else {
      mergedDistances[currentK] = distance2;
      mergedLabels[currentK] = secondLabels[j] + labelsOffset;
      j++;
    }
  }
}

void mergeKnn(int k, int kMax, int numVecs, int begin, int end,
              std::string inputFilePrefixDistances,
              std::string inputFilePrefixLabels,
              std::string outputFilePrefixDistances,
              std::string outputFilePrefixLabels, const int batchSizeM) {
  if (begin < 0) {
    return;
  }

  const int batchSize = batchSizeM * 1000000;

  std::string inFileNameDistances, inFileNameLabels, outFileName;
  int kRead;
  float *firstDistances = nullptr;
  int *firstLabels = nullptr;

  inFileNameDistances = inputFilePrefixDistances + std::to_string(0) + ".fvecs";
  inFileNameLabels = inputFilePrefixLabels + std::to_string(0) + ".ivecs";

  firstDistances =
      faiss::fvecs_read(inFileNameDistances.c_str(), numVecs, 0, &kRead);
  assert(kMax == kRead);

  firstLabels = faiss::ivecs_read(inFileNameLabels.c_str(), numVecs, 0, &kRead);
  assert(kMax == kRead);

  outFileName =
      outputFilePrefixDistances + std::to_string(batchSizeM) + "M.fvecs";
  faiss::fvecs_write(outFileName.c_str(), numVecs, k, firstDistances);

  outFileName = outputFilePrefixLabels + std::to_string(batchSizeM) + "M.ivecs";
  faiss::ivecs_write(outFileName.c_str(), numVecs, k, firstLabels);

  for (int i = 1; i < end; i++) {
    float *secondDistances = nullptr;
    int *secondLabels = nullptr;
    float *mergedDistances = nullptr;
    int *mergedLabels = nullptr;

    inFileNameDistances =
        inputFilePrefixDistances + std::to_string(i) + ".fvecs";
    inFileNameLabels = inputFilePrefixLabels + std::to_string(i) + ".ivecs";
    secondDistances =
        faiss::fvecs_read(inFileNameDistances.c_str(), numVecs, 0, &kRead);
    secondLabels =
        faiss::ivecs_read(inFileNameLabels.c_str(), numVecs, 0, &kRead);

    mergedDistances = new float[k * numVecs];
    mergedLabels = new int[k * numVecs];

    int labelsOffset = batchSize * i;
#pragma omp for
    for (int j = 0; j < numVecs; j++) {
      merge(k, firstDistances + j * k, firstLabels + j * k,
            secondDistances + j * k, secondLabels + j * k,
            mergedDistances + j * k, mergedLabels + j * k, labelsOffset);
    }

    delete[] firstDistances;
    delete[] firstLabels;
    delete[] secondDistances;
    delete[] secondLabels;

    firstDistances = mergedDistances;
    firstLabels = mergedLabels;

    if (i >= begin) {
      outFileName = outputFilePrefixDistances +
                    std::to_string(batchSizeM * (i + 1)) + "M.fvecs";
      faiss::fvecs_write(outFileName.c_str(), numVecs, k, firstDistances);

      outFileName = outputFilePrefixLabels +
                    std::to_string(batchSizeM * (i + 1)) + "M.ivecs";
      faiss::ivecs_write(outFileName.c_str(), numVecs, k, firstLabels);
    }
  }

  delete[] firstDistances;
  delete[] firstLabels;
}

int main(int argc, char **argv) {

  if (argc <= 10) {
    std::cout << "There must be 10 or more parameters" << std::endl;
    return 1;
  }

  int k, kMax, numVecs, begin, end, numThreads, batchSizeM;
  std::string inputFilePrefixDistances, inputFilePrefixLabels,
      outputFilePrefixDistances, outputFilePrefixLabels;

  k = std::stoi(argv[1]);
  kMax = std::stoi(argv[2]);
  numVecs = std::stoi(argv[3]);
  begin = std::stoi(argv[4]);
  end = std::stoi(argv[5]);
  inputFilePrefixDistances = argv[6];
  inputFilePrefixLabels = argv[7];
  outputFilePrefixDistances = argv[8];
  outputFilePrefixLabels = argv[9];
  batchSizeM = std::stoi(argv[10]);
  numThreads = argc > 11 ? std::stoi(argv[11]) : 1;

  omp_set_num_threads(numThreads);

  mergeKnn(k, kMax, numVecs, begin, end, inputFilePrefixDistances,
           inputFilePrefixLabels, outputFilePrefixDistances,
           outputFilePrefixLabels, batchSizeM);

  return 0;
}
