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
#include <faiss/index_io.h>
#include <faiss/utils/vecs_storage.h>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <string>
#include <sys/types.h>

void search(faiss::Index *index, float *queries, int *groundTruth,
            size_t numQueries, int kBegin, int kEnd, int groundTruthK) {
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
    tEnd = clock();
    tGpu = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
    std::cout << "IMIPQ search time on CPU: " << tGpu << std::endl;

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

template <bool isVecFloat>
void demo_imipq(int d, int nbitsCoarseQuantizer, int numSubQuantizers,
                int nbitsSubQuantizer, std::string fileNameTraining,
                size_t numTrainingVecs, std::string fileNameIndexing,
                size_t numIndexingVecs, std::string fileNameQueries,
                size_t queriesOffset, std::string fileNameGroundTruth,
                int numQueriesBegin, int numQueriesEnd, int nprobeBegin,
                int nprobeEnd, int kBegin, int kEnd,
                std::string fileNameIndex) {
  int coarseCodebookSize = 1 << nbitsCoarseQuantizer;
  constexpr int NUM_COARSE_CODEBOOKS = 2;
  size_t nlist = coarseCodebookSize * coarseCodebookSize;
  faiss::MultiIndexQuantizer multiIndexCpu(d, NUM_COARSE_CODEBOOKS,
                                           nbitsCoarseQuantizer);

  faiss::IndexIVFPQ *imipq;

  clock_t tStart, tEnd;
  double tGpu;
  int dRead;
  bool isLoadead = false;

  if (!fileNameIndex.empty()) {
    FILE *f = fopen(fileNameIndex.c_str(), "rb");
    if (f) {
      fclose(f);
      imipq = dynamic_cast<faiss::IndexIVFPQ *>(
          faiss::read_index(fileNameIndex.c_str()));
      isLoadead = true;
    }
  }

  if (!isLoadead) {
    imipq = new faiss::IndexIVFPQ(&multiIndexCpu, d, nlist, numSubQuantizers,
                                  nbitsSubQuantizer);
    imipq->quantizer_trains_alone = true;

    { // train
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
      imipq->train(numTrainingVecs, trainingVecs);
      tEnd = clock();
      tGpu = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
      std::cout << "IMIPQ train time on GPU: " << tGpu << std::endl;
      delete trainingVecs;
    }

    { // add
      size_t maxAddTileSize = 512 * 1024 * 1024;
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
        imipq->add(currentNumVecsTile, indexingVecs);
        tEnd = clock();
        tGpu = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
        std::cout << "IMIPQ add time on CPU: " << tGpu << std::endl;
        delete indexingVecs;
      }
    }
    imipq->use_precomputed_table = 2;
    imipq->precompute_table();

    if (!fileNameIndex.empty()) {
      faiss::write_index(imipq, fileNameIndex.c_str());
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

  for (int i = numQueriesBegin > 0 ? numQueriesBegin : 0;
       i < numQueriesEnd && i < numQueriesList.size(); i++) {
    int numQueries = numQueriesList[i];
    std::cout << "numOfQueries: " << numQueries
              << " ===============" << std::endl;
    for (int j = nprobeBegin > 0 ? nprobeBegin : 0;
         j < nprobeEnd && j < nprobeList.size(); j++) {
      int nprobe = nprobeList[j];
      std::cout << "nprobe: " << nprobe << "---------" << std::endl;
      imipq->nprobe = nprobe;
      search(imipq, queries, groundTruth, numQueries, kBegin, kEnd, dRead);
    }
  }
  delete queries;
  delete groundTruth;
  delete imipq;
}

int main(int argc, char **argv) {
  if (argc <= 18) {
    std::cout << "There must be 18 or more parameters" << std::endl;
    return 1;
  }

  int d, nbitsCoarseQuantizer, numSubQuantizers, nbitsSubQuantizer,
      queriesOffset, numQueriesBegin, numQueriesEnd, kBegin, kEnd, nprobeBegin,
      nprobeEnd, isFloat, numThreads;
  size_t numTrainingVecs, numIndexingVecs;
  std::string fileNameTraining, fileNameIndexing, fileNameQueries,
      fileNameGroundTruth, fileNameIndex;

  d = std::stoi(argv[1]);
  nbitsCoarseQuantizer = std::stoi(argv[2]);
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
  fileNameIndex = "";

  omp_set_num_threads(numThreads);

  std::cout << std::setprecision(6) << std::fixed;

  if (isFloat == 1) {
    demo_imipq<true>(
        d, nbitsCoarseQuantizer, numSubQuantizers, nbitsSubQuantizer,
        fileNameTraining, numTrainingVecs, fileNameIndexing, numIndexingVecs,
        fileNameQueries, queriesOffset, fileNameGroundTruth, numQueriesBegin,
        numQueriesEnd, nprobeBegin, nprobeEnd, kBegin, kEnd, fileNameIndex);
  } else {
    demo_imipq<false>(
        d, nbitsCoarseQuantizer, numSubQuantizers, nbitsSubQuantizer,
        fileNameTraining, numTrainingVecs, fileNameIndexing, numIndexingVecs,
        fileNameQueries, queriesOffset, fileNameGroundTruth, numQueriesBegin,
        numQueriesEnd, nprobeBegin, nprobeEnd, kBegin, kEnd, fileNameIndex);
  }
  return 0;
}
