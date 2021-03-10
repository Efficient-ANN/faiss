/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cassert>
#include <faiss/utils/vecs_storage.h>
#include <iostream>
#include <string>
#include <sys/types.h>

int main(int argc, char **argv) {
  if (argc <= 4) {
    std::cout << "params: <fileName1> <fileName2> <numVecs> <type>"
              << std::endl;
    return 1;
  }

  std::string fileName1, fileName2;
  size_t numVecs;
  int type;

  fileName1 = argv[1];
  fileName2 = argv[2];
  numVecs = std::stoul(argv[3]);
  type = std::stoi(argv[4]);

  if (type == 0) {
    int readedDim1, readedDim2;
    float *vecs1, *vecs2;
    vecs1 = faiss::bvecs_read(fileName1.c_str(), numVecs, 0, &readedDim1);
    vecs2 = faiss::bvecs_read(fileName2.c_str(), numVecs, 0, &readedDim2);

    int minDim = std::min(readedDim1, readedDim2);
    for (size_t i = 0; i < numVecs; i++) {
      for (size_t j = 0; j < minDim; j++) {
        assert(vecs1[i * readedDim1 + j] == vecs2[i * readedDim2 + j]);
      }
    }
    delete[] vecs1;
    delete[] vecs2;
  } else if (type == 1) {
    int readedDim1, readedDim2;
    int *vecs1, *vecs2;
    vecs1 = faiss::ivecs_read(fileName1.c_str(), numVecs, 0, &readedDim1);
    vecs2 = faiss::ivecs_read(fileName2.c_str(), numVecs, 0, &readedDim2);

    int minDim = std::min(readedDim1, readedDim2);
    for (size_t i = 0; i < numVecs; i++) {
      for (size_t j = 0; j < minDim; j++) {
        std::cout << vecs1[i * readedDim1 + j] << " "
                  << vecs2[i * readedDim2 + j] << std::endl;
        assert(vecs1[i * readedDim1 + j] == vecs2[i * readedDim2 + j]);
      }
    }
    delete[] vecs1;
    delete[] vecs2;
  } else {
    int readedDim1, readedDim2;
    float *vecs1, *vecs2;
    vecs1 = faiss::fvecs_read(fileName1.c_str(), numVecs, 0, &readedDim1);
    vecs2 = faiss::fvecs_read(fileName2.c_str(), numVecs, 0, &readedDim2);

    int minDim = std::min(readedDim1, readedDim2);
    for (size_t i = 0; i < numVecs; i++) {
      for (size_t j = 0; j < minDim; j++) {
        assert(vecs1[i * readedDim1 + j] == vecs2[i * readedDim2 + j]);
      }
    }
    delete[] vecs1;
    delete[] vecs2;
  }

  return 0;
}
