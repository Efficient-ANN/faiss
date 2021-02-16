#include <cstdlib>
#include <faiss/utils/vecs_storage.h>
#include <iostream>
#include <string>
#include <sys/types.h>

void fillWithRandom(float *array, int size) {
  for (int i = 0; i < size; i++) {
    array[i] = drand48();
  }
}

int main(int argc, char **argv) {
  if (argc <= 4) {
    std::cout << "params: <fileName> <numVecs> <dim> <type>" << std::endl;
    return 1;
  }

  std::string fileName;
  size_t numVecs;
  int dim, type;

  fileName = argv[1];
  numVecs = std::stoul(argv[2]);
  dim = std::stoi(argv[3]);
  type = std::stoi(argv[4]);

  if (type == 0) {
    float *vecsToWrite = new float[numVecs * dim];
    for (size_t i = 0; i < numVecs; i++) {
      for (size_t j = 0; j < dim; j++) {
        vecsToWrite[i * dim + j] = j;
      }
    }
    faiss::bvecs_write(fileName.c_str(), numVecs, dim, vecsToWrite);
    delete vecsToWrite;
  } else if (type == 1) {
    int *vecsToWrite = new int[numVecs * dim];
    for (size_t i = 0; i < numVecs; i++) {
      for (size_t j = 0; j < dim; j++) {
        vecsToWrite[i * dim + j] = j;
      }
    }
    faiss::ivecs_write(fileName.c_str(), numVecs, dim, vecsToWrite);
    delete vecsToWrite;
  } else {
    float *vecsToWrite = new float[numVecs * dim];
    fillWithRandom(vecsToWrite, numVecs * dim);
    faiss::fvecs_write(fileName.c_str(), numVecs, dim, vecsToWrite);
    delete vecsToWrite;
  }

  return 0;
}
