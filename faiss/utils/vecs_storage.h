#pragma once
#include <sys/types.h>

namespace faiss {

float *bvecs_read(const char *fileName, size_t num, size_t numOffset, int *dim);

int *ivecs_read(const char *fileName, size_t num, size_t numOffset, int *dim);

float *fvecs_read(const char *fileName, size_t num, size_t numOffset, int *dim);

void bvecs_write(const char *fileName, size_t num, int dim, float *vecs);

void ivecs_write(const char *fileName, size_t num, int dim, int *vecs);

void fvecs_write(const char *fileName, size_t num, int dim, float *vecs);

} // namespace faiss
