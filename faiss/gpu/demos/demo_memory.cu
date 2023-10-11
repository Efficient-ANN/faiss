#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

int main(int argc, char const *argv[]) {
    size_t devFree = 0;
    size_t devTotal = 0;
    cudaMemGetInfo(&devFree, &devTotal);
    std::cout << "-------Memory-------" << std::endl;
    std::cout << "Free: " << devFree << std::endl;
    std::cout << "Total: " << devTotal << std::endl;
    return 0;
}
