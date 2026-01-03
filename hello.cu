#include <iostream>
#include <cuda_runtime.h>

__global__ void helloFromGPU() {
    printf("Hello World from GPU thread %d!\n", threadIdx.x);
}

int main() {
    printf("Hello World from CPU!\n");

    helloFromGPU<<<1, 128>>>();
    
    // Wait for GPU to finish before exiting
    cudaDeviceSynchronize();

    return 0;
}