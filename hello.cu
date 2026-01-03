#include <iostream>
#include <cuda_runtime.h>

__global__ void Vect(int *A, int N) {
    // Corrected: blockDim (Capital D) and N (Capital N)
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < N) {
        A[index] = index;
    }
}

int main() {
    int N = 4;
    int *A_dev; // Device pointer
    int *B_host; // Host pointer

    // Corrected: malloc (lowercase) and size
    B_host = (int*)malloc(N * sizeof(int));
    
    // Allocate on device
    cudaMalloc(&A_dev, N * sizeof(int));

    // Launch kernel
    Vect<<<1, 4>>>(A_dev, N);

    // Corrected: direction is cudaMemcpyDeviceToHost
    cudaMemcpy(B_host, A_dev, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < N; i++) {
        printf("%d ", B_host[i]);
    }
    printf("\n");

    // Cleanup
    cudaFree(A_dev);
    free(B_host);

    return 0;
}