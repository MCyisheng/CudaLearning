#include <cuda_runtime.h>
#include <stdio.h>

int main()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("GPU device count: %d\n", deviceCount);

    for (int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        printf("===== Device %d =====\n", dev);
        printf("GPU name: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("SM count: %d\n", prop.multiProcessorCount);
        printf("Warp size: %d\n", prop.warpSize);

        printf("Shared memory per SM: %.2f KB\n", prop.sharedMemPerMultiprocessor / 1024.0);
        printf("Max shared memory per Block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0);

        printf("Max threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("Total global memory: %.2f GB\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
        printf("L2 cache size: %.2f MB\n", prop.l2CacheSize / 1024.0 / 1024.0);
        printf("Total constant memory: %.2f KB\n", prop.totalConstMem / 1024.0);
    }
    return 0;
}
