//对比向量运算的加速效果

#include <cuda_runtime_api.h>
#include <iostream>
#include <chrono>

using namespace std::chrono;


__global__ void addvector(float* A, float*B, float*C, int Length)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < Length)
        C[index] = A[index] + B[index];
}


void initArray(float* A, int length)
{
    std::srand(std::time({}));
    for(int i=0; i<length; i++)
    {
        A[i] = rand() / (float)RAND_MAX;
    }
}

void CpuVecAdd(float* A, float* B, float* C,  int length)
{
    auto start = high_resolution_clock::now();
    for(int i=0; i<length; i++)
    {
        C[i] = A[i] + B[i];
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    std::cout << "CPU time: " << duration.count() << std::endl;
}

bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=0.00001)
{
    
    for(int i=0; i<length; i++)
    {
        if(fabs(A[i] -B[i]) > epsilon)
        {
            printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    std::cout << "correct!" << std::endl;
    return true;
}

void GPUVecAdd(float* A, float* B, float* C, int length)
{
    float* A_I;
    float* B_I;
    float* C_O;

    cudaMallocManaged(&A_I, sizeof(float) * length);
    cudaMallocManaged(&B_I, sizeof(float) * length);
    cudaMallocManaged(&C_O, sizeof(float) * length);

    std::memcpy(A_I, A, sizeof(float) * length);
    std::memcpy(B_I, B, sizeof(float) * length);
    std::memset(C_O, 0, sizeof(float) * length);

    int threads = 256;

    int blocks = (length + threads - 1) / threads;


    addvector<<<blocks, threads>>>(A_I, B_I, C_O, length);
    auto start = high_resolution_clock::now();

    cudaDeviceSynchronize();

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    std::cout << "GPU time: " << duration.count() << std::endl;


    std::memcpy(C, C_O, sizeof(float) * length);

    cudaFree(A_I);
    cudaFree(B_I);
    cudaFree(C_O);
}




int main()
{
    int vectorLength = 50000;

    float* array_1 = (float*)malloc(vectorLength * sizeof(float));
    float* array_2 = (float*)malloc(vectorLength * sizeof(float));

    float* gpu_output = (float*)malloc(vectorLength * sizeof(float));
    float* cpu_output = (float*)malloc(vectorLength * sizeof(float));

    initArray(array_1, vectorLength);
    initArray(array_2, vectorLength);

    GPUVecAdd(array_1, array_2, gpu_output, vectorLength);
    CpuVecAdd(array_1, array_2, cpu_output, vectorLength);

    vectorApproximatelyEqual(gpu_output, cpu_output, vectorLength);
    return 0;
}