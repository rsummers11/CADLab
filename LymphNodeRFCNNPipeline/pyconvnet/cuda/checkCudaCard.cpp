#include "cuda_runtime.h"
#include "cuda.h"

int main()
{
    int deviceCount=0;
    cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);

    // Error when running cudaGetDeviceCount
    if(cudaResultCode != cudaSuccess) // cudaSuccess=0
        return EXIT_FAILURE;

    // Returns an error if no cuda card has been detected
    if(deviceCount==0)
        return EXIT_FAILURE;

    // Returns success if the code ran fine and at least 1 card
    // has been detected
    return EXIT_SUCCESS;
}
