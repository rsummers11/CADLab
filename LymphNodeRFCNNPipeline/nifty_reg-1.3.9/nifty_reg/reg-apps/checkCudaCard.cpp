#include <stdio.h>
#include <cuda_runtime.h>

int main()
{
    int deviceCount, device;
    int gpuDeviceCount = 0;
    struct cudaDeviceProp properties;
    cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
    if (cudaResultCode != cudaSuccess) 
        deviceCount = 0;
    /* machines with no GPUs can still report one emulation device */
    for (device = 0; device < deviceCount; ++device) {
        cudaGetDeviceProperties(&properties, device);
        if (properties.major != 9999) /* 9999 means emulation only */
            ++gpuDeviceCount;
    }
    if (gpuDeviceCount > 0){
    	cudaGetDeviceProperties(&properties, 0);
	    printf("%i.%i", properties.major, properties.minor);
        return 0;
    }
    else return 1;
}
