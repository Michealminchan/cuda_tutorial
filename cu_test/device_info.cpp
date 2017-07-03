/*
*
*      Filename: device_info.cpp
*
*        Author: Haibo Hao
*        Email : haohaibo@ncic.ac.cn
*   Description: ---
*        Create: 2017-07-03 19:07:22
* Last Modified: 2017-07-03 19:07:22
**/
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

CUcontext hContext = 0;
cublasHandle_t hCublas = 0;
cublasHandle_t handle;

int main(int argc, char* argv[])
{
    char deviceName[32];
    int count, ordinal, major, minor;
    CUdevice hDevice;
    CUevent hStart, hStop;

    std::cout << "===========device info test============" << std::endl;
    // Initialize the Driver API and find a device
    cuInit(0);
    cuDeviceGetCount(&count);
    for(ordinal = 0; ordinal < count; ++ordinal)
    {
        cuDeviceGet(&hDevice, ordinal);
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, hDevice);
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, hDevice);
        cuDeviceGetName(deviceName, sizeof(deviceName), hDevice);
        
        printf("Id:%d %s (%d.%d)\n", ordinal, deviceName, major, minor);
    }
}

