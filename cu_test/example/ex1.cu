#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv)
{
	float *device_data = NULL;
	size_t size = 1024*sizeof(float);
	cudaError_t err;
	err = cudaMalloc((void **)&device_data, size);
	printf("err = %d\n",err);
	return 0;
}
