#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this lab
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    
   float Cvalue = 0;
   //Loop over the A and B tiles required to compute the P element
   for(int p = 0; p < ceil(((float)numAColumns)/TILE_WIDTH) ; p++)
   {
       // Collaborative loading of A and B tiles into shared memory
       if((row < numARows)&&(p*TILE_WIDTH + threadIdx.x < numAColumns))
       {
          ds_A[ty][tx] = A[row * numAColumns + p * TILE_WIDTH + threadIdx.x];
       }else
       {
           ds_A[ty][tx] = 0;
       }
       if((p*TILE_WIDTH + threadIdx.y < numBRows)&&(col < numBColumns))
       {
         ds_B[ty][tx] = B[(p * TILE_WIDTH + threadIdx.y)*numBColumns + col];
       }else
       {
           ds_B[ty][tx] = 0;
       }
       __syncthreads();
     
      if((row < numCRows) && (col < numCColumns))
      {
       for(int i = 0; i < TILE_WIDTH; i++)
           Cvalue += ds_A[ty][i] * ds_B[i][tx];
      } 
       __syncthreads();
   }
   if((row < numCRows)&&(col < numCColumns))
   {
     C[row * numCColumns + col] = Cvalue;
   }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows    = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  int sizeA = numARows * numAColumns * sizeof(float);
  cudaMalloc((void **)&deviceA, sizeA);
  int sizeB = numBRows * numBColumns * sizeof(float);
  cudaMalloc((void **)&deviceB, sizeB);
  int sizeC = numCRows * numCColumns * sizeof(float);
  cudaMalloc((void **)&deviceC, sizeC);

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 blockDim(16, 16);
  dim3 gridDim(ceil(((float)numCColumns)/blockDim.x), 
               ceil(((float)numCRows)/blockDim.y));

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC,
          numARows,numAColumns,numBRows,numBColumns,numCRows,numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
