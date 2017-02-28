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

//@@ INSERT CODE HERE
#define CHANNELS 3
__global__ void imageColorToGrayscale(float *rgbImageData,
	       	float *grayImageData, int imageChannels, int height, int width){

	int row;
	int col;
	row =  threadIdx.y + blockIdx.y * blockDim.y;
	col =  threadIdx.x + blockIdx.x * blockDim.x;
	
	if(row < height && col < width){
		//get 1D coordinate for the grayscale image
		int grayOffset = row * width + col;
		//one can think of the RGB image having 
		//CHANNLES times colums than the grayscale image
		int rgbOffset = grayOffset * imageChannels;
		float r = rgbImageData[rgbOffset]; //red value for pixel
		float g = rgbImageData[rgbOffset + 1]; //green value for pixel
		float b = rgbImageData[rgbOffset + 2]; // blue value for pixel
		
		//perform the rescaling and store it
		//we multiply by floating point constans
		grayImageData[grayOffset] = 0.21*r + 0.71f*g + 0.07f*b;

	}

}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *deviceInputImageData;
  float *deviceOutputImageData;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  inputImage = wbImport(inputImageFile);

  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  // For this lab the value is always 3
  imageChannels = wbImage_getChannels(inputImage);

  // Since the image is monochromatic, it only contains one channel
  outputImage = wbImage_new(imageWidth, imageHeight, 1);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE
  dim3 DimGrid(ceil(imageWidth/16.0), ceil(imageHeight/16.0), 1);
  dim3 DimBlock(16, 64 , 1);

  imageColorToGrayscale<<<DimGrid, DimBlock>>>(deviceInputImageData,
		  deviceOutputImageData, imageChannels, 
                  imageHeight, imageWidth); 
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float),
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(args, outputImage);

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
