#include<stdio.h>

int main(int argc, char** argv)
{
	dim3 Dimblock(1024, 1024, 64);
	printf("blockDim.x =  %d\n",Dimblock.x);
	printf("blockDim.y =  %d\n",Dimblock.y);
	printf("blockDim.z =  %d\n",Dimblock.z);
	return 0;
}
