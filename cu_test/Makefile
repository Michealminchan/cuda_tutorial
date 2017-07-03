device_info : device_info.cpp
	nvcc -o $@ $< -lcudart -lcuda -lcublas

clean :
	-rm -rf device_info
