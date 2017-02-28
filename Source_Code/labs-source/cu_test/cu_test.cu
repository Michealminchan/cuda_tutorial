#include <cuda.h>
#include <iostream>

__host__ void test() {
	  float a = 12.;
	  double b = 3.;
	  auto c = a * b;
	  std::cout << c << std::endl;
}

int main()
{
  test();
  return 0;
}
