#include "cuda_utils.h"
#include "linalg/vec.h"
#include "shell/data.h"

template <typename T>
__global__ void testKernel(DeviceVec<T> x) {
  // T *data = x.getPtr();
  // for (int i = 0; i < x.getSize(); i++) {
  //     printf("x[%d] = %.8e\n", i, data[i]);
  // }
  printf("here\n");
  T *data = x.getPtr();
  for (int i = 0; i < 5; i++) {
  }
  printf("x[0] = %.8e\n", data[0].E);
}

int main() {
  // using T = double;
  // DeviceVec<T> x(5);

  using T = ShellIsotropicData<double, false>;
  HostVec<T> x(5, T(1, 1, 1));
  auto d_x = x.createDeviceVec(false);

  testKernel<T><<<1, 1>>>(d_x);
  cudaDeviceSynchronize();

  return 0;
};