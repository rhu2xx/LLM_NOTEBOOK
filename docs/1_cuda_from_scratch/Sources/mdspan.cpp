#include <cuda/std/mdspan>
#include <cuda/std/array>
#include <thrust/device_ptr.h>
#include <cstdio>
int main() {
  cuda::std::array<int, 6> sd {0, 1, 2, 3, 4, 5};
  std::printf("type of sd.data(): %s\n", typeid(sd.data()).name()); // type of sd.data(): Pi i.e., pointer to int
  std::printf("type of sd.data(): %s\n", typeid(thrust::raw_pointer_cast(sd.data())).name());

  // cuda::std::mdspan md(sd.data(), 2, 3);
  cuda::std::mdspan md(thrust::raw_pointer_cast(sd.data()), 2, 3);
  std::printf("type of md(): %s\n", typeid(md).name()); 
  std::printf("md(0, 0) = %d\n", md(0, 0)); // 0
  std::printf("md(1, 2) = %d\n", md(1, 2)); // 5

  std::printf("size   = %zu\n", md.size());    // 6
  std::printf("height = %zu\n", md.extent(0)); // 2
  std::printf("width  = %zu\n", md.extent(1)); // 3
}
