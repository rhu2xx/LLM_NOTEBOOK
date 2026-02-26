## Record Time

```cpp
#include <chrono>

auto begin = std::chrono::high_resolution_clock::now();
****************
auto end = std::chrono::high_resolution_clock::now();
const double seconds = std::chrono::duration<double>(end - begin).count();
```

## Record throughput

```rust
const double gigabytes = static_cast<double>(temp.size() * sizeof(float)) / 1024 / 1024 / 1024;
const double throughput = gigabytes / seconds;
```

## Get the device properties

```rust
static double max_bandwidth() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    const std::size_t mem_freq =
        static_cast<std::size_t>(prop.memoryClockRate) * 1000; // kHz -> Hz
    const int bus_width = prop.memoryBusWidth;
    const std::size_t bytes_per_second = 2 * mem_freq * bus_width / CHAR_BIT;
    return static_cast<double>(bytes_per_second) / 1024 / 1024 /
            1024; // B/s -> GB/s
}
```