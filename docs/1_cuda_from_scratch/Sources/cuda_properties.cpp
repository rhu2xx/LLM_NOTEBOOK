#include <iostream>
#include <cuda_runtime.h>

static double max_bandwidth(cudaDeviceProp &prop) {

    const std::size_t mem_freq =
        static_cast<std::size_t>(prop.memoryClockRate) * 1000; // kHz -> Hz
    const int bus_width = prop.memoryBusWidth;
    const std::size_t bytes_per_second = 2 * mem_freq * bus_width / CHAR_BIT;
    return static_cast<double>(bytes_per_second) / 1024 / 1024 /
            1024; // B/s -> GB/s
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        std::cout << "Device " << device << ": " << prop.name << std::endl;
        std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Registers per Block: " << prop.regsPerBlock << std::endl;
        std::cout << "  Warp Size: " << prop.warpSize << std::endl;
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Threads Dimension: [" 
                  << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", "
                  << prop.maxThreadsDim[2] << "]" << std::endl;
        std::cout << "  Max Grid Size: [" 
                  << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", "
                  << prop.maxGridSize[2] << "]" << std::endl;
        std::cout << "  Total Constant Memory: " << prop.totalConstMem / 1024 << " KB" << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Max Memory Bandwidth: " << max_bandwidth(prop) << " GB/s" << std::endl;
    }

    return 0;
}
