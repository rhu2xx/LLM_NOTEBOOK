#include <cstdio>
#include <chrono>

#include <thrust/tabulate.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/universal_vector.h>

thrust::universal_vector<float> row_temperatures(
    int height, int width,
    thrust::universal_vector<float>& temp)
{
    auto row_ids_begin = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        [width]__host__ __device__ (int idx){
        return idx/width;
    });
    auto row_ids_end = row_ids_begin + temp.size();
    
    thrust::universal_vector<float> sums(height);
    thrust::reduce_by_key(
        thrust::device,
        row_ids_begin, row_ids_end,   // input keys
        temp.begin(),                     // input values
        thrust::make_discard_iterator(),  // output keys
        sums.begin());                    // output values

    return sums;
}

thrust::universal_vector<float> init(int height, int width) {
    const float low = 15.0;
    const float high = 90.0;
    thrust::universal_vector<float> temp(height * width, low);
    thrust::fill(thrust::device, temp.begin(), temp.begin() + width, high);
    return temp;
}

int main()
{
    int height = 16;
    int width = 16777216;
    thrust::universal_vector<float> temp = init(height, width);

    auto begin = std::chrono::high_resolution_clock::now();
    thrust::universal_vector<float> sums = row_temperatures(height, width, temp);
    auto end = std::chrono::high_resolution_clock::now();
    const double seconds = std::chrono::duration<double>(end - begin).count();
    const double gigabytes = static_cast<double>(temp.size() * sizeof(float)) / 1024 / 1024 / 1024;
    const double throughput = gigabytes / seconds;

    std::printf("computed in %g s\n", seconds);
    std::printf("achieved throughput: %g GB/s\n", throughput);
}
