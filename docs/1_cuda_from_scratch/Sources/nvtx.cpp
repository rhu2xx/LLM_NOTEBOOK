
#include <fstream>
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cuda/std/mdspan>
#include <nvtx3.hpp>
#include <cub/device/device_transform.cuh>


void simulate(int width, int height, const thrust::device_vector<float> &in, thrust::device_vector<float> &out) {
    cuda::std::mdspan temp_in(thrust::raw_pointer_cast(in.data()), height, width);

    cub::DeviceTransform::Transform( 
        thrust::make_counting_iterator(0),
        out.begin(),
        width * height,
        [=] __device__ __host__ (int id){
            int row = id / width;
            int column = id % width;

            if (row > 0 && column > 0 && row < height - 1 && column < width - 1) {
                float d2tdx2 =
                    temp_in(row, column - 1) - 2 * temp_in(row, column) + temp_in(row, column + 1);
                float d2tdy2 =
                    temp_in(row - 1, column) - 2 * temp_in(row, column) + temp_in(row + 1, column);

                return temp_in(row, column) + 0.2f * (d2tdx2 + d2tdy2);
            } else {
                return temp_in(row, column);
            }
        }
    );
}

thrust::device_vector<float> init(int width, int height) {
    thrust::device_vector<float> d_prev(width * height, 15.0f);
    thrust::fill_n(d_prev.begin(), width, 90.0f);
    thrust::fill_n(d_prev.begin() + width * (height - 1), width, 90.0f);
    return d_prev;
    
}

template <class ContainerT>
void store(int step, int height, int width, ContainerT &data)
{
    std::ofstream file("/tmp/heat_" + std::to_string(step) + ".bin", std::ios::binary);
    file.write(reinterpret_cast<const char*>(&height), sizeof(int));
    file.write(reinterpret_cast<const char*>(&width), sizeof(int));
    file.write(reinterpret_cast<const char *>(thrust::raw_pointer_cast(data.data())), height * width * sizeof(float));
}

int main(){
    int height = 2048;
    int width = 8192;
    thrust::device_vector<float> d_prev = init(width, height);
    thrust::device_vector<float> d_next(height * width);
    thrust::host_vector<float> h_prev(height * width);
    const int compute_steps = 750;
    const int write_steps = 3;
    for (int write_step = 0; write_step < write_steps; write_step++)
    {
        nvtx3::scoped_range r{std::string("write step ") + std::to_string(write_step)};

        {
            nvtx3::scoped_range r{"copy"};
            thrust::copy(d_prev.begin(), d_prev.end(), h_prev.begin());
        }

        {
            nvtx3::scoped_range r{"compute"};
            for (int compute_step = 0; compute_step < compute_steps; compute_step++)
            {
                // Even though the simulate function is called asynchronously, `swap` will not fall into the data race. Because eveything is serialized in a single default stream.
                simulate(width, height, d_prev, d_next);
                d_prev.swap(d_next);
            }
        }

        {
            nvtx3::scoped_range r{"write"};
            store(write_step, height, width, h_prev);
        }

        {
            nvtx3::scoped_range r{"wait"};
            cudaDeviceSynchronize();
        }
    }
}

