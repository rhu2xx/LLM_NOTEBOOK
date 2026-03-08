#include <thrust/device_vector.h>
#include <cuda/std/mdspan>


void simulate(int width, int height, const thrust::device_vector<float> &in, thrust::device_vector<float> &out) {
    cuda::std::mdspan temp_in(thrust::raw_pointer_cast(in.data()), height, width);

    cub::DeviceTransform::Transform( 
        thrust::make_counting_iterator(0),
        out.begin(),
        width * height,
        [=] __device__ __host__ (int id){
            int row = id / width;
            int col = id % width;

            if (row > 0 && column > 0 && row < height - 1 && column < width - 1) {
                float d2tdx2 =
                    temp(row, column - 1) - 2 * temp(row, column) + temp(row, column + 1);
                float d2tdy2 =
                    temp(row - 1, column) - 2 * temp(row, column) + temp(row + 1, column);

                return temp(row, column) + 0.2f * (d2tdx2 + d2tdy2);
            } else {
                return temp(row, column);
            }
        }
    );
}
thrust::device_vector<float> init(int width, int height) {
    
}

int main(){
    int height = 2048;
    int width = 8192;
    thrust::device_vector<float> d_prev = 
}