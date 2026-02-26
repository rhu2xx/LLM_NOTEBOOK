# RUN CUDA Files

# Compile the CUDA file
Compile the .cu file in Debug mode.
```bash
nvcc -g -G -O0 -o hello hello_world.cu
```

Compile the .cu file in Release mode.
```bash
nvcc -O3 -o hello hello_world.cu
```
# CUDA-gdb

Let's debug this program(hello_world.cu).
```c++
#include <cuda_runtime.h>
#include <iostream>

__global__ void hello_world(){
    printf("block idx: %d, thread idx: %d\n", blockIdx.x, threadIdx.x);
    if (threadIdx.x==0){
        printf("Hello World from block %d!\n", blockIdx.x);
    }
}

int main(){
    printf("Launching kernel with 4 blocks and 8 threads per block.\n");
    hello_world<<<4, 8>>>();
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        return -1;
    }
    else {
        std::cout << "Kernel execution completed successfully." << std::endl;
    }
    std::cout << "Exiting program." << std::endl;
    return 0;
}
```
In command Line
```bash
nvcc -g -G -O0 -o hello hello_world.cu
cuda-gdb ./hello
```
## List the cuda source codes
```bash
(cuda-gdb) list 5
1       #include <cuda_runtime.h>
2       #include <iostream>
3
4       __global__ void hello_world(){
5           printf("block idx: %d, thread idx: %d\n", blockIdx.x, threadIdx.x);
6           if (threadIdx.x==0){
7               printf("Hello World from block %d!\n", blockIdx.x);
8           }
9       }
10
```
## Breakpoints

```bash
(cuda-gdb) break 5
Breakpoint 1 at 0xaee0: file /data/home/huhu/my_packages/CUDA_FROM_SCRATCH/chapter_1/hello_world.cu, line 9.
```
> Even we breakpoint 5-th line, it is still shown at line 9???

## Run the program
Run the program with `run` and reach to the breakpoints
```bash
(cuda-gdb) run
Starting program: /data/home/huhu/my_packages/CUDA_FROM_SCRATCH/chapter_1/hello 
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
Launching kernel with 4 blocks and 8 threads per block.
[New Thread 0x7ffff5bff000 (LWP 2428441)]
[New Thread 0x7fffeffff000 (LWP 2428442)]
[Detaching after fork from child process 2428443]
[New Thread 0x7fffee7dd000 (LWP 2428450)]
[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (0,0,0), device 0, sm 0, warp 0, lane 0]

Thread 1 "hello" hit Breakpoint 1, hello_world<<<(4,1,1),(8,1,1)>>> () at hello_world.cu:5
5           printf("block idx: %d, thread idx: %d\n", blockIdx.x, threadIdx.x);
```
Examine the details of the current CUDA kernel `info cuda kernels`. This kernel was launched with 4 thread blocks, each containing 8 threads.

```bash
(cuda-gdb) info cuda kernels
  Kernel Parent Dev Grid Status                       SMs Mask GridDim BlockDim Invocation    
*      0      -   0    1 Active 0x0000000000000000000000000055 (4,1,1)  (8,1,1) hello_world() 
```

Switch to another thread
```bash
(cuda-gdb) cuda block (0,0,0) thread (6,0,0) 
[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (6,0,0), device 0, sm 0, warp 0, lane 6]
5           printf("block idx: %d, thread idx: %d\n", blockIdx.x, threadIdx.x);
```

Print the current thread info
```bash
(cuda-gdb) print blockIdx.x
$1 = 0
(cuda-gdb) print threadIdx.x
$2 = 6
```

Continue the program
```bash
(cuda-gdb) continue
Continuing.
block idx: 2, thread idx: 0
block idx: 2, thread idx: 1
block idx: 2, thread idx: 2
block idx: 2, thread idx: 3
block idx: 2, thread idx: 4
block idx: 2, thread idx: 5
block idx: 2, thread idx: 6
block idx: 2, thread idx: 7
block idx: 1, thread idx: 0
block idx: 1, thread idx: 1
block idx: 1, thread idx: 2
block idx: 1, thread idx: 3
block idx: 1, thread idx: 4
block idx: 1, thread idx: 5
block idx: 1, thread idx: 6
block idx: 1, thread idx: 7
block idx: 3, thread idx: 0
block idx: 3, thread idx: 1
block idx: 3, thread idx: 2
block idx: 3, thread idx: 3
block idx: 3, thread idx: 4
block idx: 3, thread idx: 5
block idx: 3, thread idx: 6
block idx: 3, thread idx: 7
block idx: 0, thread idx: 0
block idx: 0, thread idx: 1
block idx: 0, thread idx: 2
block idx: 0, thread idx: 3
block idx: 0, thread idx: 4
block idx: 0, thread idx: 5
block idx: 0, thread idx: 6
block idx: 0, thread idx: 7
Hello World from block 3!
Hello World from block 2!
Hello World from block 1!
Hello World from block 0!
Kernel execution completed successfully.
Exiting program.
[Thread 0x7fffeffff000 (LWP 2428442) exited]
[Thread 0x7fffee7dd000 (LWP 2428450) exited]
[Thread 0x7ffff5bff000 (LWP 2428441) exited]
[Inferior 1 (process 2428436) exited normally]
```



















