# Cuda Memory Hierarchy and Thread Management
## The architecture

| Component | Description |
|-----------|-------------|
| LD/ST Unit | Memory load and store operations, used to transfer data between registers and global/share memory. |
| SFU (Special Function Unit) | Execute special mathmatical operations, such as trigonometric functions, square roots, and exponentials. |
| CUDA Cores | The main processing units that execute instructions and perform computations. |
| Warp Scheduler | Manages the scheduling of warps (groups of 32 threads) for execution on the CUDA cores. |
| Tensor Cores | Specialized processing units designed to accelerate matrix operations, particularly for deep learning applications. |
| Dispatch Unit | Responsible for dispatching instructions to the appropriate execution units within the GPU from the warp scheduler. |
| Register File | A small, fast memory space used to store temporary variables and data for each thread during execution. |
| L1 Cache | A small, fast memory cache located close to the CUDA cores, used to store frequently accessed data. |
| Shared Memory | A fast, on-chip memory that is shared among threads within the same block, allowing for efficient data sharing and communication. |
| L2 Cache | A larger, slower memory cache that serves as a bridge between the global memory and the L1 cache, helping to reduce memory access latency. |
| Global Memory | The main memory space accessible by all threads, but with higher latency compared to shared memory and caches. |

> The execution processing: when a warp is scheduled, the dispatch unit sends instructions to the appropriate execution units (CUDA cores, SFU, LD/ST unit, Tensor cores) based on the type of operation being performed. The execution units then perform the computations and access data from the register file, shared memory, L1 cache, L2 cache, or global memory as needed.
> Threads within one block can cooperate by sharing data through shared memory and synchronizing their execution using __syncthreads().
> Each thread has its own private registers and L1 cache, which are not shared with other threads. L2 cache and global memory are shared among all threads in the GPU.


## Thread Management
CUDA provides a hierarchical thread organization model.
| Level | Description |
|-------|-------------|
| Grid | The highest level of thread organization, consisting of multiple blocks. A grid is launched to execute a kernel function. |
| Block | A group of threads that can cooperate and share data through shared memory. Each block is executed independently. Once a block is assigned to a SM, it stays on that SM until completion.
| Warp | A group of 32 threads within a block that are executed simultaneously on the CUDA cores. Warps are the basic unit of execution in CUDA. |
| Thread | The smallest unit of execution, representing a single instance of a kernel function. Each thread has its own unique thread ID and can access its own registers and local memory. |

> Once the kernel function is launched, the CUDA runtime returns control to the host code while the GPU executes the kernel function asynchronously. The host code can continue executing while the GPU processes the kernel function, allowing for overlapping computation and data transfer. (This is why we use cudaDeviceSynchronize() to wait for the GPU to finish before proceeding with the host code.)

The trade-off between the number of blocks and threads per block: The more register and shared memory each thread uses, the fewer thread blocks can be active on a single SM.

Threads within a warp execute the same instruction simultaneously. If threads within a warp diverge (e.g., due to conditional statements), the warp serially executes each branch path, which can lead to performance degradation. Therefore, it is important to minimize thread divergence within warps for optimal performance. (**<font color="red">We'd better avoid using "if-else" statements inside the kernel function.</font>**) 


GPUs utilize a hardware scheduler to manage the execution of warps on the CUDA cores. The scheduler selects warps that are ready to execute and dispatches them to the available CUDA cores. If a warp is stalled (e.g., waiting for memory access), the scheduler can switch to another ready warp, allowing for better utilization of the GPU resources and hiding memory latency.
































