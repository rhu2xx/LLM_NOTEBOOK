# Basic Concept in Parallel Programming

## Key words
1. latency: the time one instruction takes to execute
2. Memory latency: the time it takes to access memory
3. throughput: the number of instructions that can be executed in a given time 

## The goal of parallel programming in CPU and GPU
1. CPU: to reduce latency by executing instructions in parallel
2. GPU: to increase throughput by executing many instructions in parallel


### Memory Access in CPU
- Cache hit
    - data found in cache -> read data from cache -> compute -> loop
    - low latency, efficient
- Cache miss
    - data not found in cache -> fetch data from main memory -> read data -> compute -> loop
    - high latency, inefficient
- Stall
    - the status when the CPU is waiting for data to be fetched from memory 

### Parallel skills in CPU
-  Pipeline (improve throught)
    - 5-stage pipeline (RISC)
        - Default: all data is in register

        | Stage | Description |
        |---|---|
        | IF (Instruction Fetch) | Fetch instruction from instruction memory |
        | ID (Instruction Decode) | Decode instruction and read registers |
        | EX (Execute) | Perform ALU operations, address calculation, and branch decisions |
        | MEM (Memory Access) | Access data memory if needed |
        | WB (Write Back) | Write results back to register |


    - Pipeline overlap
        - 4 instructions (throught = 4 instructions per cycle)
            ```text
            Cycle: 1   2   3   4   5   6   7   8

            I1:    IF  ID  EX  MEM WB
            I2:        IF  ID  EX  MEM WB
            I3:            IF  ID  EX  MEM WB
            I4:                IF  ID  EX  MEM WB
            ```
        
- cache hierarchy (reduce latency)
    - L1 cache: small, fast, close to CPU
    - L2 cache: larger, slower, further from CPU
    - L3 cache: even larger, slower, shared among cores
    - main memory: large, slow, far from CPU

- Pre-fetching (reduce latency)
    - the technique of fetching data into cache before it is actually needed by the CPU
   

- Branch prediction (reduce latency)
    - the technique of predicting the outcome of a branch instruction to keep the pipeline full
    - if wrong prediction -> rollback and flush pipeline -> high latency

- Multi-threading (improve throughput)
    - the cores that in stall or waiting for memory access can execute other instructions from other threads 

### Parallel skills in GPU
- SIMT (Single Instruction Multiple Threads)
- High throughput and low latency
- **the purpose : improve throughput by executing many instructions in parallel**
- skills:
    - multi-threading
    - warp scheduling
    - memory coalescing
    - memory latency hiding (global memory <-> shared memory, CPU <-> GPU)