# MPI message passing Interface tutorial


https://hpc.nmsu.edu/discovery/mpi/programming-with-mpi/




## Must-know Concepts

- `communicator`: With MPI, an MPI communicator can be dynamically created and have multiple processes concurrently running on separate nodes of clusters. Each process has a unique MPI rank to identify it, its own memory space, and executes independently from the other processes. Processes communicate with each other by passing messages to exchange data. 
- `master node`


## MPI Communication methods
### Point-to-Point Communication

Blocking Communication:
- `MPI_Send`: This function is used to send a message from one process to another.
- `MPI_Recv`: This function is used to receive a message from another process.

Non-blocking Communication:
- `MPI_Isend`: This function initiates a non-blocking send operation.
- `MPI_Irecv`: This function initiates a non-blocking receive operation.


### Collective Communication
- `MPI_Bcast`: This function broadcasts a message from one process to all other processes in a communicator.
- `MPI_Reduce`: This function performs a reduction operation (like sum, max, etc.) on data from all processes and returns the result to a single process.
- `MPI_Allreduce`: This function performs a reduction operation on data from all processes and distributes the result to all processes.
- `MPI_Scatter`: This function distributes distinct chunks of data from one process to all other processes in a communicator.
- `MPI_Gather`: This function collects distinct chunks of data from all processes and gathers them into a single process.
- `MPI_Allgather`: This function collects distinct chunks of data from all processes and distributes the gathered data to all processes.

### One-sided Communication
- `MPI_Put`: This function allows a process to write data to the memory of another process without the involvement of the target process.
- `MPI_Get`: This function allows a process to read data from the memory of another process without the involvement of the target process.
- `MPI_Accumulate`: This function allows a process to perform a reduction operation on data in the memory of another process without the involvement of the target process.


