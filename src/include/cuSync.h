#include <assert.h>
#include <stdio.h>

#ifndef __CUSYNC__
#define __CUSYNC__

#define HOST_FUNC __host__
#define DEVICE_FUNC __device__

#define CUDA_CHECK(cmd) do {                        \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0);

template<typename T>
T divup(T x, T y) {
  return (x + y - 1)/y;
}

struct RowMajor {
  //overload call operator ()
  size_t order(dim3 grid, dim3 currTile) {
    return currTile.x * grid.y * grid.z + currTile.y * grid.z + currTile.z;
  }
};

template<typename Sched, typename Sync> struct CuStage;
//todo: make args constant

struct RowSync {
  uint waitValue_;
  uint postValue_;
  __device__ __host__ RowSync()  : waitValue_(0), postValue_(0) {}
  __device__ __host__ RowSync(uint waitValue) : waitValue_(waitValue), postValue_(1) {}
  __device__ __host__ RowSync(uint waitValue, uint postValue) : 
    waitValue_(waitValue), postValue_(postValue) {}
  
  __device__ uint waitValue(const dim3& tile, const dim3& grid) {
    return waitValue_;
  }

  __device__ uint tileIndex(const dim3& tile, const dim3& grid) {
    return tile.x;
  }

  __device__ bool isSync(const dim3& tile) {
    return tile.y == 0;
  }

  __device__ uint postValue(const dim3& tile, const dim3& grid) {
    return postValue_;
  }
};

struct TileSync {
  uint waitValue_;
  uint postValue_;

  __device__ __host__ TileSync(): waitValue_(1), postValue_(1) {}
  __device__ __host__ TileSync(uint waitValue, uint postValue): 
    waitValue_(waitValue), postValue_(postValue) {}
  
  __device__ __host__ uint waitValue(const dim3& tile, const dim3& grid) {return waitValue_;}
  __device__ __host__ uint postValue(const dim3& tile, const dim3& grid) {return postValue_;}

  __device__ constexpr uint tileIndex(const dim3& tile, const dim3& grid) {
    return tile.x * grid.y + tile.y;
  }

  __device__ bool isSync(const dim3& tile) {
    return true;
  }
};

template<typename Sched, typename Sync>
struct CuStage {
  dim3 grid_;
  dim3 prodGrid_;
  dim3 tileSize_;
  uint* tileCounter;
  dim3* tileOrder;
  volatile uint* tileStatusWrite_;
  volatile uint* tileStatusRead_;
  int* kernelExecuted_;
  int iter;
  bool isProducer_;
  bool isConsumer_;
  Sync syncPolicy_;
  bool canPrint;

  __device__ __host__ CuStage(): iter(0) {}

  CuStage(dim3 grid, dim3 tileSize, Sync syncPolicy) : 
    grid_(grid), tileSize_(tileSize), iter(0), prodGrid_(0), syncPolicy_(syncPolicy), isProducer_(false),
    isConsumer_(false), canPrint(false) {
      buildScheduleBuffer();
  }

  __host__ __device__ size_t numTiles() {return grid_.x * grid_.y * grid_.z;}

  void buildScheduleBuffer() {
    CUDA_CHECK(cudaMalloc(&tileCounter, sizeof(int)));
    CUDA_CHECK(cudaMemset(tileCounter, 0, sizeof(int)));
    printf("tileCounter %p\n", tileCounter);
    CUDA_CHECK(cudaMalloc(&tileOrder, sizeof(*tileOrder) * numTiles()));
    dim3* hTileOrder = new dim3[numTiles()];
  
    for (int x = 0; x < grid_.x; x++) {
    for (int y = 0; y < grid_.y; y++) {
    for (int z = 0; z < grid_.z; z++) {
      size_t id = RowMajor().order(grid_, {x, y, z});
      hTileOrder[id] = {x, y, z};
    }}}

    CUDA_CHECK(cudaMemcpy(tileOrder, hTileOrder, 
                          sizeof(*tileOrder) * numTiles(),
                          cudaMemcpyHostToDevice));
    delete[] hTileOrder;
  }

  void setTileStatusToPost(volatile uint* tileStatus) {
    tileStatusWrite_ = tileStatus;
  }

  volatile uint* getTileStatusToPost() {
    return tileStatusWrite_;
  }

  void setTileStatusToWait(volatile uint* tileStatus) {
    tileStatusRead_ = tileStatus;
  }

  volatile uint* getTileStatusToWait() {
    return tileStatusRead_;
  }

  __device__ void wait(const dim3& tile, uint waitingThread = 0) {
    if (!isConsumer()) return;
    if (!syncPolicy_.isSync(tile)) return;
  
    if (threadIdx.x == waitingThread && threadIdx.y == 0 && threadIdx.z == 0) {
      uint idx = syncPolicy_.tileIndex(tile, prodGrid_);
      while(tileStatusRead_[idx] < iter * syncPolicy_.waitValue(tile, prodGrid_));
    }

    __syncthreads();
  }

  __device__ void post(const dim3& tile, uint postThread = 0) {
    if (!isProducer()) return;
    __syncthreads();
  
    if (threadIdx.x == postThread && threadIdx.y == 0 && threadIdx.z == 0) {
      __threadfence_system();
      uint idx = syncPolicy_.tileIndex(tile, grid_);
      atomicAdd((int*)&tileStatusWrite_[idx], syncPolicy_.postValue(tile, grid_));
    }

    __syncwarp();
  }

  __device__ __host__ bool isProducer() {
    return isProducer_;
  }

  __device__ __host__ bool isConsumer() {
    return isConsumer_;
  }

  __device__ dim3 init() {}

  __device__ dim3 tile(dim3* shared_storage) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
      if (isProducer()) {
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
          *kernelExecuted_ = iter;
        }
      }
      // if (isProducerOrConsumer)
      // printf("stage.tileCounter %p stage.tileOrder %p stage.iter %d\n", stage.tileCounter, stage.tileOrder, stage.iter);   
      if (shared_storage) {
        uint linear_id = atomicAdd(tileCounter, 1) - (iter-1)*numTiles();
        *shared_storage = tileOrder[linear_id];
      }
    }

    __syncthreads();
    return (shared_storage) ? *shared_storage : dim3{0,0,0};
  }
};

__device__ inline uint glLoad(volatile uint* addr) {
  uint val;
  asm ("ld.volatile.global.u32 {%0}, [%1];" : "=r"(val) : "l"(addr));
  return val;
}

__global__ void waitKernel(volatile uint* kernelExecuted, uint expectedValue) {
  if (threadIdx.x == 0) {
    uint v = glLoad(kernelExecuted);
    while(v < expectedValue) {
      v = glLoad(kernelExecuted);
    }
  }
}

template<typename Sched1, typename Sched2, typename Sync>
struct CuSync {
  CuStage<Sched1, Sync> prod_;
  __host__ CuStage<Sched1, Sync>& prod() {return prod_;}
  CuStage<Sched2, Sync> cons_;
  __host__ CuStage<Sched2, Sync>& cons() {return cons_;}

  volatile uint* tileStatus;
  int* kernelExecuted;
  int iter;

  __device__ __host__ CuSync() {}

  void invokeWaitKernel(cudaStream_t stream) {
    waitKernel<<<1,1,0,stream>>>((uint*)kernelExecuted, prod().iter);
  }

  CuSync(CuStage<Sched1, Sync> prod, CuStage<Sched2, Sync> cons): prod_(prod), cons_(cons) {
    CUDA_CHECK(cudaMalloc(&tileStatus, prod.numTiles() * sizeof(int)));
    CUDA_CHECK(cudaMemset((uint*)tileStatus, 0, prod.numTiles() * sizeof(int)));
    iter = 0;
    cons_.prodGrid_ = prod.grid_;
    prod_.isProducer_ = true;
    cons_.isConsumer_ = true;
    prod_.setTileStatusToPost(tileStatus);
    cons_.setTileStatusToWait(tileStatus);
    CUDA_CHECK(cudaMalloc(&kernelExecuted, sizeof(int)));
    CUDA_CHECK(cudaMemset(kernelExecuted, 0, sizeof(int)));
    prod_.kernelExecuted_ = kernelExecuted;
  }
};

#endif