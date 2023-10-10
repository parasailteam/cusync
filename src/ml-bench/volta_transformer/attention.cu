/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

//<OPTIMIZATIONS>
//</OPTIMIZATIONS>

// #if defined(TILESYNC)
// #define NO_ATOMIC_ADD
// #endif

// #if defined(TILESYNC) || defined(TILEBATCH) || defined(STRIDEDSYNC)
// #define AVOID_CUSTOM_ORDER
// #define REORDER_TILE_LOADS
// #define AVOID_WAIT_KERNEL
// #endif

#include<cusync/cusync.h>

#include "common.h"

#ifndef EVAL_TILE_SIZES
//Tile sizes of all GeMMs
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 128, 32>;  
using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 64, 32>;
const int SoftmaxRowTile = 1;
#else
//<eval tiles>
const int SoftmaxRowTile = 1;
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 128, 32>;  
using ShapeMMAWarp = cutlass::gemm::GemmShape<128, 64, 32>;
//</eval tiles>
#endif


template<uint H, uint Tile, uint stride>
struct StridedSync {
  uint waitValue_;
  uint postValue_;

  __device__ __host__ StridedSync(): waitValue_(stride), postValue_(1) {}
  
  __device__ __host__ uint waitValue(const dim3& tile, const dim3& grid) {
    return stride;
  }

  __device__ __host__ uint postValue(const dim3& tile, const dim3& grid) 
    {return 1;}

  __device__ constexpr uint tileIndex(const dim3& tile, const dim3& grid) {
    if (grid.y > ((H/8)/Tile))
      return tile.x * (grid.y/((H/8)/Tile)) + tile.y%((H/8)/Tile);
    else
    return tile.x * grid.y + tile.y;
  }

  __device__ bool isSync(const dim3& tile, const dim3& grid) {
    return true; //tile.y < (H/8)/ShapeMMAThreadBlock::kN;
  }
};

const int SoftmaxThreads = ShapeMMAThreadBlock::kN;
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;  
using namespace cusync;

#ifdef ROWSYNC 
  using XQKVCuStage = CuStage<CuStageType::Producer, RowMajorZYX, NoSync, RowSync>;
  using SCuStage = CuStage<CuStageType::Producer|CuStageType::Consumer, RowMajorZYX, RowSync, RowSync>;
  using OCuStage = CuStage<CuStageType::Producer|CuStageType::Consumer, RowMajorZYX, RowSync, RowSync>;
  using XW12CuStage = CuStage<CuStageType::Consumer, RowMajorZYX, RowSync, NoSync>;
  using Sync = RowSync;
#elif defined(TILESYNC)
  using ProdCuStage = CuStage<CuStageType::Producer, RowMajorXYZ, TileSync<1>>;
  using MiddleCuStage = CuStage<CuStageType::Producer|CuStageType::Consumer, RowMajorXYZ, TileSync<1>>;
  using ConsCuStage = CuStage<CuStageType::Consumer, RowMajorXYZ, TileSync<1>>;
  using Sync = TileSync<1>;
#elif defined(STRIDEDSYNC)
  #if defined(GPT3)
    using StridedSyncImpl = StridedSync<12288, ShapeMMAThreadBlock::kN, 3>;
  #elif defined(LLaMA)
    using StridedSyncImpl = StridedSync<8192, ShapeMMAThreadBlock::kN, 3>;
  #else
    #error "GPT3 or LLaMA"
  #endif
  using ProdCuStage = CuStage<CuStageType::Producer, RowMajorXYZ, StridedSyncImpl>;
  using MiddleCuStage = CuStage<CuStageType::Producer|CuStageType::Consumer, RowMajorXYZ, StridedSyncImpl>;
  using ConsCuStage = CuStage<CuStageType::Consumer, RowMajorXYZ, TileSync<1>>;
  using Sync = TileSync<1>;
#else
  #error "Unknown Synchronization"
#endif 

//Element types of A, B, and C
using ElementAccumulator = float;
using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementComputeEpilogue = cutlass::half_t;

//All matrices are in RowMajor
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

//Use FP-16 Tensor Cores
using MMAOp = cutlass::arch::OpClassTensorOp;

using SmArch = cutlass::arch::Sm70;

//Second GeMM in MLP performs no extra fused computations 
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                        
    128 / cutlass::sizeof_bits<ElementOutput>::value,     
    ElementAccumulator,
    ElementComputeEpilogue>;

template<bool splitK>
class BaseMLPGemm : public cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, 
                                                        ElementInputB, LayoutInputB,
                                                        ElementOutput, LayoutOutput,
                                                        ElementAccumulator, MMAOp,
                                                        SmArch, ShapeMMAThreadBlock,
                                                        ShapeMMAWarp, ShapeMMAOp,
                                                        EpilogueOp, 
                                                        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 
                                                        2, 8, 8, splitK> {};

// Baseline GeMMs
using Gemm1 = BaseMLPGemm<false>;
using Gemm2 = BaseMLPGemm<false>;
using LayoutK = cutlass::layout::ColumnMajor;

template<bool splitK>
class BColumnMajorGemm : public cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, 
                                                     ElementInputB, LayoutK,
                                                     ElementOutput, LayoutOutput,
                                                     ElementAccumulator, MMAOp,
                                                     SmArch, ShapeMMAThreadBlock,
                                                     ShapeMMAWarp, ShapeMMAOp,
                                                     EpilogueOp, 
                                                     cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 
                                                     2, 8, 8, splitK> {};

// Baseline GeMMs
using BColumnMajorGemm1 = BColumnMajorGemm<false>;
using BColumnMajorGemmSplitK1 = BColumnMajorGemm<true>;

//Baseline GeMMs with SplitK enabled
using GemmSplitK1 = BaseMLPGemm<true>;

//CuSync GeMMs
template<typename CuStage, bool splitK>
class CuSyncAttentionGemm : public cutlass::gemm::device::CuSyncGemm<CuStage, 
                                                    ElementInputA, LayoutInputA, 
                                                    ElementInputB, LayoutInputB,
                                                    ElementOutput, LayoutOutput,
                                                    ElementAccumulator, MMAOp,
                                                    SmArch, ShapeMMAThreadBlock,
                                                    ShapeMMAWarp, ShapeMMAOp,
                                                    EpilogueOp, 
                                                    cutlass::gemm::threadblock::CuSyncGemmIdentityThreadblockSwizzle<>, 
                                                    2, 8, 8, splitK> {};

template<typename CuStage, bool splitK>
class CuSyncBColumnMajorGemm : public cutlass::gemm::device::CuSyncGemm<CuStage, ElementInputA, LayoutInputA, 
                                                     ElementInputB, LayoutK,
                                                     ElementOutput, LayoutOutput,
                                                     ElementAccumulator, MMAOp,
                                                     SmArch, ShapeMMAThreadBlock,
                                                     ShapeMMAWarp, ShapeMMAOp,
                                                     EpilogueOp,
                                                     cutlass::gemm::threadblock::CuSyncGemmIdentityThreadblockSwizzle<>,
                                                     2, 8, 8, splitK> {};

// CuSync GeMMs
using XQKVCuSyncGemm = CuSyncAttentionGemm<XQKVCuStage, false>;
using SCuSyncGemm = CuSyncBColumnMajorGemm<SCuStage, false>;
using OCuSyncGemm = CuSyncAttentionGemm<OCuStage, false>;
using XW12CuSyncGemm = CuSyncAttentionGemm<XW12CuStage, false>;

using XQKVCuSyncGemmSplitK = CuSyncAttentionGemm<XQKVCuStage, true>;
using SCuSyncGemmSplitK = CuSyncBColumnMajorGemm<SCuStage, true>;
using OCuSyncGemmSplitK = CuSyncAttentionGemm<OCuStage, true>;
using XW12CuSyncGemmSplitK = CuSyncAttentionGemm<XW12CuStage, true>;

using HostTensor = cutlass::HostTensor<ElementInputA, LayoutInputA>;

struct AttentionParams {
  //Attention does following computations:
  //XQKV = X * QKV
  //Q, K, V = QKV[:,0:H/3], QKV[:,H/3:2H/3] , QKV[:,2H/3]
  //S = Q * K^T
  //P = softmax(S)
  //O = P * V
  //XW12 = O * W2  
  HostTensor x;
  HostTensor qkv;
  HostTensor xqkv;
  HostTensor s;
  HostTensor p;
  HostTensor o;
  HostTensor w2;
  HostTensor xw12;

  HostTensor ref_xqkv;
  HostTensor ref_s;
  HostTensor ref_p;
  HostTensor ref_o;
  HostTensor ref_xw12;

  cutlass::gemm::GemmCoord gemm_size_xqkv, gemm_size_s, gemm_size_o, gemm_size_xw12;
  curandState* randStates;
  bool refCheck;
  ElementComputeEpilogue alpha;
  ElementComputeEpilogue beta;

  AttentionParams(int problem[4], bool check) {
    gemm_size_xqkv = cutlass::gemm::GemmCoord(problem[0], problem[1] * 3, problem[2]);
    gemm_size_s = cutlass::gemm::GemmCoord(problem[0], problem[0], problem[1]);
    gemm_size_o = cutlass::gemm::GemmCoord(problem[0], problem[1], problem[0]);
    gemm_size_xw12 = cutlass::gemm::GemmCoord(problem[0], problem[3], problem[1]);
    
    alpha = ElementComputeEpilogue(1);
    beta = ElementComputeEpilogue(0);
  
    x    = HostTensor(gemm_size_xqkv.mk());
    qkv  = HostTensor(gemm_size_xqkv.kn());
    xqkv = HostTensor(gemm_size_xqkv.mn());
    s = HostTensor(gemm_size_s.mn());
    o = HostTensor(gemm_size_o.mn());
    w2   = HostTensor(gemm_size_xw12.kn());
    xw12 = HostTensor(gemm_size_xw12.mn());

    ref_xqkv = HostTensor(gemm_size_xqkv.mn());
    ref_s = HostTensor(gemm_size_s.mn());
    ref_o = HostTensor(gemm_size_o.mn());
    ref_xw12 = HostTensor(gemm_size_xw12.mn());

    size_t numRandStates = gemm_size_xqkv.m() * 1024;
    CUDA_CHECK(cudaMalloc(&randStates, sizeof(curandState)*(numRandStates)));
    init_curand_states<<<numRandStates/128, 128>>>(randStates, numRandStates);
    CUDA_CHECK(cudaDeviceSynchronize());
    refCheck = check;
  }

  void initIns() {
    if (refCheck) {
      memset_random2(x.host_data(), ElementOutput(0.005), 
                     ElementOutput(0.01), x.size());
      memset_random2(qkv.host_data(), ElementOutput(0.005), 
                     ElementOutput(0.01), qkv.size());
      memset_random2(w2.host_data(), ElementOutput(0.01),
                     ElementOutput(0.05), w2.size());
    } else {
      cutlass::reference::host::TensorFill(x.host_view(),
                                           ElementOutput(0.05));
      cutlass::reference::host::TensorFill(qkv.host_view(),
                                           ElementOutput(0.5));
      cutlass::reference::host::TensorFill(w2.host_view(),
                                           ElementOutput(0.01));
    }

    // Copy data from host to GPU
    x.sync_device();
    qkv.sync_device();
    w2.sync_device();
  }

  void initOuts() {
    //Zeros all output tensors
    cutlass::reference::host::TensorFill(xqkv.host_view());
    cutlass::reference::host::TensorFill(s.host_view());
    cutlass::reference::host::TensorFill(p.host_view());
    cutlass::reference::host::TensorFill(o.host_view());
    cutlass::reference::host::TensorFill(xw12.host_view());
  }

  void initRefs() {
    cutlass::reference::host::TensorFill(ref_xqkv.host_view());
    cutlass::reference::host::TensorFill(ref_s.host_view());
    cutlass::reference::host::TensorFill(ref_p.host_view());
    cutlass::reference::host::TensorFill(ref_o.host_view());
    cutlass::reference::host::TensorFill(ref_xw12.host_view());
  }
};

// template<uint NTHREADS, typename T, typename AT, uint TileM, uint TileN, uint RowTile, bool enableOverlap>
// __global__ void selfAttnDotProdSoftmaxDropout(uint32_t M, uint32_t N,
//                                               T* XQKV, T* out, float p,
//                                               curandState* randStates,
//                                               MiddleCuStage cons1, MiddleCuStage prod2) {
//   extern __shared__ half xqkRows[];

//   __shared__ AT sum;
//   if (enableOverlap)
//     prod2.tile(nullptr);
//   int linearThreadId = blockIdx.x * blockDim.x + threadIdx.x;
//   curandState* localRandState = &randStates[linearThreadId];
//   // __shared__ shRandStates[sizeof(curandState) * NTHREADS];
//   uint ROW = blockIdx.x * RowTile;
//   const uint tileRow = blockIdx.x;
//   const uint tileM = ROW/TileM;
//   if (enableOverlap) {
//     // && tileM == 0) printf("TileM %d TileN %d ROW %d\n", TileM, TileN, ROW);
//     // handle1.waitOnTilesWithSyncValue(tileM, 0, 0, 1);
//     // if (tileM < M/TileM) {
//     //   {tileM + 1, 0, 0};
//     //   handle1.waitOnTile();
//     // }
//   }

//   for (uint ti = 0; ti < RowTile && ROW < M; ti++) {
//     if (threadIdx.x == 0) {
//       sum = 0;
//     }

//     AT threadSum = (AT)0.0f;

//     for (uint COL = threadIdx.x; COL < N; COL += blockDim.x) {
//       if (enableOverlap) {
//         if (ti == 0 && ROW % TileM == 0) {
//           dim3 tile = {tileM, COL/TileN, 0};
//           cons1.wait(tile, (COL/TileN)%NTHREADS);
//         }
//       }
//       T xq = XQKV[ROW * 3 * N + COL];
//       if (enableOverlap  && ti == 0 && ROW % TileM == 0) {
//         dim3 tile = {tileM, N/TileN + COL/TileN, 0};
//         #ifdef TILESYNC
//         cons1.wait(tile, (COL/TileN)%NTHREADS);
//         #endif
//       }
//       T xk = XQKV[ROW * 3 * N + (COL + N)];
//       T xqk = xq * xk;
//       threadSum += (AT)exp((AT)xqk);
//       xqkRows[COL] = xqk;
//     }
//     __syncthreads();
//     atomicAdd(&sum, (AT)threadSum);
//     __syncthreads();
//     for (uint COL = threadIdx.x; COL < N; COL += blockDim.x) {
//       float r = curand_uniform(localRandState);
//       // if (enableOverlap && ti == 0) {
//       //   if (rowSyncOrTileSync) {

//       //   } else {
//       if (enableOverlap && ti == 0 && ROW % TileM == 0) {
//         dim3 tile = {tileM, N/TileN*2 + COL/TileN, 0};
//         #ifndef TILESYNC
//         cons1.wait(tile, (COL/TileN)%NTHREADS);
//         #endif
//       }
//       __half v = (r <= p) ? (__half)(((float)(exp((AT)xqkRows[COL]) * 
//                                      (float)XQKV[ROW* 3 * N + (COL + 2 * N)]))/sum) : (__half)0.0f;
//       out[ROW * N + COL] = v;
//       if (enableOverlap && ti == SoftmaxRowTile - 1) {
//         dim3 tile = {tileM, COL/TileN, 0};
//         prod2.post(tile, ((COL/TileN)*TileN)%NTHREADS);
//       }
//     }
//     __syncthreads();

//     ROW++;
//   }

//   // if (enableOverlap) {
//   //   if (rowSyncOrTileSync) {
//   //     // tileM = ROW/TileM;
//   //     handle2.setRowStatus(tileM, 0, 0, RowTile);
//   //   } else {
      
//   //   }
//   // }
// }

void attnRefMatmul(cutlass::gemm::GemmCoord size, ElementOutput* a, ElementOutput* b, ElementOutput* c) {
  ref_matmul<ElementOutput, ElementAccumulator>(size.m(), size.n(), 
                                                size.k(), a, b, c);
}

cudaError_t host_attention(AttentionParams& attnParams) {
  attnRefMatmul(attnParams.gemm_size_xqkv, attnParams.x.device_data(), 
                attnParams.qkv.device_data(), attnParams.ref_xqkv.host_data());
  
  //assert(attnParams.ref_xdot.size() == attnParams.gemm_size1.m() * attnParams.gemm_size1.n()/3);
  size_t N = attnParams.gemm_size_xqkv.n()/3;
  size_t B = attnParams.gemm_size_xqkv.m();
  ElementOutput* host_xqkv = attnParams.ref_xqkv.host_data();

  ElementOutput* host_s = attnParams.ref_s.host_data();

  for (size_t i = 0; i < B; i++) {
    for (size_t j = 0; j < B; j++) {
      ElementAccumulator result = 0.0f;
      ElementOutput r1 = (ElementOutput)0.0f;

      for (size_t k = 0; k < N; k++) {
        ElementOutput host_xq = host_xqkv[i * 3 * N + k];
        ElementOutput host_xk = host_xqkv[j * 3 * N + k + N];
        result += host_xq * host_xk;
        r1 += host_xq * host_xk;
      }
      host_s[i * B + j] = (ElementOutput)result;
    }
  }

  ElementOutput* host_p = host_s; //new ElementOutput[B * B];

  // for (size_t i = 0; i < B; i++) {
  //   float sum = 0.0f;
  //   for (size_t j = 0; j < B; j++) {
  //     sum += exp((float)host_s[i*B + j]);
  //   }
    
  //   for (size_t j = 0; j < B; j++) {
  //     //Assume dropout probability is 1.0
  //     host_p[i*B + j] = exp(host_s[i*B + j])/sum;
  //   }
  // }
  
  ElementOutput* host_o = attnParams.ref_o.host_data();
  
  for (size_t i = 0; i < B; i++) {
    for (size_t j = 0; j < N; j++) {
      ElementAccumulator result = 0.0f;
      
      for (size_t k = 0; k < B; k++) {
        ElementOutput host_xv = host_xqkv[k * 3 * N + j + N * 2];
        
        result += host_xv * host_p[i*B + k];
      }
      host_o[i * N + j] = (ElementOutput)result;
    }
  }

  attnParams.ref_o.sync_device();

  attnRefMatmul(attnParams.gemm_size_xw12, attnParams.ref_o.device_data(), 
                attnParams.w2.device_data(), attnParams.ref_xw12.host_data());
  
  return cudaSuccess;
}

cudaError_t check_results(AttentionParams& attnParams) {
  attnParams.xqkv.sync_host();
  printf("Checking XQKV=X*QKV\n");
  bool eq = equals(attnParams.ref_xqkv.size(), 
                   attnParams.ref_xqkv.host_data(), 
                   attnParams.xqkv.host_data(), 1e-1f);
  if (eq == false) {
    printf("not correct\n");
    return cudaErrorUnknown;
  }
  
  attnParams.s.sync_host();
  printf("Checking S=Q*K.T\n");
  eq = equals(attnParams.ref_s.size(), attnParams.ref_s.host_data(),
              attnParams.s.host_data(), 1e-1f);
  if (eq == false) {
    printf("not correct\n");
    return cudaErrorUnknown;
  }

  attnParams.o.sync_host();
  printf("Checking O=S*V\n");
  eq = equals(attnParams.ref_o.size(), attnParams.ref_o.host_data(),
              attnParams.o.host_data(), 1e-1f);
  if (eq == false) {
    printf("not correct\n");
    return cudaErrorUnknown;
  }

  printf("Passed\n");

  return cudaSuccess;
}

__global__ void print_kernel(ElementOutput* data) {
  if (threadIdx.x < 10) {
    printf("%p %f\n", data, (float)data[threadIdx.x]);
  }
}

//Run our baseline of Self-Attention
template<typename GemmTy1, typename GemmTy2, typename GemmTy3, typename GemmTy4>
cudaError_t runAttentionBaseline(int split_k1, int split_k2, int split_k3, int split_k4,
                                 AttentionParams& attnParams,
                                 cudaStream_t streams[],
                                 double& execTime,
                                 double& matmul1Time,
                                 double& matmul2Time,
                                 double& matmul3Time,
                                 double& matmul4Time,
                                 int iters = 100) {  
  // ElementOutput* device_xqkv = tensor_xqkv.device_data();
  cutlass::Status status;
  //Setup First GeMM
  typename GemmTy1::Arguments args1{attnParams.gemm_size_xqkv,
                                    attnParams.x.device_ref(),
                                    attnParams.qkv.device_ref(),
                                    attnParams.xqkv.device_ref(),
                                    attnParams.xqkv.device_ref(),
                                    {attnParams.alpha, attnParams.beta},
                                    split_k1};
  size_t workspace_size = GemmTy1::get_workspace_size(args1);
  cutlass::device_memory::allocation<uint8_t> workspace1(workspace_size);
  GemmTy1 gemm_op1;
  status = gemm_op1.can_implement(args1);
  CUTLASS_CHECK(status);
  status = gemm_op1.initialize(args1, workspace1.get());
  CUTLASS_CHECK(status);

  size_t N = attnParams.gemm_size_xqkv.n()/3;

  ElementOutput* device_xq = attnParams.xqkv.device_data() + 0;
  cutlass::TensorRef xq{device_xq, LayoutInputA(3*N)}; 
  ElementOutput* device_xk = attnParams.xqkv.device_data() + N;
  cutlass::TensorRef xk{device_xk, LayoutK(3*N)};

  //Setup S=Q*K.T GeMM
  typename GemmTy2::Arguments args2{attnParams.gemm_size_s,
                                    xq, xk,
                                    attnParams.s.device_ref(),
                                    attnParams.s.device_ref(),
                                    {attnParams.alpha, attnParams.beta},
                                    split_k2};
  workspace_size = GemmTy2::get_workspace_size(args2);
  cutlass::device_memory::allocation<uint8_t> workspace2(workspace_size);
  GemmTy2 gemm_op2;
  status = gemm_op2.can_implement(args2);
  CUTLASS_CHECK(status);
  status = gemm_op2.initialize(args2, workspace2.get());
  CUTLASS_CHECK(status);

  ElementOutput* device_xv = attnParams.xqkv.device_data() + N;
  cutlass::TensorRef xv{device_xv, LayoutInputB(3*N)};

  //Setup O=S*V GeMM
  typename GemmTy3::Arguments args3{attnParams.gemm_size_o,
                                    attnParams.s.device_ref(),
                                    xv,
                                    attnParams.o.device_ref(),
                                    attnParams.o.device_ref(),
                                    {attnParams.alpha, attnParams.beta},
                                    split_k3};
  workspace_size = GemmTy3::get_workspace_size(args3);
  cutlass::device_memory::allocation<uint8_t> workspace3(workspace_size);
  GemmTy3 gemm_op3;
  status = gemm_op3.can_implement(args3);
  CUTLASS_CHECK(status);
  status = gemm_op3.initialize(args3, workspace3.get());
  CUTLASS_CHECK(status);

  //Setup XW12=O*W12 GeMM
  typename GemmTy4::Arguments args4{attnParams.gemm_size_xw12,
                                    attnParams.o.device_ref(),
                                    attnParams.w2.device_ref(),
                                    attnParams.xw12.device_ref(),
                                    attnParams.xw12.device_ref(),
                                    {attnParams.alpha, attnParams.beta},
                                    split_k4};
  workspace_size = GemmTy4::get_workspace_size(args4);
  cutlass::device_memory::allocation<uint8_t> workspace4(workspace_size);
  GemmTy4 gemm_op4;
  status = gemm_op4.can_implement(args4);
  CUTLASS_CHECK(status);
  status = gemm_op4.initialize(args4, workspace4.get());
  CUTLASS_CHECK(status);

  execTime = 0;
  
  //Launch kernels
  for (int r = 0; r < iters; r++) {
    double start = timeInMicroSeconds();
    status = gemm_op1(streams[0]);
    CUTLASS_CHECK(status);
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    double middle1 = timeInMicroSeconds();
    double iterMatMul1 = middle1-start;
    matmul1Time += iterMatMul1;
    
    status = gemm_op2(streams[0]);
    CUTLASS_CHECK(status);
    CUDA_CHECK(cudaDeviceSynchronize());
    double middle2 = timeInMicroSeconds();
    double iterMatmul2 = middle2-middle1;
    matmul2Time += iterMatmul2;
    
    status = gemm_op3(streams[0]);
    CUTLASS_CHECK(status);
    CUDA_CHECK(cudaDeviceSynchronize());
    double middle3 = timeInMicroSeconds();
    double iterMatmul3 = middle3-middle2;
    matmul3Time += iterMatmul3;

    status = gemm_op4(streams[0]);
    CUTLASS_CHECK(status);
    CUDA_CHECK(cudaDeviceSynchronize());
    double middle4 = timeInMicroSeconds();
    double iterMatmul4 = middle4-middle3;
    matmul4Time += iterMatmul4;
  
    double end = timeInMicroSeconds();
    if (iters > 10)
      printf("{\"Total\": %lf, \"matmul1Time\": %lf, \"matmul2Time\": %lf, \"matmul3Time\": %lf, \"matmul4Time\": %lf}\n",
             end-start, iterMatMul1, iterMatmul2, iterMatmul3, iterMatmul4);
    execTime += end-start;
  }

  return cudaSuccess;
}

cudaError_t runAttentionBaseline(int split_k1, int split_k2, int split_k3, int split_k4,
                                 AttentionParams& attnParams, 
                                 cudaStream_t streams[],
                                 double& execTime,
                                 double& matmul1Time,
                                 double& matmul2Time,
                                 double& matmul3Time,
                                 double& matmul4Time,
                                 int iters = 100) {
  cudaError_t result;
  printf("652: split_k1 %d split_k2 %d split_k3 %d split_k4 %d\n", split_k1, split_k2, split_k3, split_k4);

  if (split_k1 == 1 && split_k4 == 1) {
    result = runAttentionBaseline<Gemm1, BColumnMajorGemmSplitK1, GemmSplitK1, Gemm1>(split_k1, split_k2, split_k3, split_k4, attnParams, streams, execTime, matmul1Time, matmul2Time, matmul3Time, matmul4Time, iters);
  } else if (split_k1 > 1 && split_k4 == 1) {
     result = runAttentionBaseline<GemmSplitK1, BColumnMajorGemmSplitK1, GemmSplitK1, Gemm1>(split_k1, split_k2, split_k3, split_k4, attnParams, streams, execTime, matmul1Time, matmul2Time, matmul3Time, matmul4Time, iters);
  } else if (split_k1 == 1 && split_k4 > 1) {
    result = runAttentionBaseline<Gemm1, BColumnMajorGemmSplitK1, GemmSplitK1, GemmSplitK1>(split_k1, split_k2, split_k3, split_k4, attnParams, streams, execTime, matmul1Time, matmul2Time, matmul3Time, matmul4Time, iters);
  } else if (split_k1 > 1 && split_k4 > 1) {
    result = runAttentionBaseline<GemmSplitK1, BColumnMajorGemmSplitK1, GemmSplitK1, GemmSplitK1>(split_k1, split_k2, split_k3, split_k4, attnParams, streams, execTime, matmul1Time, matmul2Time, matmul3Time, matmul4Time, iters);
  }

  return result;
}

//Self-Attention using CuSync
template<typename XQKVCuSyncGemmTy, typename SCuSyncGemmTy, typename OCuSyncGemmTy, typename XW12CuSyncGemmTy>
cudaError_t runAttentionCuSync(int split_k1, int split_k2, int split_k3, int split_k4,
                               AttentionParams& attnParams,
                               XQKVCuStage& xqkvstage, SCuStage& scustage, OCuStage& ocustage, XW12CuStage& xw12custage,
                               cudaStream_t streams[],
                               double& execTime,
                               int iters = 100) {  
  //Setup XQKV = X * QKV GeMM
  typename XQKVCuSyncGemmTy::Arguments args1{xqkvstage,
                                            attnParams.gemm_size_xqkv,
                                            attnParams.x.device_ref(),
                                            attnParams.qkv.device_ref(),
                                            attnParams.xqkv.device_ref(),
                                            attnParams.xqkv.device_ref(),
                                            {attnParams.alpha, attnParams.beta},
                                            split_k1};
  size_t workspace_size = XQKVCuSyncGemmTy::get_workspace_size(args1);
  cutlass::device_memory::allocation<uint8_t> workspace1(workspace_size);
  XQKVCuSyncGemmTy gemm_op1;
  cutlass::Status status;
  status = gemm_op1.can_implement(args1);
  CUTLASS_CHECK(status);
  status = gemm_op1.initialize(args1, workspace1.get());
  CUTLASS_CHECK(status);

  size_t N = attnParams.gemm_size_xqkv.n()/3;
  ElementOutput* device_xq = attnParams.xqkv.device_data() + 0;
  cutlass::TensorRef xq{device_xq, LayoutInputA(3*N)}; 
  ElementOutput* device_xk = attnParams.xqkv.device_data() + N;
  cutlass::TensorRef xk{device_xk, LayoutK(3*N)};

  //Setup S = Q * K.T GeMM
  typename SCuSyncGemmTy::Arguments args2{scustage,
                                    attnParams.gemm_size_s,
                                    xq, xk,
                                    attnParams.s.device_ref(),
                                    attnParams.s.device_ref(),
                                    {attnParams.alpha, attnParams.beta},
                                    split_k2};
  workspace_size = SCuSyncGemmTy::get_workspace_size(args2);
  cutlass::device_memory::allocation<uint8_t> workspace2(workspace_size);
  SCuSyncGemmTy gemm_op2;
  status = gemm_op2.can_implement(args2);
  CUTLASS_CHECK(status);
  status = gemm_op2.initialize(args2, workspace2.get());
  CUTLASS_CHECK(status);
  
  ElementOutput* device_xv = attnParams.xqkv.device_data() + N;
  cutlass::TensorRef xv{device_xv, LayoutInputB(3*N)};

  //Setup O=S*V GeMM
  typename OCuSyncGemmTy::Arguments args3{ocustage,
                                    attnParams.gemm_size_o,
                                    attnParams.s.device_ref(),
                                    xv,
                                    attnParams.o.device_ref(),
                                    attnParams.o.device_ref(),
                                    {attnParams.alpha, attnParams.beta},
                                    split_k3};
  workspace_size = OCuSyncGemmTy::get_workspace_size(args3);
  cutlass::device_memory::allocation<uint8_t> workspace3(workspace_size);
  OCuSyncGemmTy gemm_op3;
  status = gemm_op3.can_implement(args3);
  CUTLASS_CHECK(status);
  status = gemm_op3.initialize(args3, workspace3.get());
  CUTLASS_CHECK(status);

  //Setup XW12=O*W12 GeMM
  typename XW12CuSyncGemmTy::Arguments args4{xw12custage,
                                    attnParams.gemm_size_xw12,
                                    attnParams.o.device_ref(),
                                    attnParams.w2.device_ref(),
                                    attnParams.xw12.device_ref(),
                                    attnParams.xw12.device_ref(),
                                    {attnParams.alpha, attnParams.beta},
                                    split_k4};
  workspace_size = XW12CuSyncGemmTy::get_workspace_size(args4);
  cutlass::device_memory::allocation<uint8_t> workspace4(workspace_size);
  XW12CuSyncGemmTy gemm_op4;
  status = gemm_op4.can_implement(args4);
  CUTLASS_CHECK(status);
  status = gemm_op4.initialize(args4, workspace4.get());
  CUTLASS_CHECK(status);

  execTime = 0;
  
  //Run Kernels in Self-Attention
  for (int r = 0; r < iters; r++) {
    double start = timeInMicroSeconds();
    status = gemm_op1.run(true, NULL, streams[0]);
    CUTLASS_CHECK(status);

    xqkvstage.invokeWaitKernel(streams[1]);
    status = gemm_op2.run(true, NULL, streams[1]);
    CUTLASS_CHECK(status);

    scustage.invokeWaitKernel(streams[2]);
    status = gemm_op3.run(true, NULL, streams[2]);
    CUTLASS_CHECK(status);

    ocustage.invokeWaitKernel(streams[3]);
    status = gemm_op4.run(true, NULL, streams[3]);
    CUTLASS_CHECK(status);

    if (status != cutlass::Status::kSuccess) {
      return cudaErrorUnknown;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double end = timeInMicroSeconds();

    if (iters > 10)
      printf("{\"Total\": %lf}\n",end-start);
    execTime += end-start;
  }

  return cudaSuccess;
}

cudaError_t runAttentionCuSync(int split_k1, int split_k2, int split_k3, int split_k4,
                               AttentionParams& attnParams,
                               XQKVCuStage& xqkvstage, SCuStage& scustage, OCuStage& ocustage, XW12CuStage& xw12custage,
                               cudaStream_t streams[],
                               double& execTime,
                               int iters = 100) {
  cudaError_t result;
  if (split_k1 == 1 && split_k4 == 1) {
    result = runAttentionCuSync<XQKVCuSyncGemm, SCuSyncGemmSplitK, OCuSyncGemmSplitK, XW12CuSyncGemm>(split_k1, split_k2, split_k3, split_k4, attnParams, 
                                              xqkvstage, scustage, ocustage, xw12custage, streams, execTime, iters);
  } else if (split_k1 == 1 && split_k4 > 1) {
    result = runAttentionCuSync<XQKVCuSyncGemm, SCuSyncGemmSplitK, OCuSyncGemmSplitK, XW12CuSyncGemmSplitK>(split_k1, split_k2, split_k3, split_k4, attnParams, 
                                              xqkvstage, scustage, ocustage, xw12custage, streams, execTime, iters);
  } else if (split_k1 > 1 && split_k4 == 1) {
    result = runAttentionCuSync<XQKVCuSyncGemmSplitK, SCuSyncGemmSplitK, OCuSyncGemmSplitK, XW12CuSyncGemm>(split_k1, split_k2, split_k3, split_k4, attnParams, 
                                              xqkvstage, scustage, ocustage, xw12custage, streams, execTime, iters);
  } else if (split_k1 > 1 && split_k4 > 1) {
    result = runAttentionCuSync<XQKVCuSyncGemmSplitK, SCuSyncGemmSplitK, OCuSyncGemmSplitK, XW12CuSyncGemmSplitK>(split_k1, split_k2, split_k3, split_k4, attnParams, 
                                              xqkvstage, scustage, ocustage, xw12custage, streams, execTime, iters);
  }

  return result;
}

int run(int argc, char* argv[]) {
  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (props.major != 7) {
    std::cerr << "Volta Tensor Ops must be run on a machine with compute capability of 70, 72, or 75."
              << std::endl;

    // Return 0 so tests are considered passing if run on unsupported architectures or CUDA Toolkits.
    return 0;
  }
  const uint NUM_ARGS = 7;
  std::string argNames[NUM_ARGS] = {"--model", "--batch", "--check", "--split-k1", "--split-k2", "--split-k3", "--split-k4"};
  std::string argHelp[NUM_ARGS] = {"GPT-3 or LLaMa", "Batch size", "Check results", 
                                   "Split K for XQKV = X*QKV GeMM", "Split K for P=Q*K.T GeMM",
                                   "Split K for O=P*V GeMM", "Split K for XW12=O*W12 GeMM"};
  
  if (argc < NUM_ARGS+1) {
    std::cout << "usage: " << std::endl
              << argNames[0] << " gpt3|llama " << argHelp[0] << std::endl 
              << argNames[1] << " <int>" << argHelp[1] << std::endl
              << argNames[2] << " true|false" << argHelp[2] << std::endl
              << argNames[3] << " <int> " << argHelp[3] << std::endl
              << argNames[4] << " <int> " << argHelp[4] << std::endl
              << argNames[5] << " <int> " << argHelp[5] << std::endl
              << argNames[6] << " <int> " << argHelp[6] << std::endl;
    return 0;
  }

  std::string model = "";
  uint batch = 0;
  bool doChecking = false;
  uint split_k1 = 1;
  uint split_k2 = 1;
  uint split_k3 = 1;
  uint split_k4 = 1;

  for (int i = 1; i < argc; ++i) {
    std::string arg = std::string(argv[i]);
    if (arg.find(argNames[0]) == 0) {
      model = std::string(argv[i+1]);
      i = i + 1;
    } else if (arg.find(argNames[1]) == 0) {
      std::stringstream ss(argv[i+1]);
      ss >> batch;
      i = i + 1;
    } else if (arg.find(argNames[2]) == 0) {
      std::string val = std::string(argv[i+1]);
      if (val == "true") {
        doChecking = true;
      } else if (val == "false") {
        doChecking = false;
      } else {
        std::cout << "Invalid value for check " << val << std::endl;
      }
      i = i + 1;
    } else if (arg.find(argNames[3]) == 0) {
      split_k1 = atoi(argv[i+1]);
      i=i+1;
    } else if (arg.find(argNames[4]) == 0) {
      split_k2 = atoi(argv[i+1]);
      i=i+1;
    } else if (arg.find(argNames[5]) == 0) {
      split_k3 = atoi(argv[i+1]);
      i=i+1;
    } else if (arg.find(argNames[6]) == 0) {
      split_k4 = atoi(argv[i+1]);
      i=i+1;
    }
  }

  if (model == "" || batch == 0) {
    std::cout<<"invalid model or batch" <<std::endl;
    return 0;
  }
    
  std::cout << "model=" << model << " batch=" << batch << "check="<<doChecking <<std::endl;
  int problem[4] = {0,0,0,0};
  problem[0] = batch;
  
  if (model=="gpt3") {
    problem[1] = 12288/8;
    problem[2] = 12288;
    problem[3] = 12288;
  } else if (model=="llama") {
    problem[1] = 8192/8;
    problem[2] = 8192;
    problem[3] = 8192;
  }

  //
  // Run the CUTLASS GEMM test.
  //

  int highestPriority;
  int lowestPriority;
  
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lowestPriority, &highestPriority));
  if (highestPriority >= lowestPriority) {
    printf("Wrong priorites: Lowest %d highest %d\n", lowestPriority, highestPriority);
  }
  cudaStream_t streams[(lowestPriority - highestPriority + 1)];
  for (int i = highestPriority; i <= lowestPriority; i++) {
    CUDA_CHECK(cudaStreamCreateWithPriority(&streams[i - highestPriority], 0, i));
  }
  
  // Create and initialize attention tensors
  AttentionParams attnParams(problem, doChecking);
  attnParams.initIns();
  attnParams.initOuts();
  attnParams.initRefs();
  
  cudaError_t result;
  int epochs = 20;
  int warmup = 10;

  if (doChecking) {
    result = host_attention(attnParams);
    CUDA_CHECK(result);
  }
  
  double baselineTime = 0;
  double matmul1Time = 0;
  double matmul2Time = 0;
  double matmul3Time = 0;
  double matmul4Time = 0;

  if (true) {
    printf("948\n");
    result = runAttentionBaseline(split_k1, split_k2, split_k3, split_k4, attnParams, streams, baselineTime, matmul1Time, matmul2Time, matmul3Time, matmul4Time, 1);
    printf("950\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    if (doChecking) {
      result = check_results(attnParams);
      CUDA_CHECK(result);
    }

    result = runAttentionBaseline(split_k1, split_k2, split_k3, split_k4, attnParams, streams, baselineTime, matmul1Time, matmul2Time, matmul3Time, matmul4Time, warmup);

    CUDA_CHECK(cudaDeviceSynchronize());
    matmul1Time = 0;
    matmul2Time = 0;
    matmul3Time = 0;
    matmul4Time = 0;
    printf("START-BASELINE:\n");
    result = runAttentionBaseline(split_k1, split_k2, split_k3, split_k4, attnParams, streams, baselineTime, matmul1Time, matmul2Time, matmul3Time, matmul4Time, epochs);

    CUDA_CHECK(result);
  
    printf("END-BASELINE: {\"Total\": %lf, \"matmul1Time\": %lf, \"matmul2Time\": %lf, \"matmul3Time\": %lf, \"matmul4Time\": %lf} microseconds\n", baselineTime/(float)epochs, matmul1Time/(float)epochs, matmul2Time/(float)epochs, matmul3Time/(float)epochs, matmul4Time/(float)epochs);
  }
  
  attnParams.initOuts();

  dim3 gridDim1 = {(uint)DIVUP(attnParams.gemm_size_xqkv.m(), ShapeMMAThreadBlock::kM),
                   (uint)DIVUP(attnParams.gemm_size_xqkv.n(), ShapeMMAThreadBlock::kN),
                   split_k1};
  dim3 gridDim2 = {(uint)DIVUP(attnParams.gemm_size_s.m(), ShapeMMAThreadBlock::kM),
                   (uint)DIVUP(attnParams.gemm_size_s.n(), ShapeMMAThreadBlock::kN),
                   split_k2};
  dim3 gridDim3 = {(uint)DIVUP(attnParams.gemm_size_o.m(), ShapeMMAThreadBlock::kM), 
                   (uint)DIVUP(attnParams.gemm_size_o.n(), ShapeMMAThreadBlock::kN),
                   split_k3};
  dim3 gridDim4 = {(uint)DIVUP(attnParams.gemm_size_xw12.m(), ShapeMMAThreadBlock::kM), 
                   (uint)DIVUP(attnParams.gemm_size_xw12.n(), ShapeMMAThreadBlock::kN),
                   split_k4};
  dim3 tileSize = {ShapeMMAThreadBlock::kM, ShapeMMAThreadBlock::kN, 1};

#ifdef ROWSYNC
  using Sync1 = RowSync;
  RowSync sync1(gridDim1.y);
  RowSync sync2(gridDim2.y);
  RowSync sync3(gridDim3.y);
  RowSync sync4(gridDim4.y);
#elif defined(TILESYNC)
  using Sync1 = TileSync<1>;
  using Sync2 = Sync1;
  TileSync<1> sync1;
  uint waitValue = DIVUP(min(attnParams.gemm_size1.m(), ShapeMMAThreadBlock::kM), SoftmaxRowTile);
  TileSync<1> sync2(waitValue, 1);
#elif defined(STRIDEDSYNC)
    StridedSyncImpl sync1;
    uint waitValue = DIVUP(min(attnParams.gemm_size1.m(), ShapeMMAThreadBlock::kM), SoftmaxRowTile);
    TileSync<1> sync2(waitValue, 1);
#else
  #error "Unknown Policy"
#endif
  
  XQKVCuStage xqkvStage(gridDim1, tileSize, NoSync(), sync1);
  SCuStage    sStage   (gridDim2, tileSize, sync1, sync2);
  OCuStage    oStage   (gridDim3, tileSize, sync2, sync3);
  XW12CuStage xw12Stage(gridDim4, tileSize, sync3, NoSync());
  
  CuSync::setProducerConsumerPair(xqkvStage, sStage);
  CuSync::setProducerConsumerPair(sStage, oStage);
  CuSync::setProducerConsumerPair(oStage, xw12Stage);

  double overlapTime = 0;
  matmul1Time = 0;
  matmul2Time = 0;
  if (true) {
    result = runAttentionCuSync(split_k1, split_k2, split_k3, split_k4, attnParams, xqkvStage, sStage, oStage, xw12Stage, streams, overlapTime, 1);

    CUDA_CHECK(cudaDeviceSynchronize());
    if (doChecking) {
      result = check_results(attnParams);
      if (result != cudaSuccess) {
        return 1;
      }
    }
    // //warmup
    result = runAttentionCuSync(split_k1, split_k2, split_k3, split_k4, attnParams, xqkvStage, sStage, oStage, xw12Stage, streams, overlapTime, warmup);

    CUDA_CHECK(cudaDeviceSynchronize());
    // printf("START-OVERLAPPED\n");
    result = runAttentionCuSync(split_k1, split_k2, split_k3, split_k4, attnParams, xqkvStage, sStage, oStage, xw12Stage, streams, overlapTime, epochs);
    
    printf("END-OVERLAPPED: {\"Total\": %lf} microseconds\n", overlapTime/(float)epochs);
  }

  return 0;
}
