/* CUDA tutorial: Basic Linear Algebra (BLA) Library

!Copyright (C) 2018-2018 Dmitry I. Lyakh (Liakh)
!Copyright (C) 2018-2018 Oak Ridge National Laboratory (UT-Battelle)

!This file is part of CUDA BLA tutorial.

!CUDA BLA is free software: you can redistribute it and/or modify
!it under the terms of the GNU Lesser General Public License as published
!by the Free Software Foundation, either version 3 of the License, or
!(at your option) any later version.

!CUDA BLA is distributed in the hope that it will be useful,
!but WITHOUT ANY WARRANTY; without even the implied warranty of
!MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
!GNU Lesser General Public License for more details.

!You should have received a copy of the GNU Lesser General Public License
!along with CUDA BLA. If not, see <http://www.gnu.org/licenses/>. */

#include "bla_lib.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cassert>
#include <cmath>

#include <iostream>

namespace bla{

//GPU device constants:
__device__ __constant__ static float zero_fp32 = 0.0f;
__device__ __constant__ static float unity_fp32 = 1.0f;
__device__ __constant__ static double zero_fp64 = 0.0;
__device__ __constant__ static double unity_fp64 = 1.0;


//CUDA floating point data type selector:
template <typename T> struct CudaFPData{};
template <> struct CudaFPData<float>{
 using type = float;
 static constexpr cudaDataType_t kind = CUDA_R_32F;
};
template <> struct CudaFPData<double>{
 using type = double;
 static constexpr cudaDataType_t kind = CUDA_R_64F;
};
template <> struct CudaFPData<std::complex<float>>{
 using type = cuComplex;
 static constexpr cudaDataType_t kind = CUDA_C_32F;
};
template <> struct CudaFPData<std::complex<double>>{
 using type = cuDoubleComplex;
 static constexpr cudaDataType_t kind = CUDA_C_64F;
};


//Number of present GPU devices:
static int totalNumGPUs = 0;

//Current GEMM algorithm:
static int gemmAlgorithm = 0;

//CUDA device properties (for all GPU devices):
cudaDeviceProp * gpuProperty;

//cuBLAS handles (one per device):
cublasHandle_t * cublasHandle;


//Internal tests:
bool test_hello();
bool test_norm();


//CUDA kernel prototypes:
__global__ void gpu_test_presence(size_t str_len, char * __restrict__ dst, const char * __restrict__ src);


template <typename T>
__global__ void gpu_array_norm2(size_t arr_size, const T * __restrict__ arr, volatile T * norm);
__device__ static unsigned int norm_wr_lock = 0; //reduction lock (per GPU)


template <typename T>
__global__ void gpu_array_add(size_t arr_size, T * __restrict__ arr0, const T * __restrict__ arr1);


template <typename T>
__global__ void gpu_gemm_nn(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right);

template <typename T, int TILE_EXT_N = 16, int TILE_EXT_M = 16, int TILE_EXT_K = 64>
__global__ void gpu_gemm_sh_nn(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right);

template <typename T, int TILE_EXT_N = 64, int TILE_EXT_M = 64, int TILE_EXT_K = 16>
__global__ void gpu_gemm_sh_reg_nn(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right);

template <typename T>
__global__ void gpu_gemm_tn(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right);

template <typename T, int TILE_EXT_N = 16, int TILE_EXT_M = 16, int TILE_EXT_K = 64>
__global__ void gpu_gemm_sh_tn(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right);

template <typename T, int TILE_EXT_N = 64, int TILE_EXT_M = 64, int TILE_EXT_K = 16>
__global__ void gpu_gemm_sh_reg_tn(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right);

template <typename T>
__global__ void gpu_gemm_nt(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right);

template <typename T, int TILE_EXT_N = 16, int TILE_EXT_M = 16, int TILE_EXT_K = 64>
__global__ void gpu_gemm_sh_nt(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right);

template <typename T, int TILE_EXT_N = 64, int TILE_EXT_M = 64, int TILE_EXT_K = 16>
__global__ void gpu_gemm_sh_reg_nt(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right);

template <typename T>
__global__ void gpu_gemm_tt(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right);

template <typename T, int TILE_EXT_N = 16, int TILE_EXT_M = 16, int TILE_EXT_K = 64>
__global__ void gpu_gemm_sh_tt(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right);

template <typename T, int TILE_EXT_N = 64, int TILE_EXT_M = 64, int TILE_EXT_K = 16>
__global__ void gpu_gemm_sh_reg_tt(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right);

//template <typename T, int TILE_EXT_N = 16, int TILE_EXT_M = 16, int TILE_EXT_K = 64, int FRAG_EXT_N = 4, int FRAG_EXT_M = 8>
//__global__ void gpu_gemm_sh_reg_old_nn(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right);


//Dispatch wrappers:
template <typename T>
T matrix_norm2_gpu_(size_t num_elems, const T * matrix_body);

template <typename T>
void matrix_addition_gpu_(size_t num_elems, T * matrix0_body, const T * matrix1_body);

template <typename T>
void matrix_multiplication_gpu_(bool left_transp, bool right_transp,
                                T * matrix0_body, int nrows0, int ncols0,
                                const T * matrix1_body, int nrows1, int ncols1,
                                const T * matrix2_body, int nrows2, int ncols2);


//IMPLEMENTATION:
__global__ void gpu_test_presence(size_t str_len, char * __restrict__ dst, const char * __restrict__ src)
{
 int tid = blockIdx.x * blockDim.x + threadIdx.x;
 while(tid < str_len){
  dst[tid] = src[tid];
  tid += gridDim.x * blockDim.x;
 }
 return;
}


template <typename T>
__global__ void gpu_array_norm2(size_t arr_size,            //in: array size
                                const T * __restrict__ arr, //in: pointer to arr[arr_size]
                                volatile T * norm)          //inout: sum of the squared elements of the array
{
 extern __shared__ double thread_norm[]; //blockDim.x

 size_t n = gridDim.x * blockDim.x;
 double tnorm = 0.0;
 for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < arr_size; i += n) tnorm += arr[i] * arr[i];
 thread_norm[threadIdx.x] = tnorm;
 __syncthreads();

 unsigned int s = blockDim.x;
 while(s > 1){
  unsigned int j = (s+1U)>>1; //=(s+1)/2
  if(threadIdx.x + j < s) thread_norm[threadIdx.x] += thread_norm[threadIdx.x+j];
  __syncthreads();
  s = j;
 }

 if(threadIdx.x == 0){
  unsigned int j = 1;
  while(j){j = atomicMax(&norm_wr_lock,1);} //lock
  __threadfence();
  *norm += thread_norm[0]; //accumulate
  __threadfence();
  j=atomicExch(&norm_wr_lock,0); //unlock
 }
 __syncthreads();
 return;
}


template <typename T>
__global__ void gpu_array_add(size_t arr_size,             //in: array size
                              T * __restrict__ arr0,       //inout: pointer to arr0[arr_size]
                              const T * __restrict__ arr1) //in: pointer to arr1[arr_size]
{
 size_t n = gridDim.x * blockDim.x;
 for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < arr_size; i += n) arr0[i] += arr1[i];
 return;
}


template <typename T>
__global__ void gpu_gemm_nn(int m, int n, int k,          //in: matrix dimensions: C(m,n)+=A(m,k)*B(k,n)
                            T * __restrict__ dest,        //inout: pointer to C matrix data
                            const T * __restrict__ left,  //in: pointer to A matrix data
                            const T * __restrict__ right) //in: pointer to B matrix data
{
 size_t ty = blockIdx.y*blockDim.y + threadIdx.y; //global thread index Y
 size_t tx = blockIdx.x*blockDim.x + threadIdx.x; //global thread index X

 size_t n_pos = ty;
 while(n_pos < n){

  size_t m_pos = tx;
  while(m_pos < m){

   T tmp = static_cast<T>(0.0);
   for(size_t k_pos = 0; k_pos < k; ++k_pos){
    tmp += left[k_pos*m + m_pos] * right[n_pos*k + k_pos];
   }
   dest[n_pos*m + m_pos] += tmp;

   m_pos += gridDim.x*blockDim.x;
  }

  n_pos += gridDim.y*blockDim.y;
 }
 return;
}


template <typename T, int TILE_EXT_N, int TILE_EXT_M, int TILE_EXT_K>
__global__ void gpu_gemm_sh_nn(int m, int n, int k,          //in: matrix dimensions: C(m,n)+=A(m,k)*B(k,n)
                               T * __restrict__ dest,        //inout: pointer to C matrix data
                               const T * __restrict__ left,  //in: pointer to A matrix data
                               const T * __restrict__ right) //in: pointer to B matrix data
{
 using int_t = int; //either int or size_t
 __shared__ T lbuf[TILE_EXT_K][TILE_EXT_M], rbuf[TILE_EXT_N][TILE_EXT_K];

 for(int_t n_pos = blockIdx.y*blockDim.y; n_pos < n; n_pos += gridDim.y*blockDim.y){ //tile offset in Y dimension

  for(int_t m_pos = blockIdx.x*blockDim.x; m_pos < m; m_pos += gridDim.x*blockDim.x){ //tile offset in X dimension

   T tmp = static_cast<T>(0.0); //accumulator

   for(int_t k_pos = 0; k_pos < k; k_pos += TILE_EXT_K){ //k_pos is the position of the CUDA thread along the K dimension
    int_t k_end = k_pos + TILE_EXT_K; if(k_end > k) k_end = k;

    //Load a tile of matrix A(m_pos:TILE_EXT_M, k_pos:TILE_EXT_K):
    if(m_pos + threadIdx.x < m){
     for(int_t k_loc = k_pos + threadIdx.y; k_loc < k_end; k_loc += blockDim.y){
      lbuf[k_loc-k_pos][threadIdx.x] = left[k_loc*m + (m_pos+threadIdx.x)];
     }
    }

    //Load a tile of matrix B(k_pos:TILE_EXT_K, n_pos:TILE_EXT_N):
    if(n_pos + threadIdx.y < n){
     for(int_t k_loc = k_pos + threadIdx.x; k_loc < k_end; k_loc += blockDim.x){
      rbuf[threadIdx.y][k_loc-k_pos] = right[(n_pos+threadIdx.y)*k + k_loc];
     }
    }
    __syncthreads();

    //Multiply two loaded tiles to produce a tile of matrix C(m_pos:TILE_EXT_M,n_pos:TILE_EXT_N):
    if(m_pos + threadIdx.x < m && n_pos + threadIdx.y < n){
     if(k_end - k_pos == TILE_EXT_K){ //number of loop iterations is known at compile time: Unroll it
#pragma unroll
      for(int_t l = 0; l < TILE_EXT_K; ++l){
       tmp += lbuf[l][threadIdx.x] * rbuf[threadIdx.y][l];
      }
     }else{ //number of loop iterations is not known at compile time
      for(int_t l = 0; l < (k_end - k_pos); ++l){
       tmp += lbuf[l][threadIdx.x] * rbuf[threadIdx.y][l];
      }
     }
    }
    __syncthreads();

   } //k_pos

   //Store element of the C matrix in global memory:
   if(m_pos + threadIdx.x < m && n_pos + threadIdx.y < n)
    dest[(n_pos+threadIdx.y)*m + (m_pos+threadIdx.x)] += tmp;

  } //m_pos

 } //n_pos
 return;
}


template <typename T, int TILE_EXT_N, int TILE_EXT_M, int TILE_EXT_K>
__global__ void gpu_gemm_sh_reg_nn(int m, int n, int k,          //in: matrix dimensions: C(m,n)+=A(m,k)*B(k,n)
                                   T * __restrict__ dest,        //inout: pointer to C matrix data
                                   const T * __restrict__ left,  //in: pointer to A matrix data
                                   const T * __restrict__ right) //in: pointer to B matrix data
{
 using int_t = int; //either int or size_t
 __shared__ T lbuf[TILE_EXT_K][TILE_EXT_M], rbuf[TILE_EXT_N][TILE_EXT_K];

 for(int_t n_pos = blockIdx.y*TILE_EXT_N; n_pos < n; n_pos += gridDim.y*TILE_EXT_N){ //tile offset in Y dimension
  int_t n_end = n_pos + TILE_EXT_N; if(n_end > n) n_end = n;

  for(int_t m_pos = blockIdx.x*TILE_EXT_M; m_pos < m; m_pos += gridDim.x*TILE_EXT_M){ //tile offset in X dimension
   int_t m_end = m_pos + TILE_EXT_M; if(m_end > m) m_end = m;

   if((m_end - m_pos == TILE_EXT_M) && (n_end - n_pos == TILE_EXT_N)){ //complete tile C(TILE_EXT_M,TILE_EXT_N)

    //Initialize registers to zero:
    T dreg[4][4] = {static_cast<T>(0.0)};
    T rreg[4] = {static_cast<T>(0.0)};
    T lreg[4] = {static_cast<T>(0.0)};

    for(int_t k_pos = 0; k_pos < k; k_pos += TILE_EXT_K){ //k_pos is the position of the CUDA thread along the K dimension
     int_t k_end = k_pos + TILE_EXT_K; if(k_end > k) k_end = k;

     //Load a tile of matrix A(m_pos:TILE_EXT_M, k_pos:TILE_EXT_K):
     for(int_t m_loc = m_pos + threadIdx.x; m_loc < m_end; m_loc += blockDim.x){
      for(int_t k_loc = k_pos + threadIdx.y; k_loc < k_end; k_loc += blockDim.y){
       lbuf[k_loc - k_pos][m_loc - m_pos] = left[k_loc*m + m_loc];
      }
     }

     //Load a tile of matrix B(k_pos:TILE_EXT_K, n_pos:TILE_EXT_N):
     for(int_t n_loc = n_pos + threadIdx.y; n_loc < n_end; n_loc += blockDim.y){
      for(int_t k_loc = k_pos + threadIdx.x; k_loc < k_end; k_loc += blockDim.x){
       rbuf[n_loc - n_pos][k_loc - k_pos] = right[n_loc*k + k_loc];
      }
     }
     __syncthreads();

     //Multiply two loaded tiles to produce a tile of matrix C(m_pos:TILE_EXT_M,n_pos:TILE_EXT_N):
     if(k_end - k_pos == TILE_EXT_K){
#pragma unroll
      for(int_t l = 0; l < TILE_EXT_K; ++l){
#pragma unroll
       for(int_t j = 0; j < 4; ++j) rreg[j] = rbuf[threadIdx.y + blockDim.y*j][l];
#pragma unroll
       for(int_t j = 0; j < 4; ++j) lreg[j] = lbuf[l][threadIdx.x + blockDim.x*j];
#pragma unroll
       for(int_t j = 0; j < 4; ++j){
#pragma unroll
        for(int_t i = 0; i < 4; ++i){
         dreg[j][i] += lreg[i] * rreg[j];
        }
       }
      }
     }else{
      for(int_t l = 0; l < (k_end - k_pos); ++l){
#pragma unroll
       for(int_t j = 0; j < 4; ++j) rreg[j] = rbuf[threadIdx.y + blockDim.y*j][l];
#pragma unroll
       for(int_t j = 0; j < 4; ++j) lreg[j] = lbuf[l][threadIdx.x + blockDim.x*j];
#pragma unroll
       for(int_t j = 0; j < 4; ++j){
#pragma unroll
        for(int_t i = 0; i < 4; ++i){
         dreg[j][i] += lreg[i] * rreg[j];
        }
       }
      }
     }
     __syncthreads();

    } //k_pos

    //Store elements of the C matrix in global memory:
#pragma unroll
    for(int_t j = 0; j < 4; ++j){
#pragma unroll
     for(int_t i = 0; i < 4; ++i){
      dest[(n_pos + threadIdx.y + blockDim.y*j)*m + (m_pos + threadIdx.x + blockDim.x*i)] += dreg[j][i];
     }
    }

   }else{ //incomplete tile of C

    //Initialize registers to zero:
    T dreg[4][4] = {static_cast<T>(0.0)};
    T rreg[4] = {static_cast<T>(0.0)};
    T lreg[4] = {static_cast<T>(0.0)};

    for(int_t k_pos = 0; k_pos < k; k_pos += TILE_EXT_K){ //k_pos is the position of the CUDA thread along the K dimension
     int_t k_end = k_pos + TILE_EXT_K; if(k_end > k) k_end = k;

     //Load a tile of matrix A(m_pos:TILE_EXT_M, k_pos:TILE_EXT_K):
     for(int_t m_loc = m_pos + threadIdx.x; m_loc < m_end; m_loc += blockDim.x){
      for(int_t k_loc = k_pos + threadIdx.y; k_loc < k_end; k_loc += blockDim.y){
       lbuf[k_loc - k_pos][m_loc - m_pos] = left[k_loc*m + m_loc];
      }
     }

     //Load a tile of matrix B(k_pos:TILE_EXT_K, n_pos:TILE_EXT_N):
     for(int_t n_loc = n_pos + threadIdx.y; n_loc < n_end; n_loc += blockDim.y){
      for(int_t k_loc = k_pos + threadIdx.x; k_loc < k_end; k_loc += blockDim.x){
       rbuf[n_loc - n_pos][k_loc - k_pos] = right[n_loc*k + k_loc];
      }
     }
     __syncthreads();

     //Multiply two loaded tiles to produce a tile of matrix C(m_pos:TILE_EXT_M,n_pos:TILE_EXT_N):
     for(int_t l = 0; l < (k_end - k_pos); ++l){
      for(int_t i = 0, j = threadIdx.y; j < n_end - n_pos; j += blockDim.y, i++) rreg[i] = rbuf[j][l];
      for(int_t i = 0, j = threadIdx.x; j < m_end - m_pos; j += blockDim.x, i++) lreg[i] = lbuf[l][j];
#pragma unroll
      for(int_t j = 0; j < 4; ++j){
#pragma unroll
       for(int_t i = 0; i < 4; ++i){
        dreg[j][i] += lreg[i] * rreg[j];
       }
      }
     }
     __syncthreads();

    } //k_pos

    //Store element of the C matrix in global memory:
    for(int_t j = 0, n_loc = n_pos + threadIdx.y; n_loc < n_end; n_loc += blockDim.y, j++){
     for(int_t i = 0, m_loc = m_pos + threadIdx.x; m_loc < m_end; m_loc += blockDim.x, i++){
      dest[n_loc*m + m_loc] += dreg[j][i];
     }
    }

   }

  } //m_pos

 } //n_pos
 return;
}


template <typename T>
__global__ void gpu_gemm_tn(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right)
{
 //`Finish
 return;
}


template <typename T, int TILE_EXT_N, int TILE_EXT_M, int TILE_EXT_K>
__global__ void gpu_gemm_sh_tn(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right)
{
 //`Finish
 return;
}


template <typename T, int TILE_EXT_N, int TILE_EXT_M, int TILE_EXT_K>
__global__ void gpu_gemm_sh_reg_tn(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right)
{
 //`Finish
 return;
}


template <typename T>
__global__ void gpu_gemm_nt(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right)
{
 //`Finish
 return;
}


template <typename T, int TILE_EXT_N, int TILE_EXT_M, int TILE_EXT_K>
__global__ void gpu_gemm_sh_nt(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right)
{
 //`Finish
 return;
}


template <typename T, int TILE_EXT_N, int TILE_EXT_M, int TILE_EXT_K>
__global__ void gpu_gemm_sh_reg_nt(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right)
{
 //`Finish
 return;
}


template <typename T>
__global__ void gpu_gemm_tt(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right)
{
 //`Finish
 return;
}


template <typename T, int TILE_EXT_N, int TILE_EXT_M, int TILE_EXT_K>
__global__ void gpu_gemm_sh_tt(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right)
{
 //`Finish
 return;
}


template <typename T, int TILE_EXT_N, int TILE_EXT_M, int TILE_EXT_K>
__global__ void gpu_gemm_sh_reg_tt(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right)
{
 //`Finish
 return;
}


/*
template <typename T, int TILE_EXT_N, int TILE_EXT_M, int TILE_EXT_K, int FRAG_EXT_N, int FRAG_EXT_M>
__global__ void gpu_gemm_sh_reg_old_nn(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right)
{
 using int_t = int; //either int or size_t
 __shared__ T lbuf[TILE_EXT_K][TILE_EXT_M], rbuf[TILE_EXT_N][TILE_EXT_K];
 T lreg[FRAG_EXT_M], rreg[FRAG_EXT_N], dreg[FRAG_EXT_N][FRAG_EXT_M];

 const int_t wyb = ((threadIdx.y*blockDim.x + threadIdx.x) / warpSize) / (TILE_EXT_M/FRAG_EXT_M) * FRAG_EXT_N;
 const int_t wxb = ((threadIdx.y*blockDim.x + threadIdx.x) / warpSize) % (TILE_EXT_M/FRAG_EXT_M) * FRAG_EXT_M;
 const int_t ln = (threadIdx.y*blockDim.x + threadIdx.x) % warpSize; //thread lane index inside a warp
 const int_t lny = ln / FRAG_EXT_M; //Y position inside warp fragment
 const int_t lnx = ln % FRAG_EXT_M; //X position inside warp fragment

 for(int_t n_pos = blockIdx.y*blockDim.y; n_pos < n; n_pos += gridDim.y*blockDim.y){ //tile offset in Y dimension

  for(int_t m_pos = blockIdx.x*blockDim.x; m_pos < m; m_pos += gridDim.x*blockDim.x){ //tile offset in X dimension

   if((m_pos + TILE_EXT_M <= m) && (n_pos + TILE_EXT_N <= n)){ //complete tile (TILE_EXT_N * TILE_EXT_M)

    //Initialize C accumulators to zero:
#pragma unroll
    for(int_t j = 0; j < FRAG_EXT_N; ++j){
#pragma unroll
     for(int_t i = 0; i < FRAG_EXT_M; ++i){
      dreg[j][i] = static_cast<T>(0.0);
     }
    }

    for(int_t k_pos = 0; k_pos < k; k_pos += TILE_EXT_K){ //k_pos is the position of the CUDA thread along the K dimension
     int_t k_end = k_pos + TILE_EXT_K; if(k_end > k) k_end = k;

     //Load a tile of matrix A(m_pos:TILE_EXT_M, k_pos:TILE_EXT_K):
     for(int_t k_loc = k_pos + threadIdx.y; k_loc < k_end; k_loc += blockDim.y){
      lbuf[k_loc-k_pos][threadIdx.x] = left[k_loc*m + (m_pos+threadIdx.x)];
     }

     //Load a tile of matrix B(k_pos:TILE_EXT_K, n_pos:TILE_EXT_N):
     for(int_t k_loc = k_pos + threadIdx.x; k_loc < k_end; k_loc += blockDim.x){
      rbuf[threadIdx.y][k_loc-k_pos] = right[(n_pos+threadIdx.y)*k + k_loc];
     }
     __syncthreads();

     //Multiply two loaded tiles to produce a tile of matrix C(m_pos:TILE_EXT_M,n_pos:TILE_EXT_N):
     for(int_t l = ln; l < (k_end - k_pos); l += warpSize){
      //Load fragments of shared memory tiles into registers:
#pragma unroll
      for(int_t j = 0; j < FRAG_EXT_N; ++j) rreg[j] = rbuf[wyb + j][l];
#pragma unroll
      for(int_t j = 0; j < FRAG_EXT_M; ++j) lreg[j] = lbuf[l][wxb + j];
      //Compute outer product of tile fragments in registers:
#pragma unroll
      for(int_t j = 0; j < FRAG_EXT_N; ++j){
#pragma unroll
       for(int_t i = 0; i < FRAG_EXT_M; ++i){
        dreg[j][i] += lreg[i] * rreg[j];
       }
      }
     }
     __syncthreads();

    } //k_pos

    //Perform reduction of the C fragment within each warp:
#pragma unroll
    for(int_t j = 0; j < FRAG_EXT_N; ++j){
#pragma unroll
     for(int_t i = 0; i < FRAG_EXT_M; ++i){
#pragma unroll
      dreg[j][i] += __shfl_xor_sync(0xffffffff,dreg[j][i],16);
      dreg[j][i] += __shfl_xor_sync(0xffffffff,dreg[j][i],8);
      dreg[j][i] += __shfl_xor_sync(0xffffffff,dreg[j][i],4);
      dreg[j][i] += __shfl_xor_sync(0xffffffff,dreg[j][i],2);
      dreg[j][i] += __shfl_xor_sync(0xffffffff,dreg[j][i],1);
     }
    }

    //Upload C fragments into C matrix in global memory:
    dest[(n_pos + wyb + lny)*m + (m_pos + wxb + lnx)] = dreg[lny][lnx];

   }else{ //incomplete tile

    //Initialize accumulator to zero:
    T tmp = static_cast<T>(0.0);

    for(int_t k_pos = 0; k_pos < k; k_pos += TILE_EXT_K){ //k_pos is the position of the CUDA thread along the K dimension
     int_t k_end = k_pos + TILE_EXT_K; if(k_end > k) k_end = k;

     //Load a tile of matrix A(m_pos:TILE_EXT_M, k_pos:TILE_EXT_K):
     if(m_pos + threadIdx.x < m){
      for(int_t k_loc = k_pos + threadIdx.y; k_loc < k_end; k_loc += blockDim.y){
       lbuf[k_loc-k_pos][threadIdx.x] = left[k_loc*m + (m_pos+threadIdx.x)];
      }
     }

     //Load a tile of matrix B(k_pos:TILE_EXT_K, n_pos:TILE_EXT_N):
     if(n_pos + threadIdx.y < n){
      for(int_t k_loc = k_pos + threadIdx.x; k_loc < k_end; k_loc += blockDim.x){
       rbuf[threadIdx.y][k_loc-k_pos] = right[(n_pos+threadIdx.y)*k + k_loc];
      }
     }
     __syncthreads();

     //Multiply two loaded tiles to produce a tile of matrix C(m_pos:TILE_EXT_M,n_pos:TILE_EXT_N):
     if(m_pos + threadIdx.x < m && n_pos + threadIdx.y < n){
      if(k_end - k_pos == TILE_EXT_K){ //number of loop iterations is known at compile time: Unroll it
#pragma unroll
       for(int_t l = 0; l < TILE_EXT_K; ++l){
        tmp += lbuf[l][threadIdx.x] * rbuf[threadIdx.y][l];
       }
      }else{ //number of loop iterations is not known at compile time
       for(int_t l = 0; l < (k_end - k_pos); ++l){
        tmp += lbuf[l][threadIdx.x] * rbuf[threadIdx.y][l];
       }
      }
     }
     __syncthreads();

    } //k_pos

    //Store in C matrix into global memory:
    if(m_pos + threadIdx.x < m && n_pos + threadIdx.y < n) dest[(n_pos+threadIdx.y)*m + (m_pos+threadIdx.x)] += tmp;

   }

  } //m_pos

 } //n_pos
 return;
}
*/


template <typename T>
T matrix_norm2_gpu_(size_t num_elems, const T * matrix_body)
{
 T norm2 = static_cast<T>(0);
 int dev; cudaError_t cuerr = cudaGetDevice(&dev); assert(cuerr == cudaSuccess);
 T * dnorm2 = static_cast<T*>(allocate(sizeof(T),dev,MemKind::Regular));
 cuerr = cudaMemset((void*)dnorm2,0,sizeof(T)); assert(cuerr == cudaSuccess);
 unsigned int num_blocks = 1024; unsigned int num_threads = 256;
 gpu_array_norm2<<<num_blocks,num_threads,num_threads*sizeof(double)>>>(num_elems,matrix_body,dnorm2);
 cuerr = cudaDeviceSynchronize();
 cuerr = cudaGetLastError(); assert(cuerr == cudaSuccess);
 cuerr = cudaMemcpy((void*)(&norm2),(void*)dnorm2,sizeof(T),cudaMemcpyDefault);
 deallocate((void*)dnorm2);
 return norm2;
}


template <typename T>
void matrix_addition_gpu_(size_t num_elems, T * matrix0_body, const T * matrix1_body)
{
 unsigned int num_blocks = 1024; unsigned int num_threads = 256;
 gpu_array_add<<<num_blocks,num_threads>>>(num_elems,matrix0_body,matrix1_body);
 cudaError_t cuerr = cudaDeviceSynchronize();
 cuerr = cudaGetLastError(); assert(cuerr == cudaSuccess);
 return;
}


template <typename T>
void matrix_multiplication_gpu_(bool left_transp, bool right_transp,
                               T * matrix0_body, int nrows0, int ncols0,
                               const T * matrix1_body, int nrows1, int ncols1,
                               const T * matrix2_body, int nrows2, int ncols2)
{
 if(gemmAlgorithm == 0){ //BLA GEMM brute-force
  if(!left_transp && !right_transp){
   int m = nrows0, n = ncols0, k = ncols1;
   dim3 threads(32,32);
   dim3 blocks((nrows0-1)/32+1,(ncols0-1)/32+1);
   gpu_gemm_nn<<<blocks,threads>>>(m,n,k,matrix0_body,matrix1_body,matrix2_body);
  }else if(left_transp && !right_transp){
   int m = nrows0, n = ncols0, k = nrows1;
   dim3 threads(32,32);
   dim3 blocks((nrows0-1)/32+1,(ncols0-1)/32+1);
   gpu_gemm_tn<<<blocks,threads>>>(m,n,k,matrix0_body,matrix1_body,matrix2_body);
  }else if(!left_transp && right_transp){
   int m = nrows0, n = ncols0, k = ncols1;
   dim3 threads(32,32);
   dim3 blocks((nrows0-1)/32+1,(ncols0-1)/32+1);
   gpu_gemm_nt<<<blocks,threads>>>(m,n,k,matrix0_body,matrix1_body,matrix2_body);
  }else if(left_transp && right_transp){
   int m = nrows0, n = ncols0, k = nrows1;
   dim3 threads(32,32);
   dim3 blocks((nrows0-1)/32+1,(ncols0-1)/32+1);
   gpu_gemm_tt<<<blocks,threads>>>(m,n,k,matrix0_body,matrix1_body,matrix2_body);
  }
 }else if(gemmAlgorithm == 1){ //BLA GEMM with shared memory
  if(!left_transp && !right_transp){
   int m = nrows0, n = ncols0, k = ncols1;
   dim3 threads(16,16);
   dim3 blocks((nrows0-1)/16+1,(ncols0-1)/16+1);
   gpu_gemm_sh_nn<T,16,16,64><<<blocks,threads>>>(m,n,k,matrix0_body,matrix1_body,matrix2_body);
  }else if(left_transp && !right_transp){
   int m = nrows0, n = ncols0, k = nrows1;
   dim3 threads(16,16);
   dim3 blocks((nrows0-1)/16+1,(ncols0-1)/16+1);
   gpu_gemm_sh_tn<T,16,16,64><<<blocks,threads>>>(m,n,k,matrix0_body,matrix1_body,matrix2_body);
  }else if(!left_transp && right_transp){
   int m = nrows0, n = ncols0, k = ncols1;
   dim3 threads(16,16);
   dim3 blocks((nrows0-1)/16+1,(ncols0-1)/16+1);
   gpu_gemm_sh_nt<T,16,16,64><<<blocks,threads>>>(m,n,k,matrix0_body,matrix1_body,matrix2_body);
  }else if(left_transp && right_transp){
   int m = nrows0, n = ncols0, k = nrows1;
   dim3 threads(16,16);
   dim3 blocks((nrows0-1)/16+1,(ncols0-1)/16+1);
   gpu_gemm_sh_tt<T,16,16,64><<<blocks,threads>>>(m,n,k,matrix0_body,matrix1_body,matrix2_body);
  }
 }else if(gemmAlgorithm == 2){ //BLA GEMM with shared memory and register file
  if(!left_transp && !right_transp){
   int m = nrows0, n = ncols0, k = ncols1;
   dim3 threads(16,16);
   dim3 blocks((nrows0-1)/16+1,(ncols0-1)/16+1);
   gpu_gemm_sh_reg_nn<T,64,64,16><<<blocks,threads>>>(m,n,k,matrix0_body,matrix1_body,matrix2_body);
  }else if(left_transp && !right_transp){
   int m = nrows0, n = ncols0, k = nrows1;
   dim3 threads(16,16);
   dim3 blocks((nrows0-1)/16+1,(ncols0-1)/16+1);
   //gpu_gemm_sh_reg_tn<T,64,64,16><<<blocks,threads>>>(m,n,k,matrix0_body,matrix1_body,matrix2_body);
  }else if(!left_transp && right_transp){
   int m = nrows0, n = ncols0, k = ncols1;
   dim3 threads(16,16);
   dim3 blocks((nrows0-1)/16+1,(ncols0-1)/16+1);
   //gpu_gemm_sh_reg_nt<T,64,64,16><<<blocks,threads>>>(m,n,k,matrix0_body,matrix1_body,matrix2_body);
  }else if(left_transp && right_transp){
   int m = nrows0, n = ncols0, k = nrows1;
   dim3 threads(16,16);
   dim3 blocks((nrows0-1)/16+1,(ncols0-1)/16+1);
   //gpu_gemm_sh_reg_tt<T,64,64,16><<<blocks,threads>>>(m,n,k,matrix0_body,matrix1_body,matrix2_body);
  }
 }else{ //cuBLAS GEMM
  int dev; cudaError_t cuerr = cudaGetDevice(&dev); assert(cuerr == cudaSuccess);
  cublasOperation_t transa = CUBLAS_OP_N; if(left_transp) transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N; if(right_transp) transb = CUBLAS_OP_T;
  int m = nrows0, n = ncols0, k = ncols1;
  if(left_transp) k = nrows1;
  void *alpha, *beta;
  if(CudaFPData<T>::kind == CUDA_R_32F){
   cuerr = cudaGetSymbolAddress(&alpha,unity_fp32); assert(cuerr == cudaSuccess);
   cuerr = cudaGetSymbolAddress(&beta,unity_fp32); assert(cuerr == cudaSuccess);
  }else if(CudaFPData<T>::kind == CUDA_R_64F){
   cuerr = cudaGetSymbolAddress(&alpha,unity_fp64); assert(cuerr == cudaSuccess);
   cuerr = cudaGetSymbolAddress(&beta,unity_fp64); assert(cuerr == cudaSuccess);
  }else{
   assert(false);
  }
  cublasStatus_t custat = cublasGemmEx(cublasHandle[dev],
                                       transa,transb,
                                       m,n,k,
                                       alpha,
                                       matrix1_body,CudaFPData<T>::kind,nrows1,
                                       matrix2_body,CudaFPData<T>::kind,nrows2,
                                       beta,
                                       matrix0_body,CudaFPData<T>::kind,nrows0,
                                       CudaFPData<T>::kind, CUBLAS_GEMM_DEFAULT);
  if(custat != CUBLAS_STATUS_SUCCESS) std::cout << "#ERROR(cublasGemmEx): Eror " << custat << std::endl;
  assert(custat == CUBLAS_STATUS_SUCCESS);
 }
 cudaError_t cuerr = cudaDeviceSynchronize();
 cuerr = cudaGetLastError();
 if(cuerr != cudaSuccess){
  const char * error_str = cudaGetErrorString(cuerr);
  std::cout << "ERROR(bla::matrix_multiplication_gpu_): CUDA kernel launch failure: " << std::endl;
  printf("%s\n",error_str);
 }
 assert(cuerr == cudaSuccess);
 return;
}


float matrix_norm2_gpu(size_t num_elems, const float * matrix_body)
{
 return matrix_norm2_gpu_(num_elems,matrix_body);
}

double matrix_norm2_gpu(size_t num_elems, const double * matrix_body)
{
 return matrix_norm2_gpu_(num_elems,matrix_body);
}


void matrix_addition_gpu(size_t num_elems, float * matrix0_body, const float * matrix1_body)
{
 return matrix_addition_gpu_(num_elems,matrix0_body,matrix1_body);
}

void matrix_addition_gpu(size_t num_elems, double * matrix0_body, const double * matrix1_body)
{
 return matrix_addition_gpu_(num_elems,matrix0_body,matrix1_body);
}


void matrix_multiplication_gpu(bool left_transp, bool right_transp,
                               float * matrix0_body, int nrows0, int ncols0,
                               const float * matrix1_body, int nrows1, int ncols1,
                               const float * matrix2_body, int nrows2, int ncols2)
{
 return matrix_multiplication_gpu_(left_transp,right_transp,
                                   matrix0_body,nrows0,ncols0,
                                   matrix1_body,nrows1,ncols1,
                                   matrix2_body,nrows2,ncols2);
}

void matrix_multiplication_gpu(bool left_transp, bool right_transp,
                               double * matrix0_body, int nrows0, int ncols0,
                               const double * matrix1_body, int nrows1, int ncols1,
                               const double * matrix2_body, int nrows2, int ncols2)
{
 return matrix_multiplication_gpu_(left_transp,right_transp,
                                   matrix0_body,nrows0,ncols0,
                                   matrix1_body,nrows1,ncols1,
                                   matrix2_body,nrows2,ncols2);
}


void init()
{
 totalNumGPUs = 0;
 cudaError_t cuerr = cudaGetDeviceCount(&totalNumGPUs); assert(cuerr == cudaSuccess);
 std::cout << "Found " << totalNumGPUs << " NVIDIA GPU" << std::endl;
 if(totalNumGPUs > 0){
  cublasStatus_t cuberr;
  gpuProperty = new cudaDeviceProp[totalNumGPUs];
  cublasHandle = new cublasHandle_t[totalNumGPUs];
  //Init each GPU:
  for(int i = (totalNumGPUs - 1); i >= 0; --i){
   cuerr = cudaSetDevice(i); assert(cuerr == cudaSuccess);
   cuerr = cudaGetDeviceProperties(&(gpuProperty[i]),i); assert(cuerr == cudaSuccess);
   cuberr = cublasCreate(&(cublasHandle[i])); assert(cuberr == CUBLAS_STATUS_SUCCESS);
   cuberr = cublasSetPointerMode(cublasHandle[i],CUBLAS_POINTER_MODE_DEVICE); assert(cuberr == CUBLAS_STATUS_SUCCESS);
   cuerr = cudaGetLastError(); assert(cuerr == cudaSuccess);
   std::cout << "Initialized GPU " << i << std::endl;
  }
  //Enable P2P access between GPU:
  if(totalNumGPUs > 1){
   for(int i = (totalNumGPUs - 1); i >= 0; --i){
    if(gpuProperty[i].unifiedAddressing != 0){
     cuerr = cudaSetDevice(i); assert(cuerr == cudaSuccess);
     for(int j = (totalNumGPUs - 1); j >= 0; --j){
      if(j != i){
       if(gpuProperty[j].unifiedAddressing != 0){
        cuerr = cudaDeviceEnablePeerAccess(j,0);
        if(cuerr == cudaSuccess){
         std::cout << "GPU " << i << " can access peer GPU " << j << std::endl;
        }else{
         std::cout << "GPU " << i << " cannot access peer GPU " << j << std::endl;
        }
       }
      }
     }
    }
   }
  }
  cuerr = cudaGetLastError();
 }
 std::cout << "BLA library initialized successfully" << std::endl;
 return;
}


void shutdown()
{
 if(totalNumGPUs > 0){
  cudaError_t cuerr;
  cublasStatus_t cuberr;
  for(int i = 0; i < totalNumGPUs; ++i){
   cuberr = cublasDestroy(cublasHandle[i]); assert(cuberr == CUBLAS_STATUS_SUCCESS);
   cuerr = cudaDeviceReset(); assert(cuerr == cudaSuccess);
   std::cout << "Destroyed primary context on GPU " << i << std::endl;
  }
  delete [] cublasHandle;
  delete [] gpuProperty;
 }
 totalNumGPUs = 0;
 std::cout << "BLA library shut down successfully" << std::endl;
 return;
}


void print_device_properties(int device)
{
 cudaDeviceProp prop;
 cudaError_t cuerr = cudaGetDeviceProperties(&prop,device);
 if(cuerr == cudaSuccess){
  std::cout << "Properties of NVIDIA GPU " << device << std::endl;
  std::cout << " Compute capability: " << prop.major << "." << prop.minor << std::endl;
  std::cout << " Register file size: " << prop.regsPerBlock << std::endl;
  std::cout << " Shared memory size: " << prop.sharedMemPerBlock << std::endl;
 }else{
  std::cout << "#ERROR(bla::print_device_properties): Unable to get properties for device " << device << std::endl;
  assert(false);
 }
 return;
}


void reset_gemm_algorithm(int algo)
{
 gemmAlgorithm = algo;
 return;
}


bool test_hello()
{
 std::cout << "Testing presence on GPU ..." << std::endl;
 const std::string s1("Am I really on GPU?");
 const std::string s2("Waiting for the answer ...");
 const std::string s3("Yes, you are!");

 size_t max_len = std::max(s1.size(),std::max(s2.size(),s3.size()));
 size_t str_len = max_len+1;

 char * hs1 = static_cast<char*>(allocate(str_len,-1,MemKind::Pinned)); assert(hs1 != nullptr);
 char * ds1 = static_cast<char*>(allocate(str_len,0,MemKind::Regular)); assert(ds1 != nullptr);
 int i = 0; for(const char & symb: s1) hs1[i++]=symb; hs1[s1.size()]='\0';
 printf("%s ",hs1);

 char * hs3 = static_cast<char*>(allocate(str_len,-1,MemKind::Pinned)); assert(hs3 != nullptr);
 char * ds3 = static_cast<char*>(allocate(str_len,0,MemKind::Regular)); assert(ds3 != nullptr);
 i = 0; for(const char & symb: s3) hs3[i++]=symb; hs3[s3.size()]='\0';

 cudaError_t cuerr = cudaMemcpy((void*)ds1,(void*)hs1,str_len,cudaMemcpyDefault); assert(cuerr == cudaSuccess);
 cuerr = cudaMemcpy((void*)ds3,(void*)hs3,str_len,cudaMemcpyDefault); assert(cuerr == cudaSuccess);

 cuerr = cudaGetLastError(); assert(cuerr == cudaSuccess);
 gpu_test_presence<<<16,256>>>(str_len,ds1,ds3);
 std::cout << s2 << " ";
 cuerr = cudaDeviceSynchronize();
 cuerr = cudaGetLastError(); assert(cuerr == cudaSuccess);

 cuerr = cudaMemcpy((void*)hs1,(void*)ds1,str_len,cudaMemcpyDefault); assert(cuerr == cudaSuccess);
 printf("%s\n",hs1);

 deallocate((void*)ds3);
 deallocate((void*)hs3);

 deallocate((void*)ds1);
 deallocate((void*)hs1);

 return true;
}


bool test_norm()
{
 std::cout << "Testing norm2 on GPU 0 ... ";
 const float num_tolerance = 1e-5;
 const size_t vol = 1000000;
 const size_t dsize = vol * sizeof(float);
 float * arr0 = static_cast<float*>(allocate(dsize,-1,MemKind::Pinned));
 float * arr1 = static_cast<float*>(allocate(dsize,0,MemKind::Regular));
 float * dnorm2 = static_cast<float*>(allocate(sizeof(float),0,MemKind::Regular));

 for(size_t i = 0; i < vol; ++i) arr0[i]=1.0f/sqrt((float)vol); //value of each element to make norm equal 1

 cudaError_t cuerr = cudaMemcpy((void*)arr1,(void*)arr0,dsize,cudaMemcpyDefault); assert(cuerr == cudaSuccess);

 unsigned int num_blocks = 1024; unsigned int num_threads = 256;
 gpu_array_norm2<<<num_blocks,num_threads,num_threads*sizeof(double)>>>(vol,arr1,dnorm2);
 cuerr = cudaDeviceSynchronize();
 cuerr = cudaGetLastError(); assert(cuerr == cudaSuccess);

 float norm2 = 0.0f;
 cuerr = cudaMemcpy((void*)(&norm2),(void*)dnorm2,sizeof(float),cudaMemcpyDefault);
 std::cout << "Norm2 = " << norm2 << " (correct value is 1.0)" << std::endl;
 assert(abs(norm2-1.0f) < num_tolerance);

 deallocate((void*)dnorm2);
 deallocate((void*)arr1);
 deallocate((void*)arr0);
 return true;
}


bool test_bla()
{
 if(!test_hello()) return false;
 if(!test_norm()) return false;
 return true;
}

} //namespace bla
