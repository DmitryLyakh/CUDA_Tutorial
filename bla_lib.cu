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

#include <stdio.h>
#include <iostream>
#include <assert.h>

#include "bla_lib.hpp"

namespace bla{

//Number of present GPU devices:
static int gpuAmount = 0;

//CUDA device properties:
cudaDeviceProp * gpuProperty;

//cuBLAS handles (one per device):
cublasHandle_t * cublasHandle;

//CUDA kernel prototypes:
template <typename T>
__global__ void gpu_array_norm(size_t arr_size, const T * __restrict__ arr, volatile T * norm);
__device__ static unsigned int norm_wr_lock = 0; //reduction lock (per GPU)

template <typename T>
__global__ void gpu_array_add(size_t arr_size, T * __restrict__ arr0, const T * __restrict__ arr1);

template <typename T>
__global__ void gpu_gemm_nn(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right);
static const int TILE_EXT_X = 16;
static const int TILE_EXT_Y = 16;


//DEFINITIONS:
void init()
{
 gpuAmount=0;
 cudaError_t cuerr = cudaGetDeviceCount(&gpuAmount); assert(cuerr == cudaSuccess);
 std::cout << "Found " << gpuAmount << " NVIDIA GPU" << std::endl;
 if(gpuAmount > 0){
  cublasStatus_t cuberr;
  gpuProperty = new cudaDeviceProp[gpuAmount];
  cublasHandle = new cublasHandle_t[gpuAmount];
  //Init each GPU:
  for(int i = gpuAmount-1; i >= 0; --i){
   cuerr = cudaSetDevice(i); assert(cuerr == cudaSuccess);
   cuerr = cudaGetDeviceProperties(&(gpuProperty[i]),i); assert(cuerr == cudaSuccess);
   cuberr = cublasCreate(&(cublasHandle[i])); assert(cuberr == CUBLAS_STATUS_SUCCESS);
   cuberr = cublasSetPointerMode(cublasHandle[i],CUBLAS_POINTER_MODE_DEVICE); assert(cuberr == CUBLAS_STATUS_SUCCESS);
   std::cout << "Initialized GPU " << i << std::endl;
  }
  //Enable P2P access between GPU:
  for(int i = gpuAmount-1; i >= 0; --i){
   if(gpuProperty[i].unifiedAddressing != 0){
    cuerr = cudaSetDevice(i); assert(cuerr == cudaSuccess);
    for(int j = gpuAmount-1; j >= 0; --j){
     if(j != i){
      if(gpuProperty[j].unifiedAddressing != 0){
       cuerr = cudaDeviceEnablePeerAccess(j,0); assert(cuerr == cudaSuccess);
       std::cout << "GPU " << i << " can access peer GPU " << j << std::endl;
      }
     }
    }
   }
  }
 }
 std::cout << "BLA library initialized successfully" << std::endl;
 return;
}


void shutdown()
{
 if(gpuAmount > 0){
  cudaError_t cuerr;
  cublasStatus_t cuberr;
  for(int i = 0; i < gpuAmount; ++i){
   cuberr = cublasDestroy(cublasHandle[i]); assert(cuberr == CUBLAS_STATUS_SUCCESS);
   cuerr = cudaDeviceReset(); assert(cuerr == cudaSuccess);
   std::cout << "Destroyed primary context for GPU " << i << std::endl;
  }
  delete [] cublasHandle;
  delete [] gpuProperty;
 }
 gpuAmount=0;
 std::cout << "BLA library shut down successfully" << std::endl;
 return;
}


template <typename T>
__global__ void gpu_array_norm(size_t arr_size, const T * __restrict__ arr, volatile T * norm)
{
 extern __shared__ T thread_norm[]; //blockDim.x

 size_t n = gridDim.x*blockDim.x;
 T tnorm = static_cast<T>(0.0);
 for(size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < arr_size; i += n) tnorm += arr[i] * arr[i];
 thread_norm[threadIdx.x] = tnorm;
 __syncthreads();

 unsigned int s = blockDim.x;
 while(s > 1){
  unsigned int j = (s+1U)>>1; //=(s+1)/2
  if(threadIdx.x + j < s) thread_norm[threadIdx.x] += thread_norm[threadIdx.x+j];
  __syncthreads();
  s=j;
 }

 if(threadIdx.x == 0){
  unsigned int j = 1;
  while(j){j = atomicMax(&norm_wr_lock,1);} //lock
  *norm += thread_norm[0]; //accumulate
  __threadfence();
  j=atomicExch(&norm_wr_lock,0); //unlock
 }
 __syncthreads();
 return;
}


template <typename T>
__global__ void gpu_array_add(size_t arr_size, T * __restrict__ arr0, const T * __restrict__ arr1)
{
 size_t n = gridDim.x * blockDim.x;
 for(size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < arr_size; i += n) arr0[i] += arr1[i];
 return;
}


template <typename T>
__global__ void gpu_gemm_nn(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right)
{
 T __shared__ lbuf[TILE_EXT_X][TILE_EXT_Y],rbuf[TILE_EXT_X][TILE_EXT_Y];

 return;
}

} //namespace bla
