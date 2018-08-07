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
#include <cmath>

#include "bla_lib.hpp"

namespace bla{

//Number of present GPU devices:
static int gpuAmount = 0;

//CUDA device properties:
cudaDeviceProp * gpuProperty;

//cuBLAS handles (one per device):
cublasHandle_t * cublasHandle;

//CUDA kernel prototypes:
__global__ void gpu_test_presence(size_t str_len, char * __restrict__ dst, const char * __restrict__ src);

template <typename T>
__global__ void gpu_array_norm(size_t arr_size, const T * __restrict__ arr, volatile T * norm);
__device__ static unsigned int norm_wr_lock = 0; //reduction lock (per GPU)

template <typename T>
__global__ void gpu_array_add(size_t arr_size, T * __restrict__ arr0, const T * __restrict__ arr1);

template <typename T>
__global__ void gpu_gemm_nn(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right);
static const int TILE_EXT_X = 16;
static const int TILE_EXT_Y = 16;

//Internal tests:
void test_hello();
void test_norm();


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
   cuerr = cudaGetLastError(); assert(cuerr == cudaSuccess);
   std::cout << "Initialized GPU " << i << std::endl;
  }
  //Enable P2P access between GPU:
  if(gpuAmount > 1){
   for(int i = gpuAmount-1; i >= 0; --i){
    if(gpuProperty[i].unifiedAddressing != 0){
     cuerr = cudaSetDevice(i); assert(cuerr == cudaSuccess);
     for(int j = gpuAmount-1; j >= 0; --j){
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


void test_hello()
{
 std::cout << "Testing presence on GPU ..." << std::endl;
 const std::string s1("Am I really on GPU?");
 const std::string s2("Waiting for the answer ...");
 const std::string s3("Yes, you are!");

 size_t max_len = std::max(s1.size(),std::max(s2.size(),s3.size()));
 size_t str_len = max_len+1;

 char * hs1 = static_cast<char*>(bla::allocate(-1,str_len,bla::MemKind::Pinned)); assert(hs1 != nullptr);
 char * ds1 = static_cast<char*>(bla::allocate(0,str_len,bla::MemKind::Regular)); assert(ds1 != nullptr);
 int i = 0; for(const char & symb: s1) hs1[i++]=symb; hs1[s1.size()]='\0';
 printf("%s ",hs1);

 char * hs3 = static_cast<char*>(bla::allocate(-1,str_len,bla::MemKind::Pinned)); assert(hs3 != nullptr);
 char * ds3 = static_cast<char*>(bla::allocate(0,str_len,bla::MemKind::Regular)); assert(ds3 != nullptr);
 i = 0; for(const char & symb: s3) hs3[i++]=symb; hs3[s3.size()]='\0';

 cudaError_t cuerr = cudaMemcpy((void*)ds1,(void*)hs1,str_len,cudaMemcpyDefault); assert(cuerr == cudaSuccess);
 cuerr = cudaMemcpy((void*)ds3,(void*)hs3,str_len,cudaMemcpyDefault); assert(cuerr == cudaSuccess);

 cuerr = cudaGetLastError(); assert(cuerr == cudaSuccess);
 bla::gpu_test_presence<<<16,256>>>(str_len,ds1,ds3);
 std::cout << s2 << " ";
 cuerr = cudaDeviceSynchronize();
 cuerr = cudaGetLastError(); assert(cuerr == cudaSuccess);

 cuerr = cudaMemcpy((void*)hs1,(void*)ds1,str_len,cudaMemcpyDefault); assert(cuerr == cudaSuccess);
 printf("%s\n",hs1);

 bla::deallocate(0,(void*)ds3,bla::MemKind::Regular);
 bla::deallocate(-1,(void*)hs3,bla::MemKind::Pinned);

 bla::deallocate(0,(void*)ds1,bla::MemKind::Regular);
 bla::deallocate(-1,(void*)hs1,bla::MemKind::Pinned);

 return;
}


void test_norm()
{
 std::cout << "Testing norm2 on GPU 0 ... ";
 const float num_tolerance = 1e-5;
 const size_t vol = 1000000;
 const size_t dsize = vol * sizeof(float);
 float * arr0 = static_cast<float*>(allocate(-1,dsize,MemKind::Pinned));
 float * arr1 = static_cast<float*>(allocate(0,dsize,MemKind::Regular));
 float * dnorm2 = static_cast<float*>(allocate(0,sizeof(float),MemKind::Regular));

 for(size_t i = 0; i < vol; ++i) arr0[i]=1.0/sqrt((float)vol); //value of each element to make norm equal 1

 cudaError_t cuerr = cudaMemcpy((void*)arr1,(void*)arr0,dsize,cudaMemcpyDefault);

 unsigned int numBlocks = 1024; unsigned int numThreads = 256;
 gpu_array_norm<<<numBlocks,numThreads,numThreads*sizeof(float)>>>(vol,arr1,dnorm2);
 cuerr = cudaDeviceSynchronize();
 cuerr = cudaGetLastError(); assert(cuerr == cudaSuccess);

 float norm2 = 0.0f;
 cuerr = cudaMemcpy((void*)(&norm2),(void*)dnorm2,sizeof(float),cudaMemcpyDefault);
 std::cout << "Norm2 = " << norm2 << std::endl;
 assert(abs(norm2-1.0f) < num_tolerance);

 deallocate(0,(void*)dnorm2,MemKind::Regular);
 deallocate(0,(void*)arr1,MemKind::Regular);
 deallocate(-1,(void*)arr0,MemKind::Pinned);
 return;
}


void test_bla()
{
 test_hello();
 test_norm();
 return;
}


__global__ void gpu_test_presence(size_t str_len, char * __restrict__ dst, const char * __restrict__ src)
{
 int tid = blockIdx.x * blockDim.x + threadIdx.x;
 if(tid < str_len) dst[tid] = src[tid];
 return;
}


template <typename T>
__global__ void gpu_array_norm(size_t arr_size, const T * __restrict__ arr, volatile T * norm)
{
 extern __shared__ T thread_norm[]; //blockDim.x

 size_t n = gridDim.x*blockDim.x;
 T tnorm = static_cast<T>(0);
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
 __shared__ T lbuf[TILE_EXT_X][TILE_EXT_Y],rbuf[TILE_EXT_X][TILE_EXT_Y];

 return;
}

} //namespace bla
