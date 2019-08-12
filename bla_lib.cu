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

#include <cstdio>
#include <cassert>
#include <cmath>

#include <iostream>

namespace bla{

//CUDA floating point data type selector:
template <typename T> struct CudaFPData{};
template <> struct CudaFPData<float>{
 using type = float;
 const cudaDataType_t kind = CUDA_R_32F;
};
template <> struct CudaFPData<double>{
 using type = double;
 const cudaDataType_t kind = CUDA_R_64F;
};
template <> struct CudaFPData<std::complex<float>>{
 using type = cuComplex;
 const cudaDataType_t kind = CUDA_C_32F;
};
template <> struct CudaFPData<std::complex<double>>{
 using type = cuDoubleComplex;
 const cudaDataType_t kind = CUDA_C_64F;
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

const int TILE_EXT_X = 16;
const int TILE_EXT_Y = 16;
template <typename T>
__global__ void gpu_gemm_nn(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right);
template <typename T>
__global__ void gpu_gemm_tn(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right);
template <typename T>
__global__ void gpu_gemm_nt(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right);
template <typename T>
__global__ void gpu_gemm_tt(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right);

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
__global__ void gpu_array_norm2(size_t arr_size, const T * __restrict__ arr, volatile T * norm)
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
__global__ void gpu_array_add(size_t arr_size, T * __restrict__ arr0, const T * __restrict__ arr1)
{
 size_t n = gridDim.x * blockDim.x;
 for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < arr_size; i += n) arr0[i] += arr1[i];
 return;
}


template <typename T>
__global__ void gpu_gemm_nn(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right)
{
 __shared__ T lbuf[TILE_EXT_X][TILE_EXT_Y],rbuf[TILE_EXT_X][TILE_EXT_Y];

 //Load a tile of the left matrix into shared memory:

 //Load a tile of the right matrix into shared memory:

 //Multiply tiles and store the result in global memory:

 return;
}


template <typename T>
__global__ void gpu_gemm_tn(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right)
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


template <typename T>
__global__ void gpu_gemm_tt(int m, int n, int k, T * __restrict__ dest, const T * __restrict__ left, const T * __restrict__ right)
{
 //`Finish
 return;
}


template <typename T>
T matrix_norm2_gpu_(size_t num_elems, const T * matrix_body)
{
 T norm2 = static_cast<T>(0);
 int dev; cudaError_t cuerr = cudaGetDevice(&dev); assert(cuerr == cudaSuccess);
 T * dnorm2 = static_cast<T*>(allocate(sizeof(T),dev,MemKind::Regular));
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
 dim3 blocks(64,64); dim3 threads(16,16);
 if(!left_transp && !right_transp){
  int m = nrows0, n = ncols0, k = ncols1;
  gpu_gemm_nn<<<blocks,threads>>>(m,n,k,matrix0_body,matrix1_body,matrix2_body);
 }else if(left_transp && !right_transp){
  int m = nrows0, n = ncols0, k = nrows1;
  gpu_gemm_tn<<<blocks,threads>>>(m,n,k,matrix0_body,matrix1_body,matrix2_body);
 }else if(!left_transp && right_transp){
  int m = nrows0, n = ncols0, k = ncols1;
  gpu_gemm_nt<<<blocks,threads>>>(m,n,k,matrix0_body,matrix1_body,matrix2_body);
 }else if(left_transp && right_transp){
  int m = nrows0, n = ncols0, k = nrows1;
  gpu_gemm_tt<<<blocks,threads>>>(m,n,k,matrix0_body,matrix1_body,matrix2_body);
 }
 cudaError_t cuerr = cudaDeviceSynchronize();
 cuerr = cudaGetLastError(); assert(cuerr == cudaSuccess);
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
