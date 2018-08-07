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

#include <stdlib.h>
#include <assert.h>

#include <cuda_runtime.h>

#include "memory.hpp"

namespace bla{

void * allocate(int device, size_t size, MemKind mem_kind)
{
 void * ptr = nullptr;
 cudaError_t cuerr;
 if(size > 0){
  switch(mem_kind){
  case MemKind::Regular:
   if(device < 0){ //Host
    ptr = malloc(size);
   }else{ //Device
    int dev;
    cuerr = cudaGetDevice(&dev); assert(cuerr == cudaSuccess);
    cuerr = cudaSetDevice(device); assert(cuerr == cudaSuccess);
    cuerr = cudaMalloc(&ptr,size); assert(cuerr == cudaSuccess);
    cuerr = cudaSetDevice(dev); assert(cuerr == cudaSuccess);
   }
   break;
  case MemKind::Pinned:
   if(device < 0){ //Host
    cuerr = cudaHostAlloc(&ptr,size,cudaHostAllocPortable); assert(cuerr == cudaSuccess);
   }else{ //Device (fall back to regular)
    int dev;
    cuerr = cudaGetDevice(&dev); assert(cuerr == cudaSuccess);
    cuerr = cudaSetDevice(device); assert(cuerr == cudaSuccess);
    cuerr = cudaMalloc(&ptr,size); assert(cuerr == cudaSuccess);
    cuerr = cudaSetDevice(dev); assert(cuerr == cudaSuccess);
   }
   break;
  case MemKind::Mapped:
   if(device < 0){ //Host
    cuerr = cudaHostAlloc(&ptr,size,cudaHostAllocPortable|cudaHostAllocMapped); assert(cuerr == cudaSuccess);
   }else{ //Device (fall back to regular)
    int dev;
    cuerr = cudaGetDevice(&dev); assert(cuerr == cudaSuccess);
    cuerr = cudaSetDevice(device); assert(cuerr == cudaSuccess);
    cuerr = cudaMalloc(&ptr,size); assert(cuerr == cudaSuccess);
    cuerr = cudaSetDevice(dev); assert(cuerr == cudaSuccess);
   }
   break;
  case MemKind::Unified:
   assert(false);
   break;
  }
 }
 return ptr;
}

void deallocate(int device, void * ptr, MemKind mem_kind)
{
 assert(ptr != nullptr);
 cudaError_t cuerr;
 switch(mem_kind){
 case MemKind::Regular:
  if(device < 0){ //Host
   free(ptr);
  }else{ //Device
   int dev;
   cuerr = cudaGetDevice(&dev); assert(cuerr == cudaSuccess);
   cuerr = cudaSetDevice(device); assert(cuerr == cudaSuccess);
   cuerr = cudaFree(ptr); assert(cuerr == cudaSuccess);
   cuerr = cudaSetDevice(dev); assert(cuerr == cudaSuccess);
  }
  break;
 case MemKind::Pinned:
  if(device < 0){ //Host
   cuerr = cudaFreeHost(ptr);
  }else{ //Device
   int dev;
   cuerr = cudaGetDevice(&dev); assert(cuerr == cudaSuccess);
   cuerr = cudaSetDevice(device); assert(cuerr == cudaSuccess);
   cuerr = cudaFree(ptr); assert(cuerr == cudaSuccess);
   cuerr = cudaSetDevice(dev); assert(cuerr == cudaSuccess);
  }
  break;
 case MemKind::Mapped:
  if(device < 0){ //Host
   cuerr = cudaFreeHost(ptr);
  }else{ //Device
   int dev;
   cuerr = cudaGetDevice(&dev); assert(cuerr == cudaSuccess);
   cuerr = cudaSetDevice(device); assert(cuerr == cudaSuccess);
   cuerr = cudaFree(ptr); assert(cuerr == cudaSuccess);
   cuerr = cudaSetDevice(dev); assert(cuerr == cudaSuccess);
  }
  break;
 case MemKind::Unified:
  assert(false);
  break;
 }
 return;
}

} //namespace bla
