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

#include "memory.hpp"

#include <cuda_runtime.h>

#include <cassert>

#include <iostream>
#include <map>

namespace bla{

//Memory chunk descriptor:
typedef struct{
 int device;
 MemKind mem_kind;
 size_t mem_size;
} MemChunkDescr;


//Register of allocated memory chunks:
std::map<void*,MemChunkDescr> mem_reg;


void * allocate(size_t size, int device, MemKind mem_kind)
{
 void * ptr = nullptr;
 cudaError_t cuerr;

 if(size > 0){
  //Allocated memory:
  switch(mem_kind){
  case MemKind::Regular:
   if(device < 0){ //Host
    ptr = malloc(size);
   }else{ //GPU device
    int dev;
    cuerr = cudaGetDevice(&dev); assert(cuerr == cudaSuccess);
    if(device != dev){
     cuerr = cudaSetDevice(device); assert(cuerr == cudaSuccess);
    }
    cuerr = cudaMalloc(&ptr,size); assert(cuerr == cudaSuccess);
    if(device != dev){
     cuerr = cudaSetDevice(dev); assert(cuerr == cudaSuccess);
    }
   }
   break;
  case MemKind::Pinned:
   if(device < 0){ //Host
    cuerr = cudaHostAlloc(&ptr,size,cudaHostAllocPortable); assert(cuerr == cudaSuccess);
   }else{ //GPU device
    std::cout << "#ERROR(BLA::memory::allocate): Pinned memory is not available on GPU!" << std::endl;
    assert(false);
   }
   break;
  case MemKind::Mapped:
   if(device < 0){ //Host
    cuerr = cudaHostAlloc(&ptr,size,cudaHostAllocPortable|cudaHostAllocMapped); assert(cuerr == cudaSuccess);
   }else{ //GPU device
    std::cout << "#ERROR(BLA::memory::allocate): Mapped pinned memory is not available on GPU!" << std::endl;
    assert(false);
   }
   break;
  case MemKind::Unified:
   std::cout << "#ERROR(BLA::memory::allocate): Unified memory allocation is not implemented!" << std::endl;
   assert(false);
   break;
  }
 }
 //Register memory with BLA:
 if(ptr != nullptr){
  auto res = mem_reg.emplace(std::make_pair(ptr,MemChunkDescr{device,mem_kind,size}));
  assert(res.second);
 }
 return ptr;
}


void deallocate(void * ptr)
{
 assert(ptr != nullptr);
 //Find the memory chunk descriptor:
 auto pos = mem_reg.find(ptr);
 if(pos == mem_reg.end()){
  std::cout << "#ERROR(BLA::memory::deallocate): Attempt to deallocate a pointer not allocated by BLA!" << std::endl;
  assert(false);
 }
 auto device = pos->second.device;
 auto mem_kind = pos->second.mem_kind;
 //Deallocate memory:
 cudaError_t cuerr;
 switch(mem_kind){
 case MemKind::Regular:
  if(device < 0){ //Host
   free(ptr);
  }else{ //Device
   int dev;
   cuerr = cudaGetDevice(&dev); assert(cuerr == cudaSuccess);
   if(device != dev){
    cuerr = cudaSetDevice(device); assert(cuerr == cudaSuccess);
   }
   cuerr = cudaFree(ptr); assert(cuerr == cudaSuccess);
   if(device != dev){
    cuerr = cudaSetDevice(dev); assert(cuerr == cudaSuccess);
   }
  }
  break;
 case MemKind::Pinned:
  cuerr = cudaFreeHost(ptr); assert(cuerr == cudaSuccess);
  break;
 case MemKind::Mapped:
  cuerr = cudaFreeHost(ptr); assert(cuerr == cudaSuccess);
  break;
 case MemKind::Unified:
  std::cout << "#ERROR(BLA::memory::deallocate): Unified memory allocation is not implemented!" << std::endl;
  assert(false);
  break;
 }
 //Delete memory chunk descriptor:
 mem_reg.erase(ptr);
 return;
}

} //namespace bla
