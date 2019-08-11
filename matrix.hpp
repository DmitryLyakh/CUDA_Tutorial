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

#ifndef MATRIX_HPP_
#define MATRIX_HPP_

#include "memory.hpp"
#include "timer.hpp"

//#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <cassert>
#include <list>
#include <type_traits>

namespace bla{

template <typename T>
class Matrix{

public:

 explicit Matrix(int nrows, int ncols);

 Matrix(const Matrix & matrix) = delete;
 Matrix & operator=(const Matrix &) = delete;
 Matrix(Matrix && matrix) noexcept = default;
 Matrix & operator=(Matrix && matrix) noexcept = default;
 virtual ~Matrix();

 /** Returns the number of rows in the matrix **/
 int getNumRows() const;
 /** Returns the number of columns in the matrix **/
 int getNumCols() const;
 /** Returns the size of the matrix in bytes **/
 std::size_t getSize() const;
 /** Returns a pointer to the memory resource on requested device (if any) **/
 T * getBodyPtr(int device) const;
 /** Allocates memory resource of requested kind on requested device **/
 void allocateBody(int device, MemKind memkind = MemKind::Regular);
 /** Deallocates memory resource on requested device **/
 void deallocateBody(int device);
 /** Marks matrix body status on a given device as up-to-date or not (outdated) **/
 void markBodyStatus(int device, bool status);
 /** Initializes matrix body to zero on a given device **/
 void zeroBody(int device);
 /** Initializes matrix body to some non-trivial value on Host **/
 void setBodyHost();
 /** Synchronizes matrix body on a given device with the body from another device.
     By default the source device is Host (if up to date). **/
 void syncBody(int device, int source_device = -1);

 /** Computes the norm of the matrix on a given device **/
 double computeNorm(int device = -1);
 /** Performs matrix addition on a given device **/
 void add(Matrix & Amat, int device = -1);
 /** Performs matrix multiplication on a given device **/
 void multiplyAdd(Matrix & Amat, Matrix & Bmat, int device = -1);

private:

 //Memory resource descriptor:
 typedef struct{
  int device;
  void * ptr;
  MemKind memkind;
  bool uptodate;
 } Resource;

 //Data members:
 int nrows_;                    //number of rows
 int ncols_;                    //number of columns
 std::size_t elem_size_;        //matrix element size in bytes
 std::list<Resource> location_; //list of memory resources occupied by the matrix
};


//TEMPLATE DEFINITIONS:
template <typename T>
Matrix<T>::Matrix(int nrows, int ncols):
 nrows_(nrows), ncols_(ncols), elem_size_(sizeof(T))
{
 static_assert(std::is_floating_point<T>::value,"#ERROR(BLA::Matrix::Matrix): Matrix type must be floating point!");
 assert(nrows_ > 0 && ncols_ > 0 && elem_size_ > 0);
 std::cout << "Matrix created with dimensions (" << nrows_ << "," << ncols_ << ")" << std::endl;
}


template <typename T>
Matrix<T>::~Matrix()
{
 for(auto & loc: location_) deallocate(loc.ptr);
 std::cout << "Matrix destroyed" << std::endl;
}


template <typename T>
int Matrix<T>::getNumRows() const
{
 return nrows_;
}


template <typename T>
int Matrix<T>::getNumCols() const
{
 return ncols_;
}


template <typename T>
std::size_t Matrix<T>::getSize() const
{
 return (static_cast<std::size_t>(nrows_)*static_cast<std::size_t>(ncols_)*elem_size_); //matrix size in bytes
}


template <typename T>
T * Matrix<T>::getBodyPtr(int device) const
{
 T * ptr = nullptr;
 for(const auto & loc: location_){
  if(loc.device == device){
   ptr = static_cast<T*>(loc.ptr);
   break;
  }
 }
 return ptr;
}


template <typename T>
void Matrix<T>::allocateBody(int device, MemKind memkind)
{
 std::size_t mat_size = this->getSize();         //matrix size in bytes
 void * ptr = allocate(mat_size,device,memkind); //allocate memory of requested kind on requested device
 assert(ptr != nullptr);
 location_.emplace_back(Resource{device,ptr,memkind,false}); //save the new memory descriptor (Resource)
 std::cout << "New resource acquired on device " << device << std::endl;
 return;
}


template <typename T>
void Matrix<T>::deallocateBody(int device)
{
 for(auto & loc: location_){
  if(loc.device == device){
   deallocate(loc.ptr);
   std::cout << "Resource released on device " << device << std::endl;
  }
 }
 location_.remove_if([device](const Resource & res){return (res.device == device);});
 return;
}


template <typename T>
void Matrix<T>::markBodyStatus(int device, bool status)
{
 for(auto & loc: location_){
  if(loc.device == device) loc.uptodate = status;
 }
 return;
}


template <typename T>
void Matrix<T>::zeroBody(int device)
{
 T * mat = this->getBodyPtr(device);
 if(mat != nullptr){
  std::size_t mat_size = this->getSize();
  assert(mat_size > 0);
  if(device < 0){ //Host
   memset(((void*)mat),0,mat_size);
  }else{ //GPU device
   int dev;
   cudaError_t cuerr = cudaGetDevice(&dev); assert(cuerr == cudaSuccess);
   if(device != dev){
    cuerr = cudaSetDevice(device); assert(cuerr == cudaSuccess);
   }
   cuerr = cudaMemset(((void*)mat),0,mat_size); assert(cuerr == cudaSuccess);
   if(device != dev){
    cuerr = cudaSetDevice(dev); assert(cuerr == cudaSuccess);
   }
  }
  this->markBodyStatus(device,true); //mark matrix body on device as up-to-date
 }else{
  std::cout << "#ERROR(BLA::Matrix::zeroBody): Matrix does not exist on device " << device << std::endl;
  assert(false);
 }
 return;
}


template <typename T>
void Matrix<T>::setBodyHost()
{
 T * mat = this->getBodyPtr(-1); //-1 is Host device id
 if(mat != nullptr){
  for(std::size_t j = 0; j < ncols_; ++j){
   std::size_t offset = j*nrows_;
   for(std::size_t i = 0; i < nrows_; ++i){
    mat[offset+i] = static_cast<T>(1)/(static_cast<T>(i) + static_cast<T>(j)); //some value
   }
  }
  this->markBodyStatus(-1,true); //mark matrix body on Host as up-to-date
 }else{
  std::cout << "#ERROR(BLA::Matrix::setBodyHost): Matrix does not exist on Host!" << std::endl;
  assert(false);
 }
 return;
}


template <typename T>
void Matrix<T>::syncBody(int device, int source_device)
{
 if(device != source_device){
  Resource destination_resource, source_resource;
  bool destination_found = false;
  bool source_found = false;
  for(auto & loc: location_){
   if(!source_found && loc.device == source_device && loc.uptodate){
    source_resource = loc;
    source_found = true;
   }
   if(!destination_found && loc.device == device){
    destination_resource = loc;
    destination_found = true;
   }
  }
  if(destination_found){
   if(source_found){
    cudaError_t cuerr = cudaMemcpy(destination_resource.ptr,source_resource.ptr,this->getSize(),cudaMemcpyDefault);
    assert(cuerr == cudaSuccess);
    this->markBodyStatus(device,true); //mark matrix body on device as up-to-date
   }else{
    std::cout << "#ERROR(BLA::Matrix::syncBody): Provided source device " << source_device << " has no up-to-date matrix body!" << std::endl;
    assert(false);
   }
  }else{
   std::cout << "#ERROR(BLA::Matrix::syncBody): Requested destination device " << device << " has no allocated resource!" << std::endl;
   assert(false);
  }
 }
 return;
}


template <typename T>
double Matrix<T>::computeNorm(int device)
{
 //`Finish
 return 0.0;
}


template <typename T>
void Matrix<T>::add(Matrix & Amat, int device)
{
 //`Finish
 return;
}


template <typename T>
void Matrix<T>::multiplyAdd(Matrix & Amat, Matrix & Bmat, int device)
{
 //`Finish
 return;
}

} //namespace bla

#endif //MATRIX_HPP_
