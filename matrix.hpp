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

#ifndef _MATRIX_HPP
#define _MATRIX_HPP

#include <assert.h>
#include <string.h>
#include <vector>

//#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "timer.hpp"
#include "memory.hpp"

namespace bla{

template <typename T>
class Matrix{

public:

 Matrix(int nrows, int ncols);
 Matrix(const Matrix & matrix) = delete;
 Matrix & operator=(const Matrix &) = delete;
 Matrix(Matrix && matrix) = default;
 Matrix & operator=(Matrix && matrix) = default;
 virtual ~Matrix();

 int getNumRows() const; //returns the number of rows
 int getNumCols() const; //returns the number of columns
 size_t getSize() const; //return the size of the matrix in bytes
 T * getBodyPtr(int device) const; //returns a pointer to the memory resource on requested device (if any)
 void allocateBody(int device, MemKind memkind); //allocates memory resource of requested kind on requested device
 void markBodyStatus(int device, bool status); //marks body status as up-to-date or not (outdated)
 void zeroBody(int device); //initializes matrix body to zero on any device
 void setBodyHost(); //initializes matrix body to some value on Host

private:

 typedef struct{
  int device;
  void * ptr;
  MemKind memkind;
  bool uptodate;
 } Resource;

 int nrows_;
 int ncols_;
 size_t elem_size_;
 std::vector<Resource> location_;
};


//TEMPLATE DEFINITIONS:
template <typename T>
Matrix<T>::Matrix(int nrows, int ncols):
 nrows_(nrows),ncols_(ncols),elem_size_(sizeof(T))
{
 assert(nrows_ > 0 && ncols_ > 0 && elem_size_ > 0);
 std::cout << "Matrix created with dimensions (" << nrows_ << "," << ncols_ << ")" << std::endl;
}


template <typename T>
Matrix<T>::~Matrix()
{
 for(auto & loc: location_){
  deallocate(loc.device,loc.ptr,loc.memkind);
 }
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
size_t Matrix<T>::getSize() const
{
 return (static_cast<size_t>(nrows_)*static_cast<size_t>(ncols_)*elem_size_); //matrix size in bytes
}


template <typename T>
T * Matrix<T>::getBodyPtr(int device) const
{
 T * ptr = nullptr;
 for(const auto & loc: location_){
  if(loc.device == device) ptr = static_cast<T*>(loc.ptr);
 }
 return ptr;
}


template <typename T>
void Matrix<T>::allocateBody(int device, MemKind memkind)
{
 size_t mat_size = nrows_ * ncols_ * elem_size_;       //matrix size in bytes
 void * ptr = allocate(device,mat_size,memkind);       //allocate memory of requested kind on requested device
 assert(ptr != nullptr);
 location_.emplace_back(Resource{device,ptr,memkind,false}); //save the new memory descriptor (Resource)
 std::cout << "New resource acquired on device " << device << std::endl;
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
 size_t mat_size = this->getSize();
 assert(mat != nullptr);
 if(device < 0){ //Host
  memset(((void*)mat),0,mat_size);
 }else{ //Device
  int dev;
  cudaError_t cuerr = cudaGetDevice(&dev); assert(cuerr == cudaSuccess);
  cuerr = cudaSetDevice(device); assert(cuerr == cudaSuccess);
  cuerr = cudaMemset(((void*)mat),0,mat_size); assert(cuerr == cudaSuccess);
  cuerr = cudaSetDevice(dev); assert(cuerr == cudaSuccess);
 }
 return;
}


template <typename T>
void Matrix<T>::setBodyHost()
{
 T * mat = this->getBodyPtr(-1); //-1 is Host id
 assert(mat != nullptr);
 for(int j = 0; j < ncols_; ++j){
  int offset = j*nrows_;
  for(int i = 0; i < nrows_; ++i){
   mat[offset+i] = static_cast<T>(1)/(static_cast<T>(i) + static_cast<T>(j)); //some value
  }
 }
 this->markBodyStatus(-1,true); //mark matrix body on Host as up-to-date
 return;
}


} //namespace bla

#endif //_MATRIX_HPP
