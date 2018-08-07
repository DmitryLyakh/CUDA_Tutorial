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
 ~Matrix();

 T * getBodyPtr(int device) const;
 void initBody(int device, MemKind memkind);

private:

 typedef struct{
  int device;
  void * ptr;
  MemKind memkind;
 } Place;

 void setBody();

 int nrows_;
 int ncols_;
 size_t elem_size_;
 std::vector<Place> location_;
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
T * Matrix<T>::getBodyPtr(int device) const
{
 T * ptr = nullptr;
 for(const auto & loc: location_){
  if(loc.device == device) ptr = static_cast<T*>(loc.ptr);
 }
 return ptr;
}


template <typename T>
void Matrix<T>::initBody(int device, MemKind memkind)
{
 size_t mat_size = nrows_ * ncols_ * elem_size_;    //matrix size in bytes
 void * ptr = allocate(device,mat_size,memkind);    //allocate memory of requested kind on requested device
 location_.emplace_back(Place{device,ptr,memkind}); //save the new memory descriptor (Place)
 if(location_.size() == 1) setBody(); //set matrix body to some value when first allocated
 return;
}


template <typename T>
void Matrix<T>::setBody()
{
 T * mat = this->getBodyPtr(-1); //-1 is Host id
 assert(mat != nullptr);
 for(int j = 0; j < ncols_; ++j){
  int offset = j*nrows_;
  for(int i = 0; i < nrows_; ++i){
   mat[offset+i] = static_cast<T>(1)/(static_cast<T>(i) + static_cast<T>(j));
  }
 }
 return;
}


} //namespace bla

#endif //_MATRIX_HPP
