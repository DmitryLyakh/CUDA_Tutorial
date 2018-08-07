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

#ifndef _BLA_LIB_HPP
#define _BLA_LIB_HPP

//#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "timer.hpp"
#include "memory.hpp"
#include "matrix.hpp"

namespace bla{

//Initialization:
void init();
//Shutdown:
void shutdown();

//Tests:
void test_bla();

} //namespace bla

#endif //_BLA_LIB_HPP
