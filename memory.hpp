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

#ifndef MEMORY_HPP_
#define MEMORY_HPP_

#include <cstdlib>

namespace bla{

//Memory kinds:
enum class MemKind{
 Regular, //regular global memory (either Host or Device)
 Pinned,  //pinned memory (only Host)
 Mapped,  //mapped pinned memory (only Host)
 Unified  //unified memory (regardless)
};

//Allocates memory on any device (Host: -1; Device: >=0):
void * allocate(size_t size,                          //in: requested memory size in bytes
                int device = -1,                      //in: device (-1: Host; >=0: corresponding GPU)
                MemKind mem_kind = MemKind::Regular); //in: requested memory kind

//Deallocates previously allocated memory on any device:
void deallocate(void * ptr); //in: pointer to previously allocated memory

} //namespace bla

#endif //MEMORY_HPP_
