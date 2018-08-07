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

#ifndef _MEMORY_HPP
#define _MEMORY_HPP

namespace bla{

//Memory kinds:
enum class MemKind{
 Regular, //regular memory
 Pinned,  //pinned memory (only matters for Host)
 Mapped,  //mapped pinned memory (only matters for Host)
 Unified  //unified memory
};

//Allocates memory on any device (Host:-1; Device:>=0):
void * allocate(int device, size_t size, MemKind mem_kind);
//Deallocates memory on any device (Host:-1; Device:>=0):
void deallocate(int device, void * ptr, MemKind mem_kind);

} //namespace bla

#endif //_MEMORY_HPP
