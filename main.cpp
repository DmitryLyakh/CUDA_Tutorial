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

#include "bla.hpp"

#include <iostream>

void use_bla()
{
 std::cout << "Let's try to use BLA library ..." << std::endl;
 //Create matrix A:
 bla::Matrix<float> A(1000,2000);
 //Allocate matrix A body on Host:
 A.allocateBody(-1,bla::MemKind::Pinned);
 //Set matrix A body to some non-trivial value on Host:
 A.setBodyHost();

 //Create matrix B:
 bla::Matrix<float> B(2000,3000);
 //Allocate matrix B body on Host:
 B.allocateBody(-1,bla::MemKind::Pinned);
 //Set matrix B body to some non-trivial value on Host:
 B.setBodyHost();

 //Create matrix C:
 bla::Matrix<float> C(1000,3000);
 //Allocate matrix C body on GPU#0:
 C.allocateBody(0,bla::MemKind::Regular);
 //Set matrix C body to zero on GPU#0:
 C.zeroBody(0);

 //Copy matrix A to GPU#0 from Host:
 A.syncBody(0,-1);
 auto normA = A.computeNorm(0);
 std::cout << "Matrix A norm = " << normA << std::endl;
 //Copy matrix B to GPU#0 from Host:
 B.syncBody(0,-1);
 auto normB = B.computeNorm(0);
 std::cout << "Matrix B norm = " << normB << std::endl;

 //Perform matrix multiplication on GPU#0:
 bla::reset_gemm_algorithm(1);
 std::cout << "Performing matrix multiplication C+=A*B ... ";
 double tms = bla::time_sys_sec();
 C.multiplyAdd(false,false,A,B,0);
 double tmf = bla::time_sys_sec();
 std::cout << "Done: Time = " << tmf-tms << " s" << std::endl;

 //Compute C norm on GPU#0:
 auto normC = C.computeNorm(0);
 std::cout << "Matrix C norm = " << normC << std::endl;

 std::cout << "Seems like it works!" << std::endl;
 return;
}


int main(int argc, char ** argv)
{

//Initialize BLA library:
 bla::init();

//Test BLA library:
 bla::test_bla();

//Use BLA library:
 use_bla();

//Shutdown BLA library:
 bla::shutdown();

 return 0;
}
