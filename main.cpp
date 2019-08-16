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

 //Copy matrix A to GPU#0 from Host:
 A.syncBody(0,-1);
 auto normA = A.computeNorm(0);
 std::cout << "Matrix A norm = " << normA << std::endl;
 //Copy matrix B to GPU#0 from Host:
 B.syncBody(0,-1);
 auto normB = B.computeNorm(0);
 std::cout << "Matrix B norm = " << normB << std::endl;

 //Determine total number of floating point operations:
 double flops = 2.0*std::sqrt(static_cast<double>(A.getVolume()) * static_cast<double>(B.getVolume()) * static_cast<double>(C.getVolume()));
 std::cout << "Matrix multiplication C+=A*B requires " << flops/1e9 << " Gflop" << std::endl;

 //Perform matrix multiplication on GPU#0 with cuBLAS:
 double normC;
 for(int repeat = 0; repeat < 2; ++repeat){
  C.zeroBody(0); //set matrix C body to zero on GPU#0
  bla::reset_gemm_algorithm(2);
  std::cout << "Performing matrix multiplication C+=A*B with cuBLAS ... ";
  double tms = bla::time_sys_sec();
  C.multiplyAdd(false,false,A,B,0);
  double tmf = bla::time_sys_sec();
  std::cout << "Done: Time = " << tmf-tms << " s: Gflop/s = " << flops/(tmf-tms)/1e9 << std::endl;
  //Compute C norm on GPU#0:
  normC = C.computeNorm(0);
  std::cout << "Matrix C norm = " << normC << std::endl;
 }

 //Perform matrix multiplication on GPU#0 with BLA GEMM brute-force:
 for(int repeat = 0; repeat < 2; ++repeat){
  C.zeroBody(0); //set matrix C body to zero on GPU#0
  bla::reset_gemm_algorithm(0);
  std::cout << "Performing matrix multiplication C+=A*B with BLA GEMM brute-force ... ";
  double tms = bla::time_sys_sec();
  C.multiplyAdd(false,false,A,B,0);
  double tmf = bla::time_sys_sec();
  std::cout << "Done: Time = " << tmf-tms << " s: Gflop/s = " << flops/(tmf-tms)/1e9 << std::endl;
  //Compute C norm on GPU#0:
  auto norm_diff = normC;
  normC = C.computeNorm(0);
  norm_diff -= normC;
  std::cout << "Matrix C norm = " << normC << ": Error = " << std::abs(norm_diff) << std::endl;
 }

 //Perform matrix multiplication on GPU#0 with BLA GEMM with shared memory:
 for(int repeat = 0; repeat < 2; ++repeat){
  C.zeroBody(0); //set matrix C body to zero on GPU#0
  bla::reset_gemm_algorithm(1);
  std::cout << "Performing matrix multiplication C+=A*B with BLA GEMM with shared memory ... ";
  double tms = bla::time_sys_sec();
  C.multiplyAdd(false,false,A,B,0);
  double tmf = bla::time_sys_sec();
  std::cout << "Done: Time = " << tmf-tms << " s: Gflop/s = " << flops/(tmf-tms)/1e9 << std::endl;
  //Compute C norm on GPU#0:
  auto norm_diff = normC;
  normC = C.computeNorm(0);
  norm_diff -= normC;
  std::cout << "Matrix C norm = " << normC << ": Error = " << std::abs(norm_diff) << std::endl;
 }

 std::cout << "Seems like it works?" << std::endl;
 return;
}


int main(int argc, char ** argv)
{
//Initialize BLA library:
 bla::init();
 bla::print_device_properties(0); //check compute capability

//Test BLA library:
 bla::test_bla();

//Use BLA library:
 use_bla();

//Shutdown BLA library:
 bla::shutdown();

 return 0;
}
