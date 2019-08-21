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
 //Pick which GEMM tests you enable:
 const bool TEST_BLA_GEMM_BRUTE = true; //enables/disables testing of brute-force GEMM
 const bool TEST_BLA_GEMM_SHARED = true; //enables/disables testing of shared memory GEMM
 const bool TEST_BLA_GEMM_REGISTER = true; //enables/disables testing of register-based GEMM

 std::cout << "Let's try to use BLA library ..." << std::endl;

 //Create matrix A:
 bla::Matrix<float> A(2000,2000);
 //Allocate matrix A body on Host:
 A.allocateBody(-1,bla::MemKind::Pinned);
 //Set matrix A body to some non-trivial value on Host:
 A.setBodyHost();

 //Create matrix B:
 bla::Matrix<float> B(2000,2000);
 //Allocate matrix B body on Host:
 B.allocateBody(-1,bla::MemKind::Pinned);
 //Set matrix B body to some non-trivial value on Host:
 B.setBodyHost();

 //Create matrix C:
 bla::Matrix<float> C(2000,2000);
 //Allocate matrix C body on GPU#0:
 C.allocateBody(0,bla::MemKind::Regular);

 //Create matrix D:
 bla::Matrix<float> D(2000,2000);
 //Allocate matrix D body on GPU#0:
 D.allocateBody(0,bla::MemKind::Regular);

 //Copy matrix A to GPU#0 from Host:
 A.syncBody(0,-1); //Host (-1) --> GPU#0 (0)
 //Compute matrix A norm on GPU#0:
 auto normA = A.computeNorm(0);
 std::cout << "Matrix A norm = " << normA << std::endl;

 //Copy matrix B to GPU#0 from Host:
 B.syncBody(0,-1); //Host (-1) --> GPU#0 (0)
 //Compute matrix B norm on GPU#0:
 auto normB = B.computeNorm(0);
 std::cout << "Matrix B norm = " << normB << std::endl;

 //Determine total number of floating point operations:
 double flops = 2.0 * std::sqrt(static_cast<double>(A.getVolume()) *
                                static_cast<double>(B.getVolume()) *
                                static_cast<double>(C.getVolume()));
 std::cout << "Matrix multiplication C+=A*B requires " << flops/1e9 << " Gflop" << std::endl;

 //Perform reference matrix multiplication on GPU#0 with cuBLAS:
 for(int repeat = 0; repeat < 2; ++repeat){
  C.zeroBody(0); //set matrix C body to zero on GPU#0
  bla::reset_gemm_algorithm(7);
  std::cout << "Performing matrix multiplication C+=A*B with cuBLAS ... ";
  double tms = bla::time_sys_sec();
  C.multiplyAdd(false,false,A,B,0);
  double tmf = bla::time_sys_sec();
  std::cout << "Done: Time = " << tmf-tms << " s: Gflop/s = " << flops/(tmf-tms)/1e9 << std::endl;
  //Compute C norm on GPU#0:
  auto normC = C.computeNorm(0); //correct C matrix norm
  std::cout << "Matrix C norm = " << normC << std::endl;
  D.zeroBody(0); //set matrix D body to zero on GPU#0
  D.add(C,-1.0f,0); //make matrix D = -C for later correctness checks
 }

 //Perform matrix multiplication on GPU#0 with BLA GEMM brute-force:
 if(TEST_BLA_GEMM_BRUTE){
  for(int repeat = 0; repeat < 2; ++repeat){
   C.zeroBody(0); //set matrix C body to zero on GPU#0
   bla::reset_gemm_algorithm(0);
   std::cout << "Performing matrix multiplication C+=A*B with BLA GEMM brute-force ... ";
   double tms = bla::time_sys_sec();
   C.multiplyAdd(false,false,A,B,0);
   double tmf = bla::time_sys_sec();
   std::cout << "Done: Time = " << tmf-tms << " s: Gflop/s = " << flops/(tmf-tms)/1e9 << std::endl;
   //Check correctness on GPU#0:
   C.add(D,1.0f,0);
   auto norm_diff = C.computeNorm(0);
   std::cout << "Norm of the matrix C deviation from correct = " << norm_diff << std::endl;
   if(std::abs(norm_diff) > 1e-7){
    std::cout << "#FATAL: Matrix C is incorrect, fix your GPU kernel implementation!" << std::endl;
    std::exit(1);
   }
  }
 }

 //Perform matrix multiplication on GPU#0 with BLA GEMM with shared memory:
 if(TEST_BLA_GEMM_SHARED){
  for(int repeat = 0; repeat < 2; ++repeat){
   C.zeroBody(0); //set matrix C body to zero on GPU#0
   bla::reset_gemm_algorithm(1);
   std::cout << "Performing matrix multiplication C+=A*B with BLA GEMM with shared memory ... ";
   double tms = bla::time_sys_sec();
   C.multiplyAdd(false,false,A,B,0);
   double tmf = bla::time_sys_sec();
   std::cout << "Done: Time = " << tmf-tms << " s: Gflop/s = " << flops/(tmf-tms)/1e9 << std::endl;
   //Check correctness on GPU#0:
   C.add(D,1.0f,0);
   auto norm_diff = C.computeNorm(0);
   std::cout << "Norm of the matrix C deviation from correct = " << norm_diff << std::endl;
   if(std::abs(norm_diff) > 1e-7){
    std::cout << "#FATAL: Matrix C is incorrect, fix your GPU kernel implementation!" << std::endl;
    std::exit(1);
   }
  }
 }

 //Perform matrix multiplication on GPU#0 with BLA GEMM with shared memory and registers:
 if(TEST_BLA_GEMM_REGISTER){
  for(int repeat = 0; repeat < 2; ++repeat){
   C.zeroBody(0); //set matrix C body to zero on GPU#0
   bla::reset_gemm_algorithm(2);
   std::cout << "Performing matrix multiplication C+=A*B with BLA GEMM with shared memory and registers ... ";
   double tms = bla::time_sys_sec();
   C.multiplyAdd(false,false,A,B,0);
   double tmf = bla::time_sys_sec();
   std::cout << "Done: Time = " << tmf-tms << " s: Gflop/s = " << flops/(tmf-tms)/1e9 << std::endl;
  //Check correctness on GPU#0:
   C.add(D,1.0f,0);
    auto norm_diff = C.computeNorm(0);
   std::cout << "Norm of the matrix C deviation from correct = " << norm_diff << std::endl;
   if(std::abs(norm_diff) > 1e-7){
    std::cout << "#FATAL: Matrix C is incorrect, fix your GPU kernel implementation!" << std::endl;
    std::exit(1);
   }
  }
 }

 std::cout << "Seems like it works!" << std::endl;
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
