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

#include <stdio.h>
#include <iostream>
#include <string>
#include <algorithm>

#include "bla_lib.hpp"

void test_hello()
{
 std::cout << "Testing presence on GPU ..." << std::endl;
 const std::string s1("Am I really on GPU?");
 const std::string s2("Waiting for the answer ...");
 const std::string s3("Yes, you are!");

 size_t max_len = std::max(s1.size(),std::max(s2.size(),s3.size()));
 size_t str_len = max_len+1;

 char * hs1 = static_cast<char*>(bla::allocate(-1,str_len,bla::MemKind::Pinned)); assert(hs1 != nullptr);
 char * ds1 = static_cast<char*>(bla::allocate(0,str_len,bla::MemKind::Regular)); assert(ds1 != nullptr);
 int i = 0; for(const char & symb: s1) hs1[i++]=symb; hs1[s1.size()]='\0';
 printf("%s ",hs1);

 char * hs3 = static_cast<char*>(bla::allocate(-1,str_len,bla::MemKind::Pinned)); assert(hs3 != nullptr);
 char * ds3 = static_cast<char*>(bla::allocate(0,str_len,bla::MemKind::Regular)); assert(ds3 != nullptr);
 i = 0; for(const char & symb: s3) hs3[i++]=symb; hs3[s3.size()]='\0';

 cudaError_t cuerr = cudaMemcpy((void*)ds1,(void*)hs1,str_len,cudaMemcpyDefault); assert(cuerr == cudaSuccess);
 cuerr = cudaMemcpy((void*)ds3,(void*)hs3,str_len,cudaMemcpyDefault); assert(cuerr == cudaSuccess);

 cuerr = cudaGetLastError(); assert(cuerr == cudaSuccess);
 bla::gpu_test_presence(str_len,ds1,ds3);
 std::cout << s2 << " ";
 cuerr = cudaDeviceSynchronize();
 cuerr = cudaGetLastError(); assert(cuerr == cudaSuccess);

 cuerr = cudaMemcpy((void*)hs1,(void*)ds1,str_len,cudaMemcpyDefault); assert(cuerr == cudaSuccess);
 printf("%s\n",hs1);

 bla::deallocate(0,(void*)ds3,bla::MemKind::Regular);
 bla::deallocate(-1,(void*)hs3,bla::MemKind::Pinned);

 bla::deallocate(0,(void*)ds1,bla::MemKind::Regular);
 bla::deallocate(-1,(void*)hs1,bla::MemKind::Pinned);

 return;
}

void test_bla()
{
 //Create matrix A:
 bla::Matrix<float> A(1000,2000);
 //Allocate matrix A body on Host:
 A.allocateBody(-1,bla::MemKind::Pinned);
 //Set matrix A body to some value:
 A.setBodyHost();

 //Create matrix B:
 bla::Matrix<float> B(2000,3000);
 //Allocate matrix B body on Host:
 B.allocateBody(-1,bla::MemKind::Pinned);
 //Set matrix B body to some value:
 B.setBodyHost();

 //Create matrix C:
 bla::Matrix<float> C(1000,3000);
 //Allocate matrix C body on GPU#0:
 C.allocateBody(0,bla::MemKind::Pinned);
 //Set matrix C body to zero:
 C.zeroBody(0);

 return;
}


int main(int argc, char ** argv)
{

//Init the BLA library:
 bla::init();

//Test hello:
 test_hello();

//Test BLA:
 //test_bla();

//Shutdown the BLA library:
 bla::shutdown();

 return 0;
}
