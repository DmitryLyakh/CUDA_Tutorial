CUDA Tutorial: Basic Linear Algebra (BLA) Library

AUTHOR: Dmitry I. Lyakh (Liakh): quant4me@gmail.com, liakhdi@ornl.gov

Copyright (C) 2018-2018 Dmitry I. Lyakh (Liakh)
Copyright (C) 2018-2018 Oak Ridge National Laboratory (UT-Battelle)

LICENSE: GNU Lesser General Public License v.3

Persistent location:
https://github.com/DmitryLyakh/CUDA_Tutorial.git

Presentation from the Petascale Computing Institute 2019:
Presentation.pdf

YouTube video of this tutorial:
https://youtu.be/Zqfa80APkDk

BUILD:
1. Prerequisites: Linux, g++ 5+, CUDA 9+.
2. Update CUDA_INC and CUDA_LIB paths in the Makefile (if needed).
3. Adjust CUDA_ARCH in the Makefile to your GPU compute capability.
4. If your g++ compiler is too new for CUDA, provide an older one in CUDA_HOST.
5. make
