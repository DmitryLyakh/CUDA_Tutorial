BINARY_NAME = bla_test.x

CXX_COMP = g++
CXX_FLAGS = -c -O3 -std=c++11 -fPIC -D_FORCE_INLINES
CXX_INC =
CXX_LIB = -lstdc++

CUDA_COMP = nvcc
CUDA_FLAGS = --compile -ccbin /usr/bin/g++ -arch=sm_50 -O3 -lineinfo -w --resource-usage -Xcompiler -fPIC -D_FORCE_INLINES
CUDA_INC = -I/usr/local/cuda/include
CUDA_LIB= -L/usr/local/cuda/lib64 -lcudart -lcublas

LINK_FLAGS = -fPIC

OBJS = timer.o bla_lib.o main.o

$(BINARY_NAME): $(OBJS)
	$(CXX_COMP) $(OBJS) $(LINK_FLAGS) $(CXX_LIB) $(CUDA_LIB)

timer.o: timer.cpp timer.hpp
	$(CXX_COMP) $(CXX_FLAGS) $(CXX_INC) $(CUDA_INC) timer.cpp

bla_lib.o: bla_lib.cu bla_lib.hpp
	$(CUDA_COMP) $(CUDA_FLAGS) $(CXX_INC) $(CUDA_INC) bla_lib.cu

main.o: main.cpp bla_lib.cu bla_lib.hpp
	$(CXX_COMP) $(CXX_FLAGS) $(CXX_INC) $(CUDA_INC) main.cpp


.PHONY: clean
clean:
	rm -f *.x *.a *.so *.o *.mod *.modmic *.ptx *.log
