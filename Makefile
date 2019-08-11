BINARY_NAME = bla_test.x

CXX_COMP = g++
CXX_FLAGS = -c -O3 -std=c++11 -fPIC -D_FORCE_INLINES
CXX_INC =
CXX_LIB = -lstdc++

CUDA_COMP = nvcc
CUDA_HOST = /usr/bin/g++
CUDA_ARCH = sm_35
CUDA_INC = -I/usr/local/cuda/include
CUDA_LIB = -L/usr/local/cuda/lib64 -lcublas -lcudart
CUDA_FLAGS = --compile -ccbin $(CUDA_HOST) -std=c++11 -arch=$(CUDA_ARCH) -O3 -lineinfo -w --resource-usage --ptxas-options=-v -Xcompiler -fPIC -D_FORCE_INLINES

LINK_FLAGS = -fPIC

OBJS = timer.o memory.o bla_lib.o main.o

$(BINARY_NAME): $(OBJS)
	$(CXX_COMP) $(OBJS) $(LINK_FLAGS) $(CXX_LIB) $(CUDA_LIB) -o $(BINARY_NAME)

timer.o: timer.cpp timer.hpp
	$(CXX_COMP) $(CXX_FLAGS) $(CXX_INC) $(CUDA_INC) timer.cpp

memory.o: memory.cpp memory.hpp
	$(CXX_COMP) $(CXX_FLAGS) $(CXX_INC) $(CUDA_INC) memory.cpp

bla_lib.o: bla_lib.cu bla_lib.hpp matrix.hpp memory.hpp timer.hpp
	$(CUDA_COMP) $(CUDA_FLAGS) $(CXX_INC) $(CUDA_INC) bla_lib.cu

main.o: main.cpp bla_lib.cu bla_lib.hpp memory.hpp
	$(CXX_COMP) $(CXX_FLAGS) $(CXX_INC) $(CUDA_INC) main.cpp


.PHONY: clean
clean:
	rm -f *.out *.x *.a *.so *.o *.mod *.modmic *.ptx *.log
