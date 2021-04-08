CXX_FLAGS := -flto -O3 -std=c++11 -Wall -Wno-switch
LD_FLAGS := -flto -lm -lfmt -fPIC

build-cffi:
	echo "Building CFFI module..."
	python3 cfdutils-cffi-build.py
	rm *.o
	rm cfdutils.c

cfdutils: cfdutils.cpp
	clang++ $(CXX_FLAGS) cfdutils.cpp $(LD_FLAGS) -shared -o libcfdutils.so
	make build-cffi
