import os.path
from cffi import FFI
ffibuilder = FFI()

proto = open("cfdutils.h", 'r').read()
proto = r"typedef _Bool bool;" + proto
ffibuilder.cdef(proto)
ffibuilder.set_source("cfdutils", proto, extra_objects=[os.path.abspath("./libcfdutils.so")])

if __name__ == "__main__":
	ffibuilder.compile(verbose=True)