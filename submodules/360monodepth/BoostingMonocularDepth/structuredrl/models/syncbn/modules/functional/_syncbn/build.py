import os
from torch.utils.ffi import create_extension

sources = ['src/syncbn.cpp']
headers = ['src/syncbn.h']
extra_objects = ['src/syncbn.cu.o']
with_cuda = True

this_file = os.path.dirname(os.path.realpath(__file__))
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
    '_ext.syncbn',
    headers=headers,
    sources=sources,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects,
    extra_compile_args=["-std=c++11"]
)

if __name__ == '__main__':
    ffi.build()
