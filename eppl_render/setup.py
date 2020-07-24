from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='eppl',
    ext_modules=[
        CUDAExtension('eppl_cuda', [
            'epplc.cpp',
            'epplcuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })