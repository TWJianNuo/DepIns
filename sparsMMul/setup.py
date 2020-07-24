from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sparsMMul',
    ext_modules=[
        CUDAExtension('sparsMMul_cuda', [
            'sparsMMulc.cpp',
            'sparsMMulcu.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
