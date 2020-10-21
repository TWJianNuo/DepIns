from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='shapeintegration_cuda',
    ext_modules=[
        CUDAExtension('shapeintegration_cuda', [
            'integration.cpp',
            'integration_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })