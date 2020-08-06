from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='inplaceShapeLoss',
    ext_modules=[
        CUDAExtension('inplaceShapeLoss_cuda', [
            'inplaceShapeLossc.cpp',
            'inplaceShapeLosscuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
