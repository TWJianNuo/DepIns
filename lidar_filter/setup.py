from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='lidar_filter',
    ext_modules=[
        CUDAExtension('bnmorph_getcorpts', [
            'lidar_filter.cpp',
            'lidar_filter.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
