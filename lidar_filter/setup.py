from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='LidarFilterCuda',
    ext_modules=[
        CUDAExtension('LidarFilterCuda', [
            'lidar_filter.cpp',
            'lidar_filter_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
