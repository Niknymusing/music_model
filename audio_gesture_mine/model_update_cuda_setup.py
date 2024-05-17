from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='weighted_sum_update',
    ext_modules=[
        CUDAExtension('weighted_sum_update', [
            'cuda/weighted_sum_update.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
