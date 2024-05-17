from setuptools import setup, Extension
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='weighted_sum_update_cpu',
    ext_modules=[
        CppExtension(
            name='weighted_sum_update_cpu',
            sources=['weighted_sum_update_cpu.cpp'],
            include_dirs=[
                '/Users/nikny/miniconda3/envs/musicai/lib/python3.8/site-packages/torch/include',
                '/Users/nikny/miniconda3/envs/musicai/lib/python3.8/site-packages/torch/include/torch/csrc/api/include',
                '/Users/nikny/miniconda3/envs/musicai/lib/python3.8/site-packages/torch/include/ATen',
                '/Users/nikny/miniconda3/envs/musicai/lib/python3.8/site-packages/torch/include/c10',
                '/Users/nikny/miniconda3/envs/musicai/lib/python3.8/site-packages/torch/include/caffe2',
                '/Users/nikny/miniconda3/envs/musicai/include/python3.8'
            ],
            extra_compile_args=['-std=c++17'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
