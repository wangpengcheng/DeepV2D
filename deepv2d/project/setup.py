from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension,CppExtension

setup(
    name='backproject',
    ext_modules=[
        # CppExtension('backproject', [
        #     'backproject_op.cpp'
        # ]),
        CUDAExtension('backproject', [
            'backproject_op.cpp',
            'backproject_op_gpu.cu'
        ])

    ],
    cmdclass={
        'build_ext': BuildExtension
    })
