from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="add2",
    include_dirs=["."],
    ext_modules=[CUDAExtension(
        "add2",
        ["add2.cpp", "add2_kernel.cu"],
    )],
    cmdclass={"build_ext": BuildExtension})
