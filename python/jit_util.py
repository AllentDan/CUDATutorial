from torch.utils.cpp_extension import load

cuda_module = load(
    name="add_my",
    extra_include_paths=["."],
    sources=["add2.cpp", "add2_kernel.cu"],
    verbose=True)

cuda_module.torch_launch_add2
