## Introduction
This is a learning note of cuda programming for novices.

### Python build
Prefer using setup tool to install python package.
```bash
cd CUDATutorial/python
python setup.py install
```
For windows users, we have to `import torch` before we `import add2`. Or, it will throw:
```bash
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ImportError: DLL load failed:
```

### References
- [cuda-c-programming-guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [NN-CUDA-Example](https://github.com/godweiyang/NN-CUDA-Example)
- [cuda_programming](https://github.com/CoffeeBeforeArch/cuda_programming)
