# 项目环境与 MinkowskiEngine

本次环境只在内层 `.venv/` 创建。系统及已有 `.venvs/dexgrasp` 环境未修改；`.venv` 通过 `--system-site-packages` 只读复用原有 PyTorch/CUDA，h5py 和 MinkowskiEngine 安装在本项目环境中。

本机检查组合：Python 3.12、PyTorch `2.7.0a0+7c8ec84dab.nv25.03`、CUDA 12.8、GCC 13、RTX 5090（SM120）、NumPy 1.26.4、h5py 3.16.0、SciPy 1.15.2。版本记录不是对其他组合的兼容性保证。

## 原生扩展来源

官方 [NVIDIA/MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) 的 `02fc608bea4c0549b0a7b00ca1bf15dee4a0b228`，版本 0.5.4。CPU 构建不需要兼容补丁；当前 CUDA/GCC 组合需要 `scripts/patches/minkowski-cuda12.patch`：

- 显式包含当前 Thrust 所需的 execution_policy、sort、remove、reduce 等头文件。
- 存在新版 NVTX 时使用系统提供的 C++ API 与 scoped_range 类型，避免旧 bundled 定义冲突。
- 保留哈希表自定义 deleter，以显式转移所有权避开 GCC13/CUDA 的 unique_ptr→shared_ptr ADL 歧义。

补丁只作用于本项目的依赖源码副本，没有修改外层 reference，也没有改系统头文件。来源及许可证见 `THIRD_PARTY_NOTICES.md`。

## 本机可复现构建

以下命令在内层仓库执行，假设当前系统已有匹配 PyTorch、CUDA toolkit、C++ 编译器和 OpenBLAS 开发文件。其他机器先准备自己的隔离基础环境，架构参数按实际 GPU 修改。

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-deps h5py==3.16.0

git clone https://github.com/NVIDIA/MinkowskiEngine.git .venv/src/MinkowskiEngine
git -C .venv/src/MinkowskiEngine checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
git -C .venv/src/MinkowskiEngine apply ../../../scripts/patches/minkowski-cuda12.patch

# 上游 setup.py 会调用 pip；PATH 明确指向项目虚拟环境。
cd .venv/src/MinkowskiEngine
PATH="$(pwd)/../../bin:$PATH" MAX_JOBS=4 OMP_NUM_THREADS=4 TORCH_CUDA_ARCH_LIST=12.0 \
  ../../bin/python setup.py bdist_wheel --dist-dir dist-cuda --force_cuda --blas=openblas
cd ../../..
.venv/bin/python -m pip install --no-deps .venv/src/MinkowskiEngine/dist-cuda/*.whl
```

需要 CPU 构建时使用 `--cpu_only --blas=openblas`，不要加 `--force_cuda`。CPU-only 扩展无法执行配置中的 `device=cuda:0`，plan 会明确报出。已有源码/已应用补丁时不要重复 clone/apply。

`MinkowskiEngine.is_cuda_available()` 检查编译后端，`torch.cuda.is_available()` 检查 PyTorch 设备可用性，两者都要满足。运行记录会保存原生扩展 `.so` 的 SHA256；恢复时扩展或依赖变化会被拒绝。
