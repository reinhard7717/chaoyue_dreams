import torch
print(f"PyTorch 版本: {torch.__version__}") # 应该显示你安装的版本，如 2.7.0
# print(f"PyTorch 是否使用 CUDA 编译: {torch.backends.cudble}") # 更准确的检查，也可以用 torch.version.cuda
print(f"PyTorch 编译所用的 CUDA 版本: {torch.version.cuda}") # 应该显示 11.8
cuda_available = torch.cuda.is_available()
print(f"{cuda_available}")
if cuda_available:
    print(f"可用的 GPU 数量: {torch.cuda.device}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA 不可用。请检查驱动和 PyTorch 安装。")
    # 尝试获取
    try:
        torch.randn(1).cuda()
    except Exception as e:
        print(f"尝试在 CUDA 上分配张量时出错: {e}")

