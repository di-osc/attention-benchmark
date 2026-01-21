# Attention Benchmark

这个项目用于基准测试不同注意力机制实现的性能，比较了 Flash Attention、Sage Attention 和 SPAS Sage2 Attention 三种实现的耗时和加速比。

## 特性

- 对比三种注意力机制实现的性能
- 支持不同序列长度的测试
- 自动生成加速比对比图表
- 支持 CUDA GPU 加速

## 依赖

- PyTorch
- flash-attn
- sageattention
- spas-sage-attn
- matplotlib
- numpy

## 使用方法

1. 确保安装了所有依赖项
2. 运行基准测试：

```bash
python bench_attention.py
```

程序会自动：
- 预热 GPU
- 测试不同序列长度下的性能
- 生成加速比对比图表
- 保存结果到 `results/` 目录

## 测试结果

以下是在 NVIDIA GeForce RTX 4090 上的测试结果：

![NVIDIA GeForce RTX 4090 加速比对比](results/NVIDIA%20GeForce%20RTX%204090.png)

图表显示了 Sage Attention 和 SPAS Sage2 Attention 相对于 Flash Attention 的加速比（越高越好，>1 表示比 Flash Attention 更快）。

## 许可证

MIT License