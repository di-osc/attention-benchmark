from flash_attn import flash_attn_func
from sageattention import sageattn
from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda
import torch
import time
import matplotlib.pyplot as plt
import numpy as np


def bench(lens):
    """
    对比不同长度的序列的几个attention的耗时

    Args:
        lens: 序列长度列表

    Returns:
        results: 包含长度和对应耗时数据的字典
    """
    results = {
        "lengths": lens,
        "flash_attn_times": [],
        "sageattn_times": [],
        "spas_sage2_attn_times": [],
    }

    print("开始正式测试...\n")

    for length in lens:
        print(f"Testing sequence length: {length}")

        # 为当前长度预热
        print(f"为长度 {length} 预热中...")
        q_warmup = torch.randn(1, length, 40, 128).to(
            dtype=torch.bfloat16, device="cuda"
        )
        k_warmup = torch.randn(1, length, 40, 128).to(
            dtype=torch.bfloat16, device="cuda"
        )
        v_warmup = torch.randn(1, length, 40, 128).to(
            dtype=torch.bfloat16, device="cuda"
        )

        # 预热Flash Attention
        for _ in range(3):
            _ = flash_attn_func(q_warmup, k_warmup, v_warmup)
        torch.cuda.synchronize()

        # 预热Sage Attention
        for _ in range(3):
            _ = sageattn(q_warmup, k_warmup, v_warmup, tensor_layout="NHD")
        torch.cuda.synchronize()

        # 预热Spas Sage2 Attention
        for _ in range(3):
            _ = spas_sage2_attn_meansim_topk_cuda(
                q_warmup,
                k_warmup,
                v_warmup,
                topk=0.5,
                output_dtype=torch.bfloat16,
                tensor_layout="NHD",
            )
        torch.cuda.synchronize()

        # 预热torch.nn.functional.scaled_dot_product_attention
        for _ in range(3):
            _ = torch.nn.functional.scaled_dot_product_attention(
                q_warmup, k_warmup, v_warmup
            )
        torch.cuda.synchronize()

        print(f"为长度 {length} 预热完成，开始测试...")

        # 创建测试用张量
        q = torch.randn(1, length, 40, 128).to(dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, length, 40, 128).to(dtype=torch.bfloat16, device="cuda")
        v = torch.randn(1, length, 40, 128).to(dtype=torch.bfloat16, device="cuda")
        torch.cuda.synchronize()

        # Test flash_attn_func
        start = time.perf_counter()
        for _ in range(2):
            _ = flash_attn_func(q, k, v)
        torch.cuda.synchronize()
        end = time.perf_counter()
        flash_time = end - start
        results["flash_attn_times"].append(flash_time)
        print(f"Length {length}: flash_attn_func time: {flash_time:.4f} seconds")

        # Test sageattn
        start = time.perf_counter()
        for _ in range(2):
            _ = sageattn(q, k, v, tensor_layout="NHD")
        torch.cuda.synchronize()
        end = time.perf_counter()
        sage_time = end - start
        results["sageattn_times"].append(sage_time)
        print(f"Length {length}: sageattn time: {sage_time:.4f} seconds")

        # Test spas_sage2_attn_meansim_topk_cuda
        start = time.perf_counter()
        for _ in range(2):
            _ = spas_sage2_attn_meansim_topk_cuda(
                q, k, v, topk=0.5, output_dtype=torch.bfloat16, tensor_layout="NHD"
            )
        torch.cuda.synchronize()
        end = time.perf_counter()
        spas_time = end - start
        results["spas_sage2_attn_times"].append(spas_time)
        print(
            f"Length {length}: spas_sage2_attn_meansim_topk_cuda time: {spas_time:.4f} seconds"
        )

    return results


def plot_benchmark_results(results):
    """
    将基准测试结果绘制为加速比柱状图（相对于Flash Attention）

    Args:
        results: bench函数返回的结果字典
    """
    # 获取GPU型号
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown GPU"
    lengths = results["lengths"]
    flash_times = np.array(results["flash_attn_times"])
    sage_times = np.array(results["sageattn_times"])
    spas_times = np.array(results["spas_sage2_attn_times"])

    # 计算加速比（相对于Flash Attention）
    # speedup = baseline_time / current_time, >1表示更快，<1表示更慢
    flash_speedup = flash_times / flash_times  # 基准为1
    sage_speedup = flash_times / sage_times
    spas_speedup = flash_times / spas_times

    # 设置柱状图的位置
    x = np.arange(len(lengths))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 8))

    # 创建柱状图
    bars1 = ax.bar(
        x - 1.5 * width,
        flash_speedup,
        width,
        label="Flash Attention",
        alpha=0.8,
        color="skyblue",
    )
    bars2 = ax.bar(
        x - 0.5 * width,
        sage_speedup,
        width,
        label="Sage Attention2++",
        alpha=0.8,
        color="lightgreen",
    )
    bars3 = ax.bar(
        x + 0.5 * width,
        spas_speedup,
        width,
        label="Spas Sage2 Attention",
        alpha=0.8,
        color="salmon",
    )

    # 添加参考线
    ax.axhline(
        y=1.0,
        color="red",
        linestyle="--",
        alpha=0.7,
        label="Baseline (Flash Attention)",
    )

    # 添加标签和标题
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Speedup Ratio (relative to Flash Attention)")
    ax.set_title(
        f"Attention Methods Speedup Comparison\nGPU: {gpu_name}\n(Higher is better, >1 means faster than Flash Attention)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{length}" for length in lengths])
    ax.legend()

    # 设置y轴范围，从0开始
    y_max = (
        max(
            max(flash_speedup), max(sage_speedup), max(spas_speedup)
        )
        * 1.1
    )
    ax.set_ylim(0, y_max)

    # 在柱子上添加数值标签
    def add_value_labels(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.2f}x",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    add_value_labels(bars1, flash_speedup)
    add_value_labels(bars2, sage_speedup)
    add_value_labels(bars3, spas_speedup)

    plt.tight_layout()
    plt.savefig(f"results/{gpu_name}.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # 测试不同序列长度
    sequence_lengths = [2 ** i for i in range(7, 16)]

    print("开始基准测试...")
    results = bench(sequence_lengths)

    print("\n绘制加速比图表...")
    plot_benchmark_results(results)

    print("基准测试完成！加速比图表已保存为 attention_benchmark_speedup.png")
