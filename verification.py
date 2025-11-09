#!/usr/bin/env python3
# verification.py
# 对比 Sequential / OpenMP / MPI 版本的 Sobel 输出结果
# 不依赖任何外部包，仅使用 Python 标准库

import os
import math

def read_pgm(path):
    """读取二进制 PGM 文件为二维像素列表"""
    with open(path, "rb") as f:
        header = f.readline().strip()
        if header not in [b'P5', b'P2']:
            raise ValueError(f"{path}: Unsupported PGM format ({header})")

        # 读取注释行（可选）
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()

        # 读取尺寸
        width, height = [int(x) for x in line.split()]
        maxval = int(f.readline())
        pixels = []

        if header == b'P5':  # 二进制灰度
            data = f.read(width * height)
            pixels = list(data)
        else:  # ASCII 灰度 (P2)
            content = f.read().split()
            pixels = [int(x) for x in content]

        return width, height, pixels


def compare_images(base_path, test_path, tol=0.0):
    """比较两张 PGM 图像，返回 (是否一致, 最大误差, MSE, 差异像素数)"""
    try:
        w1, h1, p1 = read_pgm(base_path)
        w2, h2, p2 = read_pgm(test_path)
    except Exception as e:
        print(f"读取失败: {e}")
        return False, None, None, None

    if (w1, h1) != (w2, h2):
        print(f"尺寸不一致: {base_path} vs {test_path}")
        return False, None, None, None

    diff_pixels = 0
    max_err = 0.0
    mse_acc = 0.0
    total = len(p1)

    for i in range(total):
        diff = abs(p1[i] - p2[i])
        if diff > tol:
            diff_pixels += 1
        if diff > max_err:
            max_err = diff
        mse_acc += diff * diff

    mse = mse_acc / total if total > 0 else 0.0
    passed = (max_err <= tol)
    return passed, max_err, mse, diff_pixels


def main():
    sizes = [256, 1024, 4000, 16000]
    versions = ["omp", "mpi"]

    os.makedirs("report", exist_ok=True)
    csv_path = os.path.join("report", "verification.csv")

    with open(csv_path, "w") as f:
        f.write("sample,implementation,pass,max_abs_error,mse,num_diff_pixels\n")

        for size in sizes:
            base_file = f"out_{size}_sobel.pgm"
            if not os.path.exists(base_file):
                print(f"跳过: 缺少顺序版 {base_file}")
                continue

            for v in versions:
                test_file = f"out_{size}_sobel_{v}.pgm"
                if not os.path.exists(test_file):
                    print(f"跳过: 缺少 {test_file}")
                    continue

                ok, max_err, mse, diff_pixels = compare_images(base_file, test_file)
                f.write(f"{size}x{size},{v},{ok},{max_err:.6f},{mse:.6f},{diff_pixels}\n")

                result = "匹配" if ok else "不匹配"
                print(f"[{size}x{size}] {v.upper():>4} -> {result} (max_err={max_err:.6f}, diff={diff_pixels})")

    print("\nVerification 完成")
    print(f"结果已保存到: {csv_path}")


if __name__ == "__main__":
    main()