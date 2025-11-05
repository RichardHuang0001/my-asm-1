#!/usr/bin/env python3

"""
CMSC5702 Assignment 1 - Part 1 & 2 自动化测试脚本

功能:
1. 使用 -O3 优化编译 sobel.c 和 sobel_omp.c。
2. 运行 sobel (Sequential) 3次，测试所有尺寸。
3. 运行 sobel_omp (OpenMP) 3次，测试所有尺寸和所有线程。
4. 自动解析程序的标准输出，抓取执行时间。
5. 自动生成 'report/sequential_times.csv' 和 'report/openmp_times.csv'。

用法:
1. 确保此脚本与 sobel.c, sobel_omp.c, pgmio.h 在同一目录。
2. 确保 'report' 目录已存在。
3. 运行: python3 run_tests.py
"""

import subprocess
import os
import re
import sys

# --- 测试参数 (根据作业要求配置) ---
# 1. 图像尺寸
SIZES = [256, 1024, 4000, 16000]
# 2. 线程数 (根据老师邮件更新)
THREADS = [1, 2, 4, 8, 16, 32]
# 3. 每次测试重复运行的次数
REPETITIONS = 3
# 4. 报告目录
REPORT_DIR = "report"
# 5. CSV 文件路径
SEQ_CSV_PATH = os.path.join(REPORT_DIR, "sequential_times.csv")
OMP_CSV_PATH = os.path.join(REPORT_DIR, "openmp_times.csv")
# 6. 编译优化等级
OPT_LEVEL = "-O3"

# --- 辅助函数：用于运行和解析时间 ---
def run_and_parse_time(command_args):
    """
    运行一个命令，捕获其stdout，并用正则表达式解析执行时间。
    返回: (str) 时间的字符串, 或 "ERROR"
    """
    try:
        # 【已修复】: 将 Python 3.7+ 的参数 (capture_output=True, text=True)
        #           替换为 Python 3.6 兼容的参数。
        result = subprocess.run(
            command_args,
            stdout=subprocess.PIPE,       # 替换 capture_output=True
            stderr=subprocess.PIPE,       # 替换 capture_output=True
            universal_newlines=True,  # 替换 text=True
            encoding='utf-8',
            check=True
        )
        
        # 用正则表达式查找 "执行时间: X.XXXXXX 秒"
        match = re.search(r"执行时间: (\d+\.\d+) 秒", result.stdout)
        
        if match:
            # 成功！返回抓取到的时间 (第1组)
            return match.group(1)
        else:
            # 打印整个输出以便调试
            print(f"  [错误] 未能在输出中找到时间。程序输出: \n{result.stdout}\n{result.stderr}")
            return "PARSE_ERROR"
            
    except subprocess.CalledProcessError as e:
        print(f"  [错误] 命令执行失败: {' '.join(command_args)}")
        print(f"  [错误] {e.stderr}")
        return "RUN_ERROR"
    except FileNotFoundError:
        print(f"  [错误] 命令未找到: {command_args[0]} (是否已编译?)")
        return "NOT_FOUND"

# --- 主函数 ---
def main():
    
    # --- 1. 编译 ---
    print(f"--- 1. 正在编译 (优化等级: {OPT_LEVEL}) ---")
    try:
        # 编译 sobel.c (Sequential)
        # (它也需要 -fopenmp 因为你用了 omp_get_wtime()
        print("  编译: sobel.c ...")
        compile_seq_cmd = [
            "gcc", "sobel.c", "-o", "sobel",
            "-lm", "-fopenmp", OPT_LEVEL
        ]
        subprocess.run(compile_seq_cmd, check=True)
        print("  [✓] sobel (Sequential) 编译成功")

        # 编译 sobel_omp.c (OpenMP Tiled)
        print("  编译: sobel_omp.c ...")
        compile_omp_cmd = [
            "gcc", "sobel_omp.c", "-o", "sobel_omp",
            "-lm", "-fopenmp", OPT_LEVEL
        ]
        subprocess.run(compile_omp_cmd, check=True)
        print("  [✓] sobel_omp (OpenMP Tiled) 编译成功")
        
    except Exception as e:
        print(f"\n[!!] 编译失败，脚本终止: {e}")
        sys.exit(1)

    # 确保 report 目录存在
    os.makedirs(REPORT_DIR, exist_ok=True)

    # --- 2. 运行 Part 1 (Sequential) ---
    print(f"\n--- 2. 正在运行 Part 1 (Sequential) 测试 ---")
    try:
        with open(SEQ_CSV_PATH, "w") as f:
            f.write("image_size,run1_time,run2_time,run3_time\n")
            
            for size in SIZES:
                print(f"  测试: Sequential, 尺寸: {size}...")
                times = []
                for _ in range(REPETITIONS):
                    cmd = ["./sobel", "-n", str(size)]
                    time_str = run_and_parse_time(cmd)
                    times.append(time_str)
                
                # 写入CSV
                f.write(f"{size},{','.join(times)}\n")
        print(f"  [✓] Part 1 测试完成! -> {SEQ_CSV_PATH}")
    except Exception as e:
        print(f"\n[!!] Part 1 测试失败: {e}")
        sys.exit(1)

    # --- 3. 运行 Part 2 (OpenMP) ---
    print(f"\n--- 3. 正在运行 Part 2 (OpenMP) 测试 ---")
    try:
        with open(OMP_CSV_PATH, "w") as f:
            f.write("image_size,num_threads,run1_time,run2_time,run3_time\n")
            
            for size in SIZES:
                for threads in THREADS:
                    print(f"  测试: OpenMP, 尺寸: {size}, 线程: {threads}...")
                    times = []
                    for _ in range(REPETITIONS):
                        cmd = ["./sobel_omp", "-n", str(size), str(threads)]
                        time_str = run_and_parse_time(cmd)
                        times.append(time_str)
                    
                    # 写入CSV
                    f.write(f"{size},{threads},{','.join(times)}\n")
        print(f"  [✓] Part 2 测试完成! -> {OMP_CSV_PATH}")
    except Exception as e:
        print(f"\n[!!] Part 2 测试失败: {e}")
        sys.exit(1)

    # --- 4. 最终总结 ---
    print("\n--- 所有测试已完成! ---")
    print(f"\nSequential 结果 ({SEQ_CSV_PATH}):")
    subprocess.run(["cat", SEQ_CSV_PATH])
    print(f"\nOpenMP 结果 ({OMP_CSV_PATH}):")
    subprocess.run(["cat", OMP_CSV_PATH])

# --- 脚本入口 ---
if __name__ == "__main__":
    main()

