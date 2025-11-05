#!/bin/bash
# quick_test.sh - 快速手动测试，诊断问题

echo "========================================"
echo "快速测试脚本 - 诊断时间提取问题"
echo "========================================"
echo ""

echo "第1步：检查程序是否存在"
echo "----------------------------------------"
if [ -f "./sobel" ]; then
    echo "[OK] ./sobel 存在"
else
    echo "[X] ./sobel 不存在"
    echo "    运行: gcc sobel.c -o sobel -lm -fopenmp -O3"
fi

if [ -f "./sobel_omp" ]; then
    echo "[OK] ./sobel_omp 存在"
else
    echo "[X] ./sobel_omp 不存在"
    if [ -f "sobel_omp.c" ]; then
        echo "    运行: gcc sobel_omp.c -o sobel_omp -lm -fopenmp -O3"
    elif [ -f "sobel_omp_tiled.c" ]; then
        echo "    运行: gcc sobel_omp_tiled.c -o sobel_omp -lm -fopenmp -O3"
    fi
fi
echo ""

echo "第2步：检查图像文件"
echo "----------------------------------------"
for img in sample_256.pgm sample_1024.pgm sample_4k.pgm sample_16k.pgm; do
    if [ -f "$img" ]; then
        size=$(stat -f%z "$img" 2>/dev/null || stat -c%s "$img" 2>/dev/null)
        echo "[OK] $img ($(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo $size bytes))"
    else
        echo "[X] $img 不存在"
    fi
done
echo ""

if [ ! -f "./sobel" ]; then
    echo "sobel程序不存在，无法继续测试"
    exit 1
fi

echo "第3步：运行一次测试（256x256）"
echo "----------------------------------------"
echo "完整输出："
echo ""
./sobel -n 256
echo ""

echo "第4步：尝试提取时间"
echo "----------------------------------------"
output=$(./sobel -n 256 2>&1)

echo "方法1: grep '执行时间'"
echo "$output" | grep "执行时间"
echo ""

echo "方法2: grep '执行时间' + 提取数字"
time1=$(echo "$output" | grep "执行时间" | grep -oE '[0-9]+\.[0-9]+')
echo "提取结果: [$time1]"
echo ""

echo "方法3: 查找所有包含'秒'的行"
echo "$output" | grep "秒"
echo ""

echo "方法4: 提取所有'数字.数字'模式"
echo "$output" | grep -oE '[0-9]+\.[0-9]+'
echo ""

echo "第5步：分析"
echo "----------------------------------------"
if [ -z "$time1" ]; then
    echo "[问题] 无法提取时间！"
    echo ""
    echo "可能的原因："
    echo "1. 程序输出格式与脚本预期不符"
    echo "2. 程序没有输出执行时间"
    echo "3. grep命令不支持中文匹配"
    echo ""
    echo "解决方法："
    echo "1. 检查上面'第3步'的完整输出"
    echo "2. 找到包含时间的那一行"
    echo "3. 告诉我那一行的格式（比如'Sequential版本执行时间: 0.001234 秒'）"
    echo "4. 我会帮你修改脚本的提取逻辑"
else
    echo "[成功] 提取到时间: $time1 秒"
    echo ""
    echo "如果这个值看起来正常，说明时间提取没问题。"
    echo "如果sequential_times.csv里还是空的，可能是：" 
    echo "1. 脚本没有正确保存文件"
    echo "2. 权限问题"
    echo "3. 运行的是旧版本脚本"
    echo ""
    echo "建议："
    echo "1. 使用改进版脚本: ./test_sequential_improved.sh"
    echo "2. 查看脚本输出的调试信息"
fi
echo ""

echo "第6步：如果OpenMP也有问题"
echo "----------------------------------------"
if [ -f "./sobel_omp" ]; then
    echo "运行OpenMP测试（1线程）："
    ./sobel_omp -n 256 1
    echo ""
    echo "如果看到执行时间，说明程序工作正常"
else
    echo "sobel_omp不存在，跳过"
fi
echo ""

echo "========================================"
echo "测试完成！"
echo "========================================"
echo ""
echo "下一步："
echo "  如果看到了时间 → 运行: ./test_sequential_improved.sh"
echo "  如果没看到时间 → 把'第3步'的输出发给我"
