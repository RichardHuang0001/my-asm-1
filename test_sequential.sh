#!/bin/bash
# test_sequential.sh - 自动测试Sequential版本并生成CSV

echo "======================================"
echo "Sequential版本性能测试脚本"
echo "======================================"

# 编译sequential版本（统一用omp_get_wtime计时）
echo "编译 sobel.c ..."
gcc sobel.c -o sobel -lm -fopenmp -O3
if [ $? -ne 0 ]; then
    echo "编译失败！"
    exit 1
fi
echo "编译成功！"
echo ""

# 创建CSV文件并写入表头
CSV_FILE="sequential_times.csv"
echo "image_size,run1_time,run2_time,run3_time" > $CSV_FILE

# 测试的图像尺寸
SIZES=(256 1024 4000 16000)

echo "开始测试..."
echo ""

# 对每个尺寸进行测试
for SIZE in "${SIZES[@]}"; do
    echo "----------------------------------------"
    echo "测试图像尺寸: ${SIZE}x${SIZE}"
    echo "----------------------------------------"
    
    # 存储3次运行的时间
    times=()
    
    # 运行3次
    for RUN in 1 2 3; do
        echo "  第 $RUN 次运行..."
        
        # 运行程序并捕获输出
        output=$(./sobel -n $SIZE 2>&1)
        
        # 从输出中提取时间（格式: "Sequential版本执行时间: 0.123456 秒"）
        time=$(echo "$output" | grep "执行时间" | awk '{print $3}')
        
        if [ -z "$time" ]; then
            echo "    错误：无法获取时间！"
            time="0.0"
        else
            echo "    时间: $time 秒"
        fi
        
        times+=($time)
        
        # 稍微等待一下，让系统稳定
        sleep 0.5
    done
    
    # 写入CSV
    echo "$SIZE,${times[0]},${times[1]},${times[2]}" >> $CSV_FILE
    
    echo "  完成！"
    echo ""
done

echo "======================================"
echo "测试完成！"
echo "结果已保存到: $CSV_FILE"
echo "======================================"
echo ""

# 显示CSV内容
echo "生成的CSV内容："
cat $CSV_FILE
echo ""

# 计算统计信息
echo "性能统计："
echo "----------------------------------------"
awk -F',' 'NR>1 {
    size=$1
    avg=($2+$3+$4)/3
    printf "%-6s: 平均时间 = %.6f 秒\n", size, avg
}' $CSV_FILE