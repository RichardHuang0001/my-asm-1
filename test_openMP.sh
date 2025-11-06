#!/bin/bash
# test_openmp.sh - 自动测试OpenMP版本并生成CSV

echo "======================================"
echo "OpenMP版本性能测试脚本"
echo "======================================"

# 编译OpenMP版本
echo "编译 sobel_omp_tiled.c ..."
gcc sobel_omp_tiled.c -o sobel_omp -lm -fopenmp -O3
if [ $? -ne 0 ]; then
    echo "编译失败！"
    exit 1
fi
echo "编译成功！"
echo ""

# 创建CSV文件并写入表头
CSV_FILE="report/openmp_times.csv"
echo "image_size,threads,run1_time,run2_time,run3_time" > $CSV_FILE

# 测试的图像尺寸和线程数
SIZES=(256 1024 4000 16000)
THREADS=(1 2 4 8 16 32)

echo "开始测试..."
echo "（这可能需要较长时间，特别是16000x16000的大图像）"
echo ""

# 对每个尺寸和线程数组合进行测试
for SIZE in "${SIZES[@]}"; do
    echo "========================================"
    echo "图像尺寸: ${SIZE}x${SIZE}"
    echo "========================================"
    
    for THREAD in "${THREADS[@]}"; do
        echo "  线程数: $THREAD"
        
        # 存储3次运行的时间
        times=()
        
        # 运行3次
        for RUN in 1 2 3; do
            echo -n "    第 $RUN 次运行... "
            
            # 运行程序并捕获输出
            output=$(./sobel_omp -n $SIZE $THREAD 2>&1)
            
            # 从输出中提取时间
            time=$(echo "$output" | grep "执行时间" | awk '{print $3}')
            
            if [ -z "$time" ]; then
                echo "错误：无法获取时间！"
                time="0.0"
            else
                echo "时间: $time 秒"
            fi
            
            times+=($time)
            
            # 短暂等待
            sleep 0.2
        done
        
        # 写入CSV
        echo "$SIZE,$THREAD,${times[0]},${times[1]},${times[2]}" >> $CSV_FILE
        
        # 计算平均时间
        avg=$(echo "${times[@]}" | awk '{print ($1+$2+$3)/3}')
        echo "    平均: $avg 秒"
        echo ""
    done
    echo ""
done

echo "======================================"
echo "测试完成！"
echo "结果已保存到: $CSV_FILE"
echo "======================================"
echo ""

# 显示CSV部分内容（前10行）
echo "CSV前几行内容："
head -15 $CSV_FILE
echo "..."
echo ""

# 生成性能摘要
echo "性能摘要："
echo "----------------------------------------"
awk -F',' 'NR>1 {
    key=$1 "_" $2
    avg=($3+$4+$5)/3
    data[key]=avg
    sizes[$1]=1
    threads[$2]=1
}
END {
    # 按尺寸分组
    for (size in sizes) {
        print "\n" size "x" size ":"
        for (t=1; t<=32; t*=2) {
            key=size "_" t
            if (key in data) {
                printf "  %2d 线程: %.6f 秒", t, data[key]
                if (t>1) {
                    key1=size "_1"
                    speedup=data[key1]/data[key]
                    printf " (加速比: %.2fx)", speedup
                }
                print ""
            }
        }
    }
}' $CSV_FILE

echo ""
echo "----------------------------------------"
echo "提示："
echo "1. 查看完整CSV: cat $CSV_FILE"
echo "2. 加速比 = 1线程时间 / N线程时间"
echo "3. 理想情况下，N个线程应该有接近N倍的加速"
echo "4. 如果加速比远小于线程数，可能存在：" 
echo "   - 内存带宽瓶颈"
echo "   - 同步开销"
echo "   - 负载不均衡"