/*
 * sobel_omp.c — OpenMP并行版 Sobel 边缘检测实现
 * 
 * 构建:
 *   gcc sobel_omp.c -o sobel_omp -lm -fopenmp
 *   （依赖 math 库和 OpenMP 支持）
 * 依赖:
 *   pgmio.h 提供 PGM 读写接口（pgmread / pgmwrite）
 *
 * 用法:
 *   1) 显式指定输入/输出：
 *      ./sobel_omp <输入.pgm> <输出.pgm> [线程数]
 *      示例: ./sobel_omp sample_256.pgm out_256_sobel.pgm 4
 *   2) 通过样本尺寸自动选择文件：
 *      ./sobel_omp -n <尺寸> [线程数]
 *      支持尺寸: 256 / 1024 / 4000 / 16000
 *      文件映射: 256→sample_256.pgm, 1024→sample_1024.pgm,
 *               4000→sample_4k.pgm, 16000→sample_16k.pgm
 *      输出文件名将生成为 out_<尺寸>_sobel.pgm
 *
 * 说明:
 *   - 使用 OpenMP 实现并行化，支持多线程处理
 *   - 图像内存以一维 float 数组存储，按行主序索引
 *   - 包含预处理：3×3 均值滤波；核心处理：Sobel 梯度幅值计算
 *   - 输出写入 PGM（P5 二进制）格式，像素值范围 [0,255]
 *   - 支持动态线程数控制，默认使用系统最大线程数
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h> // 用来清空内存 (memset)
#include <math.h>   // 用来算平方根 (sqrtf)
#include <stddef.h> // 用于 size_t 类型（snprintf 缓冲区长度）
#include <omp.h>    // OpenMP 支持

// 图像读写接口（pgmread/pgmwrite）
#include "pgmio.h" 

#include <time.h>     // 用于 clock() 计时

// Sobel 算子 3×3 核定义
float Gx[3][3] = {
    {-1.0f, 0.0f, 1.0f},
    {-2.0f, 0.0f, 2.0f},
    {-1.0f, 0.0f, 1.0f}
};

float Gy[3][3] = {
    {-1.0f, -2.0f, -1.0f},
    { 0.0f,  0.0f,  0.0f},
    { 1.0f,  2.0f,  1.0f}
};

/*
 * 根据样本尺寸生成默认输入/输出文件路径
 * 支持尺寸: 256 / 1024 / 4000 / 16000
 * 输入: size — 样本尺寸（像素数的一边）
 * 输出: in_path / out_path — 写入生成的文件名（缓冲区由调用者提供）
 * 返回: 0 成功；-1 尺寸不支持
 */
static int build_paths_for_size(int size, char *in_path, size_t in_len, char *out_path, size_t out_len) {
    if (size == 256) {
        snprintf(in_path, in_len, "sample_256.pgm");
        snprintf(out_path, out_len, "out_256_sobel.pgm");
        return 0;
    } else if (size == 1024) {
        snprintf(in_path, in_len, "sample_1024.pgm");
        snprintf(out_path, out_len, "out_1024_sobel.pgm");
        return 0;
    } else if (size == 4000) {
        snprintf(in_path, in_len, "sample_4k.pgm");
        snprintf(out_path, out_len, "out_4k_sobel.pgm");
        return 0;
    } else if (size == 16000) {
        snprintf(in_path, in_len, "sample_16k.pgm");
        snprintf(out_path, out_len, "out_16k_sobel.pgm");
        return 0;
    }
    return -1;
}

/*
 * OpenMP并行化 3×3 均值滤波
 * 输入:  in_image — 源图像（float，行主序，一维）
 * 输出:  out_image — 均值滤波结果（float）
 * 参数:  width（列数，cols）/ height（行数，rows）
 * 说明:  使用 OpenMP 并行化内部像素处理，边界像素置 0
 */
void blur_filter_omp(const float *in_image, float *out_image, int width, int height) {
    printf("开始模糊处理（OpenMP并行）...\n");
    
    // 并行化外层循环，动态调度以平衡负载
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            
            float sum = 0.0f; // 累积邻域像素值
            
            // 访问 3×3 邻域并求和
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    // 线性索引: idx = (y + ky) * width + (x + kx)
                    sum += in_image[(y + ky) * width + (x + kx)];
                }
            }
            
            // 写入邻域平均值（9 像素）
            out_image[y * width + x] = sum / 9.0f;
        }
    }
    
    // 边界处理：置 0（简化实现，避免越界访问）
    // 使用并行化处理边界
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        out_image[y * width] = 0.0f;                  // 最左列
        out_image[y * width + (width - 1)] = 0.0f;    // 最右列
    }
    
    #pragma omp parallel for
    for (int x = 0; x < width; x++) {
        out_image[x] = 0.0f;                          // 最上行
        out_image[(height - 1) * width + x] = 0.0f;   // 最下行
    }
}

/*
 * OpenMP并行化 Sobel 边缘检测
 * 输入:  in_image — 源图像（float）
 * 输出:  out_image — 梯度幅值结果（float）
 * 参数:  width（列数，cols）/ height（行数，rows）
 * 说明:  使用 OpenMP 并行化 Sobel 计算
 */
void sobel_filter_omp(const float *in_image, float *out_image, int width, int height) {
    printf("开始 Sobel 边缘检测（OpenMP并行）...\n");
    
    // 并行化外层循环，动态调度
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            
            float sum_x = 0.0f;
            float sum_y = 0.0f;
            
            // 3×3 核卷积累积 Gx/Gy
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    
                    // 邻域像素值
                    float pixel_val = in_image[(y + ky) * width + (x + kx)];
                    
                    // 累积与核权重乘积
                    sum_x += pixel_val * Gx[ky + 1][kx + 1];
                    sum_y += pixel_val * Gy[ky + 1][kx + 1];
                }
            }
            
            // 梯度幅值
            float magnitude = sqrtf(sum_x * sum_x + sum_y * sum_y);
            
            // 裁剪到 [0,255]
            if (magnitude > 255.0f) {
                magnitude = 255.0f;
            }
            if (magnitude < 0.0f) {
                magnitude = 0.0f;
            }
            
            out_image[y * width + x] = magnitude;
        }
    }
    
    // 边界处理：置 0（并行化）
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        out_image[y * width] = 0.0f;                  // 最左列
        out_image[y * width + (width - 1)] = 0.0f;    // 最右列
    }
    
    #pragma omp parallel for
    for (int x = 0; x < width; x++) {
        out_image[x] = 0.0f;                          // 最上行
        out_image[(height - 1) * width + x] = 0.0f;   // 最下行
    }
}

/*
 * 程序入口：参数解析、PGM 读写、OpenMP并行处理、结果输出
 */
int main(int argc, char *argv[]) {
    // 设置默认线程数
    int num_threads = omp_get_max_threads();
    char input_file_buf[128] = {0};   // 默认输入文件
    char output_file_buf[128] = {0};  // 默认输出文件
    char *input_file = NULL;
    char *output_file = NULL;

    // 计时器变量
    double start_time, end_time;
    double cpu_time_used;

    // 解析参数
    if (argc >= 3 && strcmp(argv[1], "-n") == 0) {
        // 使用样本尺寸参数: ./sobel_omp -n <尺寸> [线程数]
        int size = atoi(argv[2]);
        if (build_paths_for_size(size, input_file_buf, sizeof(input_file_buf), 
                                output_file_buf, sizeof(output_file_buf)) != 0) {
            fprintf(stderr, "尺寸不支持: %d（支持 256/1024/4000/16000）\n", size);
            return 1;
        }
        input_file = input_file_buf;
        output_file = output_file_buf;
        
        // 解析线程数参数
        if (argc >= 4) {
            num_threads = atoi(argv[3]);
            if (num_threads <= 0) {
                num_threads = omp_get_max_threads();
            }
        }
        
        printf("使用尺寸参数 -n %d，输入: %s，输出: %s，线程数: %d\n", 
               size, input_file, output_file, num_threads);
        
    } else if (argc >= 3) {
        // 传统用法: ./sobel_omp <输入> <输出> [线程数]
        input_file = argv[1];
        output_file = argv[2];
        
        // 解析线程数参数
        if (argc >= 4) {
            num_threads = atoi(argv[3]);
            if (num_threads <= 0) {
                num_threads = omp_get_max_threads();
            }
        }
        
        printf("输入: %s，输出: %s，线程数: %d\n", input_file, output_file, num_threads);
        
    } else {
        fprintf(stderr, "用法: %s <输入图片.pgm> <输出图片.pgm> [线程数]\n", argv[0]);
        fprintf(stderr, "     或: %s -n <样本尺寸> [线程数]    （支持 256/1024/4000/16000）\n", argv[0]);
        fprintf(stderr, "     示例: %s sample_256.pgm output.pgm 4\n", argv[0]);
        fprintf(stderr, "     示例: %s -n 256 4\n", argv[0]);
        return 1;
    }

    // 设置OpenMP线程数
    omp_set_num_threads(num_threads);
    printf("设置OpenMP线程数: %d\n", num_threads);

    // 缓冲区与尺寸声明
    float *image_buffer; // 原始图像
    float *blur_buffer;  // 均值滤波结果
    float *sobel_buffer; // Sobel 幅值结果
    int rows, cols;      // 行（rows）、列（cols）

    // 读取 PGM 图像
    printf("读取图片: %s\n", input_file);
    if (pgmread(input_file, &image_buffer, &rows, &cols) != 0) {
        fprintf(stderr, "读取图片失败: %s\n", input_file);
        return 1;
    }
    printf("图片尺寸: %d x %d (行 x 列)\n", rows, cols);

    // 分配结果缓冲区
    blur_buffer = (float *)malloc(rows * cols * sizeof(float));
    sobel_buffer = (float *)malloc(rows * cols * sizeof(float));
    if (blur_buffer == NULL || sobel_buffer == NULL) {
        fprintf(stderr, "内存分配失败\n");
        return 1;
    }
    
    // 初始化缓冲区为 0
    memset(blur_buffer, 0, rows * cols * sizeof(float));
    memset(sobel_buffer, 0, rows * cols * sizeof(float));

    // 开始计时
    start_time = omp_get_wtime();

    // 预处理：3×3 均值滤波（OpenMP并行）
    blur_filter_omp(image_buffer, blur_buffer, cols, rows);
    
    // 核心处理：Sobel 边缘检测（OpenMP并行）
    sobel_filter_omp(blur_buffer, sobel_buffer, cols, rows);

    // 结束计时
    end_time = omp_get_wtime();
    cpu_time_used = end_time - start_time;

    // 写入结果为 PGM
    printf("写入图片: %s\n", output_file);
    if (pgmwrite(output_file, sobel_buffer, rows, cols, 1) != 0) {
        fprintf(stderr, "写入图片失败: %s\n", output_file);
    }
    
    printf("完成!\n");
    printf("OpenMP版本 (sobel_omp.c) 执行时间: %.6f 秒\n", cpu_time_used);
    printf("使用线程数: %d\n", num_threads);

    // 释放分配资源
    free(image_buffer);
    free(blur_buffer);
    free(sobel_buffer);

    return 0;
}