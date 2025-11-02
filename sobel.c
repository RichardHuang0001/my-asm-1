/*
 * sobel.c — 顺序版 Sobel 边缘检测实现
 *
 * 构建:
 *   gcc sobel.c -o sobel -lm
 *   （依赖 math 库以使用 sqrtf）
 * 依赖:
 *   pgmio.h 提供 PGM 读写接口（pgmread / pgmwrite）
 *
 * 用法:
 *   1) 显式指定输入/输出：
 *      ./sobel <输入.pgm> <输出.pgm>
 *      示例: ./sobel sample_256.pgm out_256_sobel.pgm
 *   2) 通过样本尺寸自动选择文件：
 *      ./sobel -n <尺寸>
 *      支持尺寸: 256 / 1024 / 4000 / 16000
 *      文件映射: 256→sample_256.pgm, 1024→sample_1024.pgm,
 *               4000→sample_4k.pgm, 16000→sample_16k.pgm
 *      输出文件名将生成为 out_<尺寸>_sobel.pgm（4K/16K 采用 out_4k_sobel.pgm / out_16k_sobel.pgm）
 *
 * 说明:
 *   - 图像内存以一维 float 数组存储，按行主序索引。
 *   - 可选预处理：3×3 均值滤波；核心处理：Sobel 梯度幅值计算。
 *   - 输出写入 PGM（P5 二进制）格式，像素值范围 [0,255]。
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h> // 用来清空内存 (memset)
#include <math.h>   // 用来算平方根 (sqrtf)
#include <stddef.h> // 用于 size_t 类型（snprintf 缓冲区长度）

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
 * 说明: 为 4000/16000 采用与仓库约定的命名（sample_4k.pgm / sample_16k.pgm）
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
 * 3×3 均值滤波
 * 输入:  in_image — 源图像（float，行主序，一维）
 * 输出:  out_image — 均值滤波结果（float）
 * 参数:  width（列数，cols）/ height（行数，rows）
 * 说明:  内部像素使用 3×3 邻域平均；边界像素简单置 0 以规避越界访问。
 */
void blur_filter(const float *in_image, float *out_image, int width, int height) {
    printf("开始模糊处理...\n");
    
    // 遍历内部像素（排除边界）
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
    for (int y = 0; y < height; y++) {
        out_image[y * width] = 0.0f;                  // 最左列
        out_image[y * width + (width - 1)] = 0.0f;    // 最右列
    }
    for (int x = 0; x < width; x++) {
        out_image[x] = 0.0f;                          // 最上行
        out_image[(height - 1) * width + x] = 0.0f;   // 最下行
    }
}


/*
 * Sobel 边缘检测
 * 输入:  in_image — 源图像（float）
 * 输出:  out_image — 梯度幅值结果（float）
 * 参数:  width（列数，cols）/ height（行数，rows）
 * 说明:  使用 3×3 Sobel 核分别计算 Gx/Gy，再以 sqrt(Gx^2+Gy^2) 得到幅值；
 *       输出值裁剪至 [0,255]；边界像素置 0。
 */
void sobel_filter(const float *in_image, float *out_image, int width, int height) {
    printf("开始 Sobel 边缘检测...\n");
    
    // 遍历内部像素（排除边界）
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
    
    // 边界处理：置 0（简化实现）
    for (int y = 0; y < height; y++) {
        out_image[y * width] = 0.0f;                  // 最左列
        out_image[y * width + (width - 1)] = 0.0f;    // 最右列
    }
    for (int x = 0; x < width; x++) {
        out_image[x] = 0.0f;                          // 最上行
        out_image[(height - 1) * width + x] = 0.0f;   // 最下行
    }
}


/*
 * 程序入口：参数解析、PGM 读写、可选模糊与 Sobel 处理、结果输出
 */
int main(int argc, char *argv[]) {

    // 解析参数：支持两种用法
    // 1) ./sobel <输入.pgm> <输出.pgm>
    // 2) ./sobel -n <尺寸> （自动映射样图与输出文件名）
    char input_file_buf[128] = {0};   // 默认输入文件（当使用 -n 时写入）
    char output_file_buf[128] = {0};  // 默认输出文件（当使用 -n 时写入）
    char *input_file = NULL;
    char *output_file = NULL;

    // 计时器变量
    clock_t start, end;
    double cpu_time_used;

    if (argc == 3 && strcmp(argv[1], "-n") == 0) {
        // 使用样本尺寸参数
        int size = atoi(argv[2]);
        if (build_paths_for_size(size, input_file_buf, sizeof(input_file_buf), output_file_buf, sizeof(output_file_buf)) != 0) {
            fprintf(stderr, "尺寸不支持: %d（支持 256/1024/4000/16000）\n", size);
            return 1;
        }
        input_file = input_file_buf;
        output_file = output_file_buf;
        printf("使用尺寸参数 -n %d，输入: %s，输出: %s\n", size, input_file, output_file);
    } else if (argc == 3) {
        // 传统用法：显式指定输入/输出文件
        input_file = argv[1];
        output_file = argv[2];
    } else {
        fprintf(stderr, "用法: %s <输入图片.pgm> <输出图片.pgm>\n", argv[0]);
        fprintf(stderr, "     或: %s -n <样本尺寸>    （支持 256/1024/4000/16000）\n", argv[0]);
        return 1;
    }

    // 缓冲区与尺寸声明（float 像素，行/列）
    float *image_buffer; // 原始图像
    float *blur_buffer;  // 均值滤波结果
    float *sobel_buffer; // Sobel 幅值结果
    int rows, cols;      // 行（rows）、列（cols）

    // 读取 PGM 图像（pgmread 内部分配 image_buffer）
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

    start = clock();

    // 可选预处理：3×3 均值滤波（输入: 原始，输出: 模糊）
    blur_filter(image_buffer, blur_buffer, cols, rows); // 宽度=cols，高度=rows
    
    // 核心处理：Sobel 边缘检测（输入: 模糊，输出: 幅值）
    sobel_filter(blur_buffer, sobel_buffer, cols, rows); // 宽度=cols，高度=rows

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    // 写入结果为 PGM（P5 二进制）
    printf("写入图片: %s\n", output_file);
    if (pgmwrite(output_file, sobel_buffer, rows, cols, 1) != 0) {
        fprintf(stderr, "写入图片失败: %s\n", output_file);
    }
    printf("完成!\n");
    printf("sequential version (sobel.c) 执行时间: %f 秒\n", cpu_time_used);

    // 释放分配资源
    free(image_buffer);
    free(blur_buffer);
    free(sobel_buffer);

    return 0;
}