/*
  文件：sobel.c
  说明：顺序版的 Sobel 边缘检测示例，包含一个先做 3x3 均值模糊、再做 Sobel 的简单流水线。
       计时使用 OpenMP 的 omp_get_wtime，与 OpenMP 版本保持一致，便于对比。

  构建：gcc sobel.c -o sobel -lm -fopenmp
        需要链接 math 库用到 sqrtf，使用 -fopenmp 以获得 omp_get_wtime。

  依赖：pgmio.h 提供 PGM 图片读写函数 pgmread / pgmwrite。
        函数签名见 pgmio.h，读写的 rows 与 cols 含义为 行数 与 列数。

  用法：
    显式指定输入输出：
      ./sobel 输入.pgm 输出.pgm
    根据样本尺寸选择内置文件：
      ./sobel -n 尺寸    支持 256 / 1024 / 4000 / 16000

  注意：本实现仅关注演示与测量时间，边界像素采用置零的简化处理，输出为 8 位范围。
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <omp.h>      // 改用omp_get_wtime()以保证计时一致性
#include "pgmio.h"

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

// 根据尺寸生成默认的输入输出文件名
// 返回 0 表示成功，-1 表示不支持的尺寸
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

// 3x3 均值模糊（盒滤波）。
// in_image 为输入灰度图，out_image 为输出缓冲区，width 为列数，height 为行数。
// 内部仅处理非边界像素，边界像素在后续统一置零。
void blur_filter(const float *in_image, float *out_image, int width, int height) {
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float sum = 0.0f;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    // 行主序索引访问 3x3 邻域像素
                    sum += in_image[(y + ky) * width + (x + kx)];
                }
            }
            // 盒滤波将 3x3 区域求和后取平均
            out_image[y * width + x] = sum / 9.0f;
        }
    }
    
    // 边界处理：将最外圈像素置零，避免越界访问的复杂分支
    for (int y = 0; y < height; y++) {
        out_image[y * width] = 0.0f;
        out_image[y * width + (width - 1)] = 0.0f;
    }
    for (int x = 0; x < width; x++) {
        out_image[x] = 0.0f;
        out_image[(height - 1) * width + x] = 0.0f;
    }
}

// Sobel 梯度与幅值计算。
// 对每个像素计算在 x、y 两个方向的梯度，再用平方和开方得到梯度幅值。
// 输出范围在 [0,255] 内，超出范围的值进行截断。
void sobel_filter(const float *in_image, float *out_image, int width, int height) {
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float sum_x = 0.0f;
            float sum_y = 0.0f;
            
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    // 邻域像素值
                    float pixel_val = in_image[(y + ky) * width + (x + kx)];
                    // 与水平核、垂直核做卷积累加
                    sum_x += pixel_val * Gx[ky + 1][kx + 1];
                    sum_y += pixel_val * Gy[ky + 1][kx + 1];
                }
            }
            
            // 梯度幅值 sqrt(gx^2 + gy^2)
            float magnitude = sqrtf(sum_x * sum_x + sum_y * sum_y);
            // 截断到 8 位范围，保持与写出时的像素范围一致
            if (magnitude > 255.0f) magnitude = 255.0f;
            if (magnitude < 0.0f) magnitude = 0.0f;
            
            out_image[y * width + x] = magnitude;
        }
    }
    
    // 边界处理：同模糊滤波，最外圈置零
    for (int y = 0; y < height; y++) {
        out_image[y * width] = 0.0f;
        out_image[y * width + (width - 1)] = 0.0f;
    }
    for (int x = 0; x < width; x++) {
        out_image[x] = 0.0f;
        out_image[(height - 1) * width + x] = 0.0f;
    }
}

int main(int argc, char *argv[]) {
    // 输入输出文件路径缓存与指针
    char input_file_buf[128] = {0};
    char output_file_buf[128] = {0};
    char *input_file = NULL;
    char *output_file = NULL;

    // 改用omp_get_wtime()以保证与OpenMP版本的计时一致
    double start_time, end_time, cpu_time_used;

    // 参数解析：支持 -n 尺寸 通过内置样本选择文件
    if (argc == 3 && strcmp(argv[1], "-n") == 0) {
        int size = atoi(argv[2]);
        if (build_paths_for_size(size, input_file_buf, sizeof(input_file_buf), 
                                output_file_buf, sizeof(output_file_buf)) != 0) {
            fprintf(stderr, "尺寸不支持: %d（支持 256/1024/4000/16000）\n", size);
            return 1;
        }
        input_file = input_file_buf;
        output_file = output_file_buf;
        printf("使用尺寸参数 -n %d，输入: %s，输出: %s\n", size, input_file, output_file);
    } else if (argc == 3) {
        // 直接指定输入输出文件
        input_file = argv[1];
        output_file = argv[2];
    } else {
        fprintf(stderr, "用法: %s <输入图片.pgm> <输出图片.pgm>\n", argv[0]);
        fprintf(stderr, "     或: %s -n <样本尺寸>    （支持 256/1024/4000/16000）\n", argv[0]);
        return 1;
    }

    // 图像数据缓冲与尺寸
    float *image_buffer, *blur_buffer, *sobel_buffer;
    int rows, cols;

    printf("读取图片: %s\n", input_file);
    if (pgmread(input_file, &image_buffer, &rows, &cols) != 0) {
        // 读取失败直接退出，保持资源安全
        fprintf(stderr, "读取图片失败: %s\n", input_file);
        return 1;
    }
    printf("图片尺寸: %d x %d (行 x 列)\n", rows, cols);

    // 分配中间缓冲
    blur_buffer = (float *)malloc(rows * cols * sizeof(float));
    sobel_buffer = (float *)malloc(rows * cols * sizeof(float));
    if (blur_buffer == NULL || sobel_buffer == NULL) {
        fprintf(stderr, "内存分配失败\n");
        return 1;
    }
    
    // 初始化为 0，便于边界赋值
    memset(blur_buffer, 0, rows * cols * sizeof(float));
    memset(sobel_buffer, 0, rows * cols * sizeof(float));

    // 开始计时
    start_time = omp_get_wtime();

    // 先模糊，后 Sobel
    blur_filter(image_buffer, blur_buffer, cols, rows);
    sobel_filter(blur_buffer, sobel_buffer, cols, rows);

    // 结束计时
    end_time = omp_get_wtime();
    cpu_time_used = end_time - start_time;

    printf("写入图片: %s\n", output_file);
    if (pgmwrite(output_file, sobel_buffer, rows, cols, 1) != 0) {
        // 写出失败仅打印错误，不再做额外处理
        fprintf(stderr, "写入图片失败: %s\n", output_file);
    }
    printf("完成!\n");
    printf("Sequential版本执行时间: %.6f 秒\n", cpu_time_used);

    // 释放所有动态分配的内存
    free(image_buffer);
    free(blur_buffer);
    free(sobel_buffer);

    return 0;
}