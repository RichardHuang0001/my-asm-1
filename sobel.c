/*
 * sobel.c — 顺序版 Sobel 边缘检测实现
 *
 * 构建:
 *   gcc sobel.c -o sobel -lm -fopenmp
 *   （需要-fopenmp以使用omp_get_wtime()计时）
 * 依赖:
 *   pgmio.h 提供 PGM 读写接口（pgmread / pgmwrite）
 *
 * 用法:
 *   1) 显式指定输入/输出：
 *      ./sobel <输入.pgm> <输出.pgm>
 *   2) 通过样本尺寸自动选择文件：
 *      ./sobel -n <尺寸>
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

void blur_filter(const float *in_image, float *out_image, int width, int height) {
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float sum = 0.0f;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    sum += in_image[(y + ky) * width + (x + kx)];
                }
            }
            out_image[y * width + x] = sum / 9.0f;
        }
    }
    
    // 边界处理
    for (int y = 0; y < height; y++) {
        out_image[y * width] = 0.0f;
        out_image[y * width + (width - 1)] = 0.0f;
    }
    for (int x = 0; x < width; x++) {
        out_image[x] = 0.0f;
        out_image[(height - 1) * width + x] = 0.0f;
    }
}

void sobel_filter(const float *in_image, float *out_image, int width, int height) {
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float sum_x = 0.0f;
            float sum_y = 0.0f;
            
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    float pixel_val = in_image[(y + ky) * width + (x + kx)];
                    sum_x += pixel_val * Gx[ky + 1][kx + 1];
                    sum_y += pixel_val * Gy[ky + 1][kx + 1];
                }
            }
            
            float magnitude = sqrtf(sum_x * sum_x + sum_y * sum_y);
            if (magnitude > 255.0f) magnitude = 255.0f;
            if (magnitude < 0.0f) magnitude = 0.0f;
            
            out_image[y * width + x] = magnitude;
        }
    }
    
    // 边界处理
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
    char input_file_buf[128] = {0};
    char output_file_buf[128] = {0};
    char *input_file = NULL;
    char *output_file = NULL;

    // 改用omp_get_wtime()以保证与OpenMP版本的计时一致
    double start_time, end_time, cpu_time_used;

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
        input_file = argv[1];
        output_file = argv[2];
    } else {
        fprintf(stderr, "用法: %s <输入图片.pgm> <输出图片.pgm>\n", argv[0]);
        fprintf(stderr, "     或: %s -n <样本尺寸>    （支持 256/1024/4000/16000）\n", argv[0]);
        return 1;
    }

    float *image_buffer, *blur_buffer, *sobel_buffer;
    int rows, cols;

    printf("读取图片: %s\n", input_file);
    if (pgmread(input_file, &image_buffer, &rows, &cols) != 0) {
        fprintf(stderr, "读取图片失败: %s\n", input_file);
        return 1;
    }
    printf("图片尺寸: %d x %d (行 x 列)\n", rows, cols);

    blur_buffer = (float *)malloc(rows * cols * sizeof(float));
    sobel_buffer = (float *)malloc(rows * cols * sizeof(float));
    if (blur_buffer == NULL || sobel_buffer == NULL) {
        fprintf(stderr, "内存分配失败\n");
        return 1;
    }
    
    memset(blur_buffer, 0, rows * cols * sizeof(float));
    memset(sobel_buffer, 0, rows * cols * sizeof(float));

    // 使用omp_get_wtime()计时（与OpenMP版本一致）
    start_time = omp_get_wtime();

    blur_filter(image_buffer, blur_buffer, cols, rows);
    sobel_filter(blur_buffer, sobel_buffer, cols, rows);

    end_time = omp_get_wtime();
    cpu_time_used = end_time - start_time;

    printf("写入图片: %s\n", output_file);
    if (pgmwrite(output_file, sobel_buffer, rows, cols, 1) != 0) {
        fprintf(stderr, "写入图片失败: %s\n", output_file);
    }
    printf("完成!\n");
    printf("Sequential版本执行时间: %.6f 秒\n", cpu_time_used);

    free(image_buffer);
    free(blur_buffer);
    free(sobel_buffer);

    return 0;
}