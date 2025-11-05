//sobel_omp.c - OpenMP并行版Sobel边缘检测（Tile分块优化）

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <omp.h>
#include "pgmio.h"

// Sobel 核，水平 Gx 与垂直 Gy
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

// 根据图像尺寸自适应选择 tile 边长
// 尺寸越大使用越大的 tile，减少块数量和调度开销
static int select_tile_size(int width, int height) {
    int max_dim = (width > height) ? width : height;
    
    if (max_dim <= 1024) {
        return 256;   // 小图像采用较大 tile，块内访问更连续
    } else if (max_dim <= 4000) {
        return 512;   // 中等图像
    } else {
        return 1024;  // 大图像使用更大的 tile，减少块数量与调度开销
    }
}

/* 根据常用尺寸生成示例输入输出路径，便于批量测试。
   成功返回 0，失败返回 -1。*/
static int build_paths_for_size(int size, char *in_path, size_t in_len, 
                                char *out_path, size_t out_len) {
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
 * 3x3 均值滤波（tile 分块并行）
 * 将图像切分为 tile_size x tile_size 的块，线程按块处理。
 * 块内访问按行连续，提升 L1/L2 缓存命中率。
 */
void blur_filter_tiled(const float *in_image, float *out_image, 
                       int width, int height) {
    printf("开始模糊处理 (tile-based并行)...\n");
    
    int tile_size = select_tile_size(width, height); // 自适应选择 tile 大小
    printf("使用tile大小: %d x %d\n", tile_size, tile_size);
    
    // 分块数量，向上取整避免遗漏
    int num_tiles_x = (width + tile_size - 1) / tile_size;
    int num_tiles_y = (height + tile_size - 1) / tile_size;
    
    // 并行遍历每个tile
    // collapse(2)把两层循环合并，增加并行粒度
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ty = 0; ty < num_tiles_y; ty++) {
        for (int tx = 0; tx < num_tiles_x; tx++) {
            
            // 当前块的坐标范围
            int x_start = tx * tile_size;
            int y_start = ty * tile_size;
            int x_end = x_start + tile_size;
            int y_end = y_start + tile_size;
            
            // 最后一块可能越界，裁剪到图像范围
            if (x_end > width) x_end = width;
            if (y_end > height) y_end = height;
            
            // 遍历块内像素
            for (int y = y_start; y < y_end; y++) {
                for (int x = x_start; x < x_end; x++) {
                    
                    // 跳过边界像素（没有完整的3x3邻域）
                    if (y == 0 || y == height - 1 || x == 0 || x == width - 1) {
                        out_image[y * width + x] = 0.0f;
                        continue;
                    }
                    
                    float sum = 0.0f;
                    
                    // 3x3邻域求和
                    for (int ky = -1; ky <= 1; ky++) {
                        for (int kx = -1; kx <= 1; kx++) {
                            sum += in_image[(y + ky) * width + (x + kx)];
                        }
                    }
                    
                    out_image[y * width + x] = sum / 9.0f;
                }
            }
        }
    }
}

/*
 * Sobel 边缘检测（tile 分块并行）
 * 分块与并行方式与均值滤波一致，块内计算独立，减少跨缓存行访问。
 */
void sobel_filter_tiled(const float *in_image, float *out_image, 
                        int width, int height) {
    printf("开始Sobel边缘检测 (tile-based并行)...\n");
    
    int tile_size = select_tile_size(width, height);
    
    int num_tiles_x = (width + tile_size - 1) / tile_size;
    int num_tiles_y = (height + tile_size - 1) / tile_size;
    
    // 合并两个维度并使用静态调度，块大小一致，负载均衡
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ty = 0; ty < num_tiles_y; ty++) {
        for (int tx = 0; tx < num_tiles_x; tx++) {
            
            int x_start = tx * tile_size; // 当前块起始列
            int y_start = ty * tile_size; // 当前块起始行
            int x_end = x_start + tile_size;
            int y_end = y_start + tile_size;
            
            if (x_end > width) x_end = width;   // 边界裁剪到图像范围
            if (y_end > height) y_end = height;
            
            for (int y = y_start; y < y_end; y++) {
                for (int x = x_start; x < x_end; x++) {
                    
                    // 边界像素没有完整 3x3 邻域，输出置零
                    if (y == 0 || y == height - 1 || x == 0 || x == width - 1) {
                        out_image[y * width + x] = 0.0f;
                        continue;
                    }
                    
                    float sum_x = 0.0f;
                    float sum_y = 0.0f;
                    
                    // 应用Sobel核
                    for (int ky = -1; ky <= 1; ky++) {
                        for (int kx = -1; kx <= 1; kx++) {
                            float pixel_val = in_image[(y + ky) * width + (x + kx)];
                            sum_x += pixel_val * Gx[ky + 1][kx + 1];
                            sum_y += pixel_val * Gy[ky + 1][kx + 1];
                        }
                    }
                    
                    // 梯度幅值
                    float magnitude = sqrtf(sum_x * sum_x + sum_y * sum_y);
                    
                    // 限幅到[0, 255]
                    if (magnitude > 255.0f) magnitude = 255.0f;
                    if (magnitude < 0.0f) magnitude = 0.0f;
                    
                    out_image[y * width + x] = magnitude;
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int num_threads = omp_get_max_threads(); // 默认使用系统可用的最大线程数
    char input_file_buf[128] = {0};
    char output_file_buf[128] = {0};
    char *input_file = NULL;
    char *output_file = NULL;
    
    double start_time, end_time, cpu_time_used; // 计时变量
    
    // 命令行参数解析
    if (argc >= 3 && strcmp(argv[1], "-n") == 0) {
        int size = atoi(argv[2]);
        if (build_paths_for_size(size, input_file_buf, sizeof(input_file_buf),
                                output_file_buf, sizeof(output_file_buf)) != 0) {
            fprintf(stderr, "不支持的尺寸: %d (支持 256/1024/4000/16000)\n", size);
            return 1;
        }
        input_file = input_file_buf;
        output_file = output_file_buf;
        
        if (argc >= 4) {
            num_threads = atoi(argv[3]);
            if (num_threads <= 0) {
                num_threads = omp_get_max_threads();
            }
        }
        
        printf("样本尺寸: %d, 输入: %s, 输出: %s, 线程数: %d\n",
               size, input_file, output_file, num_threads);
    } else if (argc >= 3) {
        input_file = argv[1];
        output_file = argv[2];
        
        if (argc >= 4) {
            num_threads = atoi(argv[3]);
            if (num_threads <= 0) {
                num_threads = omp_get_max_threads();
            }
        }
        
        printf("输入: %s, 输出: %s, 线程数: %d\n", 
               input_file, output_file, num_threads);
    } else {
        fprintf(stderr, "用法: %s <输入.pgm> <输出.pgm> [线程数]\n", argv[0]);
        fprintf(stderr, "  或: %s -n <尺寸> [线程数] (256/1024/4000/16000)\n", argv[0]);
        return 1;
    }
    
    omp_set_num_threads(num_threads); // 设置 OpenMP 线程数
    
    float *image_buffer, *blur_buffer, *sobel_buffer;
    int rows, cols;
    
    printf("读取图像: %s\n", input_file);
    if (pgmread(input_file, &image_buffer, &rows, &cols) != 0) {
        fprintf(stderr, "读取失败: %s\n", input_file);
        return 1;
    }
    printf("图像尺寸: %d x %d\n", rows, cols);
    
    blur_buffer = (float *)malloc(rows * cols * sizeof(float));   // 均值滤波中间缓冲
    sobel_buffer = (float *)malloc(rows * cols * sizeof(float));  // Sobel 结果缓冲
    if (!blur_buffer || !sobel_buffer) {
        fprintf(stderr, "内存分配失败\n");
        free(image_buffer);
        return 1;
    }
    
    memset(blur_buffer, 0, rows * cols * sizeof(float));  // 初始化为 0，便于边界处理
    memset(sobel_buffer, 0, rows * cols * sizeof(float)); // 初始化为 0
    
    // 开始计时
    start_time = omp_get_wtime();
    
    // 先模糊后Sobel
    blur_filter_tiled(image_buffer, blur_buffer, cols, rows);
    sobel_filter_tiled(blur_buffer, sobel_buffer, cols, rows);
    
    end_time = omp_get_wtime();
    cpu_time_used = end_time - start_time;
    
    printf("写入结果: %s\n", output_file);
    if (pgmwrite(output_file, sobel_buffer, rows, cols, 1) != 0) {
        fprintf(stderr, "写入失败: %s\n", output_file);
    }
    
    printf("完成!\n");
    printf("Tile-based OpenMP版本执行时间: %.6f 秒\n", cpu_time_used);
    printf("使用线程数: %d\n", num_threads);
    
    free(image_buffer);
    free(blur_buffer);
    free(sobel_buffer);
    
    return 0;
}