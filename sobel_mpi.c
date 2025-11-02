/*
 * sobel_mpi.c - MPI并行版Sobel边缘检测
 * 
 * 编译: mpicc sobel_mpi.c -o sobel_mpi -lm
 * 
 * 用法:
 *   mpirun -np <N> ./sobel_mpi <输入.pgm> <输出.pgm>
 *   mpirun -np <N> ./sobel_mpi -n <尺寸>
 * 
 * 策略: 1D行分解 + MPI_Sendrecv通信 + Ghost区域
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "pgmio.h"

// Sobel算子
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

// 根据样本尺寸构建文件路径
static int build_paths_for_size(int size, char *in_path, size_t in_len, 
                                char *out_path, size_t out_len) {
    if (size == 256) {
        snprintf(in_path, in_len, "sample_256.pgm");
        snprintf(out_path, out_len, "out_256_sobel_mpi.pgm");
        return 0;
    } else if (size == 1024) {
        snprintf(in_path, in_len, "sample_1024.pgm");
        snprintf(out_path, out_len, "out_1024_sobel_mpi.pgm");
        return 0;
    } else if (size == 4000) {
        snprintf(in_path, in_len, "sample_4k.pgm");
        snprintf(out_path, out_len, "out_4k_sobel_mpi.pgm");
        return 0;
    } else if (size == 16000) {
        snprintf(in_path, in_len, "sample_16k.pgm");
        snprintf(out_path, out_len, "out_16k_sobel_mpi.pgm");
        return 0;
    }
    return -1;
}

// 计算数据分布：每个进程负责的行数和偏移量
void compute_distribution(int height, int width, int nprocs,
                         int *sendcounts, int *displs, int *local_rows) {
    int base_rows = height / nprocs;
    int remainder = height % nprocs;
    
    int offset = 0;
    for (int i = 0; i < nprocs; i++) {
        // 前remainder个进程多分配1行
        local_rows[i] = base_rows + (i < remainder ? 1 : 0);
        sendcounts[i] = local_rows[i] * width;
        displs[i] = offset;
        offset += sendcounts[i];
    }
}

// Halo Exchange: 与上下邻居交换边界数据
void halo_exchange(float *data, int local_rows, int width, 
                  int rank, int nprocs) {
    MPI_Status status;
    
    // 地址计算
    float *ghost_top = &data[0];                           // 第0行
    float *first_row = &data[width];                       // 第1行（实际数据首行）
    float *last_row = &data[local_rows * width];           // 实际数据末行
    float *ghost_bottom = &data[(local_rows + 1) * width]; // 最后1行
    
    // 与上邻居交换
    if (rank > 0) {
        MPI_Sendrecv(first_row, width, MPI_FLOAT, rank - 1, 0,
                     ghost_top, width, MPI_FLOAT, rank - 1, 1,
                     MPI_COMM_WORLD, &status);
    } else {
        // rank 0没有上邻居，ghost_top保持为0
        memset(ghost_top, 0, width * sizeof(float));
    }
    
    // 与下邻居交换
    if (rank < nprocs - 1) {
        MPI_Sendrecv(last_row, width, MPI_FLOAT, rank + 1, 1,
                     ghost_bottom, width, MPI_FLOAT, rank + 1, 0,
                     MPI_COMM_WORLD, &status);
    } else {
        // 最后一个进程没有下邻居，ghost_bottom保持为0
        memset(ghost_bottom, 0, width * sizeof(float));
    }
}

// 本地Blur Filter (3x3均值滤波)
void blur_filter_local(const float *input, float *output, 
                      int local_rows, int width, int global_start_row, int height) {
    // 遍历本地所有行（在包含ghost的坐标系中，实际数据从index 1开始）
    for (int local_y = 0; local_y < local_rows; local_y++) {
        int global_y = global_start_row + local_y;  // 全局行号
        int in_y = local_y + 1;  // 在包含ghost的数组中的位置
        
        for (int x = 0; x < width; x++) {
            // 图像边界直接设为0
            if (global_y == 0 || global_y == height - 1 || 
                x == 0 || x == width - 1) {
                output[in_y * width + x] = 0.0f;
                continue;
            }
            
            // 3x3邻域求和
            float sum = 0.0f;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    sum += input[(in_y + dy) * width + (x + dx)];
                }
            }
            output[in_y * width + x] = sum / 9.0f;
        }
    }
    
    // 保持ghost区域为0
    memset(&output[0], 0, width * sizeof(float));
    memset(&output[(local_rows + 1) * width], 0, width * sizeof(float));
}

// 本地Sobel Filter
void sobel_filter_local(const float *input, float *output,
                       int local_rows, int width, int global_start_row, int height) {
    // 输出数组不含ghost，直接对应本地行
    for (int local_y = 0; local_y < local_rows; local_y++) {
        int global_y = global_start_row + local_y;
        int in_y = local_y + 1;  // 输入包含ghost，需要偏移
        
        for (int x = 0; x < width; x++) {
            // 边界设为0
            if (global_y == 0 || global_y == height - 1 || 
                x == 0 || x == width - 1) {
                output[local_y * width + x] = 0.0f;
                continue;
            }
            
            float sum_x = 0.0f;
            float sum_y = 0.0f;
            
            // 应用Sobel算子
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    float pixel = input[(in_y + dy) * width + (x + dx)];
                    sum_x += pixel * Gx[dy + 1][dx + 1];
                    sum_y += pixel * Gy[dy + 1][dx + 1];
                }
            }
            
            float magnitude = sqrtf(sum_x * sum_x + sum_y * sum_y);
            if (magnitude > 255.0f) magnitude = 255.0f;
            if (magnitude < 0.0f) magnitude = 0.0f;
            
            output[local_y * width + x] = magnitude;
        }
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    //=========================================================================
    // 1. 命令行参数解析
    //=========================================================================
    char input_file_buf[128] = {0};
    char output_file_buf[128] = {0};
    char *input_file = NULL;
    char *output_file = NULL;
    
    if (argc == 3 && strcmp(argv[1], "-n") == 0) {
        int size = atoi(argv[2]);
        if (build_paths_for_size(size, input_file_buf, sizeof(input_file_buf),
                                output_file_buf, sizeof(output_file_buf)) != 0) {
            if (rank == 0) {
                fprintf(stderr, "不支持的尺寸: %d (支持 256/1024/4000/16000)\n", size);
            }
            MPI_Finalize();
            return 1;
        }
        input_file = input_file_buf;
        output_file = output_file_buf;
    } else if (argc == 3) {
        input_file = argv[1];
        output_file = argv[2];
    } else {
        if (rank == 0) {
            fprintf(stderr, "用法: mpirun -np <N> %s <输入.pgm> <输出.pgm>\n", argv[0]);
            fprintf(stderr, "  或: mpirun -np <N> %s -n <尺寸>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    //=========================================================================
    // 2. Master进程读取图像
    //=========================================================================
    float *full_image = NULL;
    int height, width;
    
    if (rank == 0) {
        if (pgmread(input_file, &full_image, &height, &width) != 0) {
            fprintf(stderr, "读取图像失败: %s\n", input_file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("读取图像: %s\n", input_file);
        printf("图像尺寸: %d x %d\n", height, width);
        printf("使用进程数: %d\n", nprocs);
    }
    
    // 广播图像尺寸
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    //=========================================================================
    // 3. 计算数据分布
    //=========================================================================
    int *sendcounts = NULL;
    int *displs = NULL;
    int *rows_per_proc = NULL;
    
    if (rank == 0) {
        sendcounts = (int *)malloc(nprocs * sizeof(int));
        displs = (int *)malloc(nprocs * sizeof(int));
        rows_per_proc = (int *)malloc(nprocs * sizeof(int));
        
        compute_distribution(height, width, nprocs, sendcounts, displs, rows_per_proc);
    }
    
    // 获取本进程的行数和起始行号
    int local_rows;
    int global_start_row;
    
    if (rank == 0) {
        local_rows = rows_per_proc[0];
        global_start_row = 0;
        
        // 发送给其他进程
        for (int i = 1; i < nprocs; i++) {
            int start_row = displs[i] / width;
            MPI_Send(&rows_per_proc[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&start_row, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&local_rows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&global_start_row, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    //=========================================================================
    // 4. 分配本地缓冲区
    //=========================================================================
    // local_input: 包含ghost区域 (local_rows + 2) x width
    float *local_input = (float *)malloc((local_rows + 2) * width * sizeof(float));
    float *local_blur = (float *)malloc((local_rows + 2) * width * sizeof(float));
    float *local_sobel = (float *)malloc(local_rows * width * sizeof(float));
    
    if (!local_input || !local_blur || !local_sobel) {
        fprintf(stderr, "Rank %d: 内存分配失败\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    memset(local_input, 0, (local_rows + 2) * width * sizeof(float));
    memset(local_blur, 0, (local_rows + 2) * width * sizeof(float));
    
    //=========================================================================
    // 5. Scatter: 分发数据到各进程
    //=========================================================================
    // 接收数据到实际数据区域（跳过第一行ghost）
    MPI_Scatterv(full_image, sendcounts, displs, MPI_FLOAT,
                 &local_input[width],  // 跳过ghost_top
                 local_rows * width, MPI_FLOAT,
                 0, MPI_COMM_WORLD);
    
    //=========================================================================
    // 6. 开始计时
    //=========================================================================
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    
    //=========================================================================
    // 7. Blur阶段
    //=========================================================================
    halo_exchange(local_input, local_rows, width, rank, nprocs);
    blur_filter_local(local_input, local_blur, local_rows, width, 
                     global_start_row, height);
    
    //=========================================================================
    // 8. Sobel阶段
    //=========================================================================
    halo_exchange(local_blur, local_rows, width, rank, nprocs);
    sobel_filter_local(local_blur, local_sobel, local_rows, width,
                      global_start_row, height);
    
    //=========================================================================
    // 9. 结束计时
    //=========================================================================
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double local_time = end_time - start_time;
    
    // 收集最大执行时间
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    //=========================================================================
    // 10. Gather: 收集结果
    //=========================================================================
    float *full_result = NULL;
    if (rank == 0) {
        full_result = (float *)malloc(height * width * sizeof(float));
    }
    
    MPI_Gatherv(local_sobel, local_rows * width, MPI_FLOAT,
                full_result, sendcounts, displs, MPI_FLOAT,
                0, MPI_COMM_WORLD);
    
    //=========================================================================
    // 11. Master进程写入结果
    //=========================================================================
    if (rank == 0) {
        printf("写入图像: %s\n", output_file);
        if (pgmwrite(output_file, full_result, height, width, 1) != 0) {
            fprintf(stderr, "写入图像失败: %s\n", output_file);
        }
        printf("完成!\n");
        printf("MPI版本执行时间: %.6f 秒\n", max_time);
        
        free(full_image);
        free(full_result);
        free(sendcounts);
        free(displs);
        free(rows_per_proc);
    }
    
    //=========================================================================
    // 12. 清理
    //=========================================================================
    free(local_input);
    free(local_blur);
    free(local_sobel);
    
    MPI_Finalize();
    return 0;
}