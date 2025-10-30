#include <stdio.h>
#include <stdlib.h>
#include "pgmio.h"

int main() {
    const char *input = "../sample_256.pgm";
    const char *output = "copy_output_256.pgm";
    float *img;
    int rows, cols;

    if (pgmread(input, &img, &rows, &cols) != 0) {
        fprintf(stderr, "Failed to read %s\n", input);
        return 1;
    }

    printf("Image loaded: %dx%d\n", cols, rows);

    // Save a copy in binary format
    if (pgmwrite(output, img, rows, cols, 1) != 0) {
        fprintf(stderr, "Failed to write %s\n", output);
        free(img);
        return 1;
    }

    printf("Image written to %s\n", output);
    free(img);
    return 0;
}
