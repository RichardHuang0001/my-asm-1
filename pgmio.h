#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Read PGM (P2 or P5) into float array (allocated inside)
int pgmread(const char *filename, float **img, int *rows, int *cols) {
    FILE *f = fopen(filename, "rb");
    if (!f) { perror("fopen"); return -1; }

    char magic[3];
    if (fscanf(f, "%2s", magic) != 1) { fclose(f); return -1; }

    // Skip comments and whitespace until width & height
    int w=-1,h=-1,maxval=-1;
    while(w<0 || h<0) {
        int c = fgetc(f);
        if(c=='#') while(fgetc(f)!='\n'); // skip comment line
        else if(c==' '||c=='\n'||c=='\r'||c=='\t') continue;
        else { ungetc(c,f); if(fscanf(f,"%d %d",&w,&h)!=2){ fclose(f); return -1;} }
    }

    // Skip comments/whitespace until maxval
    while(maxval<0) {
        int c = fgetc(f);
        if(c=='#') while(fgetc(f)!='\n');
        else if(c==' '||c=='\n'||c=='\r'||c=='\t') continue;
        else { ungetc(c,f); if(fscanf(f,"%d",&maxval)!=1){ fclose(f); return -1;} }
    }

    *rows = h;
    *cols = w;
    *img = malloc(w*h*sizeof(float));
    if(!*img) { fclose(f); return -1; }

    if(strcmp(magic,"P5")==0) {
        fgetc(f); // skip one whitespace
        unsigned char *tmp = malloc(w*h);
        if(!tmp) { fclose(f); free(*img); return -1; }
        if(fread(tmp,1,w*h,f)!=(size_t)(w*h)) { fclose(f); free(*img); free(tmp); return -1; }
        for(int i=0;i<w*h;i++) (*img)[i]=(float)tmp[i];
        free(tmp);
    } else if(strcmp(magic,"P2")==0) {
        for(int i=0;i<w*h;i++) {
            int val;
            if(fscanf(f,"%d",&val)!=1) { fclose(f); free(*img); return -1; }
            (*img)[i]=(float)val;
        }
    } else {
        fclose(f);
        free(*img);
        return -1;
    }

    fclose(f);
    return 0;
}

// Write PGM (P2 or P5)
int pgmwrite(const char *filename, const float *img, int rows, int cols, int binary) {
    FILE *f = fopen(filename, binary?"wb":"w");
    if(!f) { perror("fopen"); return -1; }

    if(binary) {
        fprintf(f,"P5\n%d %d\n255\n", cols, rows);
        unsigned char *tmp = malloc(rows*cols);
        if(!tmp) { fclose(f); return -1; }
        for(int i=0;i<rows*cols;i++) {
            int val = (int)(img[i]+0.5f);
            if(val<0) val=0;
            if(val>255) val=255;
            tmp[i]=(unsigned char)val;
        }
        fwrite(tmp,1,rows*cols,f);
        free(tmp);
    } else {
        fprintf(f,"P2\n%d %d\n255\n", cols, rows);
        for(int i=0;i<rows*cols;i++) {
            int val = (int)(img[i]+0.5f);
            if(val<0) val=0;
            if(val>255) val=255;
            fprintf(f,"%d ",val);
            if ((i+1)%16==0) fprintf(f,"\n");
        }
        if((rows*cols)%16!=0) fprintf(f,"\n");
    }

    fclose(f);
    return 0;
}
