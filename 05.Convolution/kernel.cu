
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// Basic usage (see HDR discussion below for HDR usage):
//    int x,y,n;
//    unsigned char *data = stbi_load(filename, &x, &y, &n, 0);
//    // ... process data if not NULL ...
//    // ... x = width, y = height, n = # 8-bit components per pixel ...
//    // ... replace '0' with '1'..'4' to force that many components per pixel
//    // ... but 'n' will always be the number that it would have been if you said 0
//    stbi_image_free(data)
//
// Standard parameters:
//    int *x                 -- outputs image width in pixels
//    int *y                 -- outputs image height in pixels
//    int *channels_in_file  -- outputs # of image components in image file
//    int desired_channels   -- if non-zero, # of image components requested in result
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

//      int stbi_write_png(char const *filename, int w, int h, int comp, const void *data, int stride_in_bytes);
//      int stbi_write_bmp(char const *filename, int w, int h, int comp, const void *data);
//      int stbi_write_tga(char const *filename, int w, int h, int comp, const void *data);
//      int stbi_write_hdr(char const *filename, int w, int h, int comp, const float *data);
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


int main()
{
    int x, y, n;
    unsigned char * image = stbi_load("image.png", &x, &y, &n, 0);
    stbi_write_png("image_out.png", x, y, n, image, 0);
    stbi_image_free(image);

    return 0;
}
