unsigned char * open_image(char *file_in, int* width, int* height, \
                           int* num_channels, int NUM_CHANNELS_RGB) ;
void free_image (unsigned char * image);
void write_image(char* file_out, int width, int height, \
                 int NUM_CHANNELS_GREYSCALE, unsigned char * image_data, \
                 int PNG_STRIDE_DEFAULT);
