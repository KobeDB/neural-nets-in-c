#ifndef CONV_H
#define CONV_H

typedef struct NN_Conv2D NN_Conv2D;
struct NN_Conv2D {
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;

    AG_ValueArray4D weights;
    AG_ValueArray   biases;
};

internal NN_Conv2D nn_make_conv2d(Arena *arena, int in_channels, int out_channels, int kernel_size, int stride, int padding, B32 has_bias);

internal AG_ValueArray3D nn_conv2d_apply(Arena *arena, NN_Conv2D *conv2d, AG_ValueArray3D *x);

#endif
