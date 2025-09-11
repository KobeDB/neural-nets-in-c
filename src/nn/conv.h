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

typedef struct NN_SmallCNN NN_SmallCNN;
struct NN_SmallCNN {
    NN_Conv2D convs[4];
    NN_Layer fc;
};

internal NN_Conv2D nn_make_conv2d(Arena *arena, int in_channels, int out_channels, int kernel_size, int stride, int padding, B32 has_bias);

internal AG_ValueArray3D nn_conv2d_apply(Arena *value_arena, Arena *array_arena, NN_Conv2D *conv2d, AG_ValueArray3D *x);

internal NN_SmallCNN nn_make_small_cnn(Arena *arena, int num_classes);

internal AG_ValueArray nn_small_cnn_apply(Arena *value_arena, Arena *array_arena, NN_SmallCNN *cnn, AG_ValueArray3D *x);

internal AG_ValueArray3D nn_relu_3d(Arena *value_arena, Arena *array_arena, AG_ValueArray3D *x);

internal AG_ValueArray nn_gap(Arena *value_arena, Arena *array_arena, AG_ValueArray3D *x);

#endif
