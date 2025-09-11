internal
NN_Conv2D nn_make_conv2d(Arena *arena, int in_channels, int out_channels, int kernel_size, int stride, int padding, B32 has_bias) {
    NN_Conv2D result = {0};
    result.in_channels = in_channels;
    result.out_channels = out_channels;
    result.kernel_size = kernel_size;
    result.stride = stride;
    result.padding = padding;
    
    result.weights = ag_push_null_value_array4d(arena, out_channels, in_channels, kernel_size, kernel_size);
    int weight_count = ag_value_array4d_element_count(&result.weights);
    for (int i = 0; i < weight_count; ++i) {
        result.weights.values[i] = ag_source(arena, sample_f64_in_range(-0.5, 0.5));
    }

    result.biases.count = out_channels;
    result.biases.values = push_array(arena, AG_Value*, result.biases.count);
    for (int i = 0; i < result.biases.count; ++i) {
        F64 bias = has_bias ? sample_f64_in_range(-0.5, 0.5) : 0;
        result.biases.values[i] = ag_source(arena, bias);
    }
    return result;
}

internal
AG_ValueArray3D nn_conv2d_apply(Arena *value_arena, Arena *array_arena, NN_Conv2D *conv2d, AG_ValueArray3D *x) {
    if (x->shape[0] != conv2d->in_channels) {
        fprintf(stderr, "conv2d_apply: x.shape[0] doesn't match conv2d.in_channels\n");
        return (AG_ValueArray3D){0};
    }
    
    int padding = conv2d->padding;
    int kernel_size = conv2d->kernel_size;
    int stride = conv2d->stride;

    int out_h = (x->shape[1] + 2*padding - kernel_size)/stride + 1;
    int out_w = (x->shape[2] + 2*padding - kernel_size)/stride + 1;

    AG_ValueArray3D result = ag_push_null_value_array3d(array_arena, conv2d->out_channels, out_h, out_w);
    int result_value_count = ag_value_array3d_element_count(&result);
    for (int i = 0; i < result_value_count; ++i) {
        result.values[i] = ag_source(value_arena, 0);
    }

    int dim_1_opl = x->shape[1] + padding; // The one-past-last valid index for dim_1 with padding
    int dim_2_opl = x->shape[2] + padding;

    for (int kernel = 0; kernel < conv2d->out_channels; ++kernel) {
        for (int cy = -padding, ty = 0; cy+kernel_size <= dim_1_opl; cy += stride, ty+=1) {
            for (int cx = -padding, tx = 0; cx+kernel_size <= dim_2_opl; cx += stride, tx+=1) {
                for (int i = cy; i < cy + kernel_size; ++i) {
                    for (int j = cx; j < cx + kernel_size; ++j) {
                        // check if we are in zero-padding
                        if (i < 0 || i >= x->shape[1] || j < 0 || j >= x->shape[2]) continue;
                        
                        for (int in_channel = 0; in_channel < x->shape[0] ; ++in_channel) {
                            AG_Value *x_pixel_value = *ag_value_array3d_get_value(x, in_channel, i, j);
                            int weight_idx_0 = i-cy;
                            int weight_idx_1 = j-cx;
                            if (!(weight_idx_0 >= 0 && weight_idx_0 < kernel_size && weight_idx_1 >= 0 && weight_idx_1 < kernel_size)) {
                                fprintf(stderr, "weight indices are WRONG\n");
                                continue;
                            }
                            AG_Value *weight = *ag_value_array4d_get_value(&conv2d->weights, kernel, in_channel, i-cy, j-cx);
                            AG_Value *weighted_pixel = ag_mul(value_arena, x_pixel_value, weight);
                            
                            AG_Value **out_pixel = ag_value_array3d_get_value(&result, kernel, ty, tx);
                            *out_pixel = ag_add(value_arena, *out_pixel, weighted_pixel);
                        }
                    }
                }
                // add bias
                AG_Value **out_pixel = ag_value_array3d_get_value(&result, kernel, ty, tx);
                *out_pixel = ag_add(value_arena, *out_pixel, conv2d->biases.values[kernel]);
            }
        }
    }

    return result;
}

internal NN_SmallCNN nn_make_small_cnn(Arena *arena, int num_classes) {
    NN_SmallCNN result = {0};

    // nn_make_conv2d(Arena *arena, int in_channels, int out_channels, int kernel_size, int stride, int padding, B32 has_bias)
    result.convs[0] = nn_make_conv2d(arena, 1,  16, 3, 1, 1, 1); // increase channels: (16,28,28)
    result.convs[1] = nn_make_conv2d(arena, 16, 16, 3, 2, 1, 1); // downsample, serves as max pooling operation: (16,14,14)
    result.convs[2] = nn_make_conv2d(arena, 16, 32, 3, 1, 1, 1); // increase channels: (32,14,14)  
    result.convs[3] = nn_make_conv2d(arena, 32, 32, 3, 2, 1, 1); // downsample, serves as max pooling operation: (32,7,7)

    int fc_input_dim = 32;
    B32 has_relu = 0;
    result.fc = nn_make_layer_with_random_init(arena, fc_input_dim, num_classes, has_relu);

    return result;
}


internal AG_ValueArray nn_small_cnn_apply(Arena *value_arena, Arena *array_arena, NN_SmallCNN *cnn, AG_ValueArray3D *x) {
    Arena *conflicts[] = {value_arena, array_arena};
    ArenaTemp scratch = scratch_begin(conflicts, ArrayCount(conflicts));

    AG_ValueArray3D cur = *x;
    
    cur = nn_conv2d_apply(value_arena, scratch.arena, &cnn->convs[0], &cur);
    cur = nn_relu_3d(value_arena, scratch.arena, &cur);

    cur = nn_conv2d_apply(value_arena, scratch.arena, &cnn->convs[1], &cur);
    cur = nn_relu_3d(value_arena, scratch.arena, &cur);

    cur = nn_conv2d_apply(value_arena, scratch.arena, &cnn->convs[2], &cur);
    cur = nn_relu_3d(value_arena, scratch.arena, &cur);

    cur = nn_conv2d_apply(value_arena, scratch.arena, &cnn->convs[3], &cur);
    cur = nn_relu_3d(value_arena, scratch.arena, &cur);

    AG_ValueArray h = nn_gap(value_arena, scratch.arena, &cur);

    // put last result's array on caller's array_arena
    AG_ValueArray result = nn_layer_apply(value_arena, array_arena, &cnn->fc, h);

    scratch_end(scratch);
    return result;
}

internal AG_ValueArray3D nn_relu_3d(Arena *value_arena, Arena *array_arena, AG_ValueArray3D *x) {
    AG_ValueArray3D result = ag_push_null_value_array3d(array_arena, x->shape[0], x->shape[1], x->shape[2]);
    int element_count = ag_value_array3d_element_count(&result);
    for (int i = 0; i < element_count; ++i) {
        result.values[i] = ag_relu(value_arena, x->values[i]);
    }
    return result;
}

internal AG_ValueArray nn_gap(Arena *value_arena, Arena *array_arena, AG_ValueArray3D *x) {
    AG_ValueArray result = {0};
    result.count = x->shape[0]; // num channels
    result.values = push_array(array_arena, AG_Value*, result.count);
    for (int c = 0; c < x->shape[0]; ++c) {
        AG_Value *channel_avg = ag_source(value_arena, 0);
        for (int i = 0; i < x->shape[1]; ++i) {
            for (int j = 0; j < x->shape[2]; ++j) {
                AG_Value **pixel_value = ag_value_array3d_get_value(x, c, i, j);
                channel_avg = ag_add(value_arena, channel_avg, *pixel_value);
            }
        }
        int channel_pixel_count = x->shape[1]*x->shape[2];
        channel_avg = ag_div(value_arena, channel_avg, ag_source(value_arena, channel_pixel_count));
        result.values[c] = channel_avg;
    }
    return result;
}
