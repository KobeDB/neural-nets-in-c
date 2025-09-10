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
AG_ValueArray3D nn_conv2d_apply(Arena *arena, NN_Conv2D *conv2d, AG_ValueArray3D *x) {
    if (x->shape[0] != conv2d->in_channels) {
        fprintf(stderr, "conv2d_apply: x.shape[0] doesn't match conv2d.in_channels\n");
        return (AG_ValueArray3D){0};
    }
    
    int padding = conv2d->padding;
    int kernel_size = conv2d->kernel_size;
    int stride = conv2d->stride;

    int out_h = (x->shape[1] + 2*padding - kernel_size)/stride + 1;
    int out_w = (x->shape[2] + 2*padding - kernel_size)/stride + 1;

    AG_ValueArray3D result = ag_push_null_value_array3d(arena, conv2d->out_channels, out_h, out_w);
    int result_value_count = ag_value_array3d_element_count(&result);
    for (int i = 0; i < result_value_count; ++i) {
        result.values[i] = ag_source(arena, 0);
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
                            AG_Value *weighted_pixel = ag_mul(arena, x_pixel_value, weight);
                            
                            AG_Value **out_pixel = ag_value_array3d_get_value(&result, kernel, ty, tx);
                            *out_pixel = ag_add(arena, *out_pixel, weighted_pixel);
                        }
                    }
                }
                // add bias
                AG_Value **out_pixel = ag_value_array3d_get_value(&result, kernel, ty, tx);
                *out_pixel = ag_add(arena, *out_pixel, conv2d->biases.values[kernel]);
            }
        }
    }

    return result;
}

internal AG_ValueArray nn_small_cnn_apply(Arena *value_arena, Arena *array_arena, NN_SmallCNN *cnn, AG_ValueArray3D *x) {
    AG_ValueArray result = {0};
    // TODO
    return result;
}
