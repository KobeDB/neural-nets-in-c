// .h
#include "base/md.h"
#include "base/md_alias.h"
#include "autograd/autograd.h"
#include "nn/nn.h"
#include <stdio.h>

// .c
#include "base/md.c"
#include "autograd/autograd.c"
#include "nn/nn.c"


void run_neural_network_things(void) {
    ArenaTemp scratch = scratch_begin(0,0);
    
    {
        AG_Value *x = push_array(scratch.arena, AG_Value, 1);
        x->value = 10;

        AG_Value *y = ag_mul(scratch.arena, x, x);
        AG_Value *z = ag_add(scratch.arena, x, y);

        ag_backward(z);

        printf("dz/dx: %f\n", x->grad);
        printf("dz/dy: %f\n", y->grad);
    }

    {
        AG_Value *a = push_array(scratch.arena, AG_Value, 1);
        a->value = 3;
        AG_Value *b = ag_add(scratch.arena, a, a);
        ag_backward(b);
        printf("db/da: %f\n", a->grad);
    }

    {
        AG_Value *x = ag_source(scratch.arena, 10);
        AG_Value *z = ag_pow(scratch.arena, x, 3);
        ag_backward(z);
        printf("");
    }

    {
        AG_Value *x = ag_source(scratch.arena, 10);
        AG_Value *y = ag_source(scratch.arena, 20);
        AG_Value *z = ag_div(scratch.arena, x, y);
        ag_backward(z);
        printf("");
    }

    {
        AG_Value *x = ag_source(scratch.arena, 10);
        AG_Value *z = ag_relu(scratch.arena, x);
        ag_backward(z);

        x = ag_source(scratch.arena, 0);
        z = ag_relu(scratch.arena, x);
        ag_backward(z);

        x = ag_source(scratch.arena, -10);
        z = ag_relu(scratch.arena, x);
        ag_backward(z);

        printf("");
    }

    // {
    //     U64 input_dim = 3;
    //     NN_MLP *mlp = 0;
    //     {
    //         ArenaTemp mlp_making_scratch = scratch_begin(&scratch.arena, 1);
    //         U64 layer_count = 3;
    //         U64 *layer_output_dims = push_array(mlp_making_scratch.arena, U64, layer_count);
    //         layer_output_dims[0] = 4;
    //         layer_output_dims[1] = 4;
    //         layer_output_dims[2] = 1;
    //         mlp = ag_make_mlp(scratch.arena, input_dim, layer_output_dims, layer_count);
    //         scratch_end(mlp_making_scratch);
    //     }

    //     F64 xvals[] = {
    //         2,    3,  -1,
    //         3,   -1,   0.5,
    //         0.5,  1,   1,
    //         1,    1,  -1
    //     };
    //     U64 sample_count = ArrayCount(xvals)/input_dim;
    //     AG_ValueArray *x = push_array(scratch.arena, AG_ValueArray, sample_count);
    //     for (int i = 0; i < sample_count; ++i) 
    //         x[i] = ag_value_array_from_raw(scratch.arena, &xvals[i * input_dim], input_dim);

    //     F64 yvals[] = {
    //         1, -1, -1, 1
    //     };

    //     AG_ValueArray y = ag_value_array_from_raw(scratch.arena, yvals, ArrayCount(yvals));

    //     for (int iter = 0; iter < 6; ++iter) {
    //         ArenaTemp grad_desct_temp = temp_begin(scratch.arena);
    //         Arena *grad_desct_arena = grad_desct_temp.arena;

    //         AG_ValueArray *y_preds = push_array(grad_desct_arena, AG_ValueArray, sample_count);

    //         for (int i = 0; i < sample_count; ++i) {
    //             y_preds[i] = nn_mlp_apply(grad_desct_arena, mlp, x[i]);
    //         }

    //         printf("y_pred: [");
    //         for (int i = 0; i < sample_count; ++i) {
    //             printf(" %f,", y_preds[i].values[0]->value);
    //         }
    //         printf("]\n");

    //         AG_ValueArray params = nn_mlp_get_params(grad_desct_arena, mlp);

    //         for (int i = 0; i < params.count; ++i) {
    //             params.values[i]->grad = 0;
    //         }

    //         AG_Value *loss = ag_source(grad_desct_arena, 0);
    //         for (int i = 0; i < sample_count; ++i) {
    //             AG_Value *y_pred = y_preds[i].values[0];
    //             AG_Value *sample_loss = ag_pow(grad_desct_arena, ag_sub(grad_desct_arena, y_pred, y.values[i]), 2);
    //             loss = ag_add(grad_desct_arena, loss, sample_loss);
    //         }

    //         printf("Loss: %f\n", loss->value);

    //         ag_backward(loss);

    //         for (int i = 0; i < params.count; ++i) {
    //             AG_Value *param = params.values[i];
    //             param->value -= 0.05 * param->grad;
    //         }

    //         temp_end(grad_desct_temp);
    //     }

    //    printf("");
    // }

    scratch_end(scratch);
}

int main(void) {
    run_neural_network_things();
}
