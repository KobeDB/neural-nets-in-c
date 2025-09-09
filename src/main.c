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

void train_mlp(void) {

    Arena *arena = arena_alloc();

    // Hyperparameters
    F64 lr = 0.01;
    int epoch_count = 100;

    // Data spec
    F64 xs_raw[] = {
        2,    3,  -1,
        3,   -1,   0.5,
        0.5,  1,   1,
        1,    1,  -1
    };
    int x_count = 4;
    int x_dim = 3;
    AG_ValueArray *xs = push_array(arena, AG_ValueArray, x_count);
    for (int i = 0; i < x_count; ++i) {
        xs[i] = ag_value_array_from_raw(arena, &xs_raw[i*3], x_dim);
    }

    F64 ys_raw[] = {1, -1, -1, 1};
    AG_ValueArray ys = ag_value_array_from_raw(arena, ys_raw, ArrayCount(ys_raw));

    // MLP creation
    int layer_dims[] = {4,4,1};
    NN_MLP mlp = nn_make_mlp_with_random_init(arena, x_dim, layer_dims, ArrayCount(layer_dims));

    // Params
    AG_ValueArray mlp_params = nn_mlp_get_params(arena, &mlp);

    // Train loop
    for (int epoch = 0; epoch < epoch_count; ++epoch) {
        ArenaTemp scratch = scratch_begin(&arena, 1);
        Arena *epoch_arena = scratch.arena;
        
        // forward
        AG_ValueArray y_preds = {0};
        y_preds.count = x_count;
        y_preds.values = push_array(epoch_arena, AG_Value*, y_preds.count);
        for (int i = 0; i < x_count; ++i) {
            y_preds.values[i] = nn_mlp_apply(epoch_arena, epoch_arena, &mlp, xs[i]).values[0];
        }

        printf("y_preds: [");
        for (int i = 0; i < x_count; ++i) {
            printf("%f ", y_preds.values[i]->value);
        }
        printf("] ");

        // loss
        AG_Value *loss = ag_source(epoch_arena, 0);
        for (int i = 0; i < x_count; ++i) {
            AG_Value *error = ag_sub(epoch_arena, ys.values[i], y_preds.values[i]);
            AG_Value *squared_error = ag_pow(epoch_arena, error, 2);
            loss = ag_add(epoch_arena, loss, squared_error);
        }
        printf("Loss: %f\n", loss->value);

        // zero grad
        for (int i = 0; i < mlp_params.count; ++i) mlp_params.values[i]->grad = 0;

        // backward
        ag_backward(loss);

        // update
        for (int i = 0; i < mlp_params.count; ++i) {
            AG_Value *param = mlp_params.values[i];
            param->value += -lr * param->grad;
        }

        scratch_end(scratch);
    }

    arena_release(arena);
}

int main(void) {
    train_mlp();
}
