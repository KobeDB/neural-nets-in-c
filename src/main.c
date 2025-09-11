// .h
#include "base/md.h"
#include "base/md_alias.h"
#include "autograd/autograd.h"
#include "nn/nn_inc.h"
#include <stdio.h>
#include <math.h>

// .c
#include "base/md.c"
#include "autograd/autograd.c"
#include "nn/nn_inc.c"


typedef struct AG_ValueArrayArray AG_ValueArrayArray;
struct AG_ValueArrayArray {
    AG_ValueArray *arrays;
    int count;
};

AG_ValueArrayArray do_forward_pass(Arena *value_arena, Arena *array_arena, NN_MLP *mlp, AG_ValueArrayArray x) {
    AG_ValueArrayArray result = {0};
    result.count = x.count;
    result.arrays = push_array(array_arena, AG_ValueArray, result.count);
    for (int i = 0; i < x.count; ++i) {
        result.arrays[i] = nn_mlp_apply(value_arena, array_arena, mlp, x.arrays[i]);
    }
    return result;
}

// AG_Value *mse_loss(Arena *value_arena, AG_ValueArrayArray y_true, AG_ValueArrayArray y_pred) {
//     AG_Value *loss = ag_source(value_arena, 0);
//     for (int i = 0; i < y_true.count; ++i) {
//         AG_Value *error = ag_sub(value_arena, ys.values[i], y_preds.arrays[i].values[0]);
//         AG_Value *squared_error = ag_pow(epoch_arena, error, 2);
//         loss = ag_add(epoch_arena, loss, squared_error);
//     }
// }

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
    AG_ValueArrayArray xs = {0};
    xs.count = x_count;
    xs.arrays = push_array(arena, AG_ValueArray, x_count);
    for (int i = 0; i < x_count; ++i) {
        xs.arrays[i] = ag_value_array_from_raw(arena, &xs_raw[i*3], x_dim);
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
        AG_ValueArrayArray y_preds = do_forward_pass(epoch_arena, epoch_arena, &mlp, xs);

        printf("y_preds: [");
        for (int i = 0; i < x_count; ++i) {
            printf("%f ", y_preds.arrays[i].values[0]->value);
        }
        printf("] ");

        // loss
        AG_Value *loss = ag_source(epoch_arena, 0);
        for (int i = 0; i < x_count; ++i) {
            AG_Value *error = ag_sub(epoch_arena, ys.values[i], y_preds.arrays[i].values[0]);
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

void train_small_cnn(void) {
    Arena *arena = arena_alloc();

    NN_SmallCNN cnn = nn_make_small_cnn(arena, 10);

    AG_ValueArray3D x = ag_push_null_value_array3d(arena, 1, 28, 28);
    for (int i = 0; i < ag_value_array3d_element_count(&x); ++i) {
        x.values[i] = ag_source(arena, sample_f64_in_range(-1,1));
    }

    AG_ValueArray result = nn_small_cnn_apply(arena, arena, &cnn, &x);

    for (int i = 0; i < result.count; ++i) {
        printf("cnn result[%d]: %f\n", i, result.values[i]->value);
    }
    

    for (int epoch = 0; epoch < 10; ++epoch) {

    }

    arena_release(arena);
}

typedef struct Dataset Dataset;
struct Dataset {
    F64 *X;
    F64 *y;
    int sample_count;
    int x_feature_count;
};

Dataset make_moons_2d(Arena *arena, int sample_count, F64 noise) {
    Dataset result = {0};
    result.sample_count = sample_count;
    int feat_count = 2;
    result.x_feature_count = feat_count;
    result.X = push_array(arena, F64, sample_count * feat_count);
    result.y = push_array(arena, F64, sample_count);
    for (int i = 0; i < sample_count; ++i) {
        int moon = sample_f64_in_range(0,1) < 0.5 ? 0 : 1;
        F64 pi = 3.14159265359;
        F64 angle = sample_f64_in_range(0,pi);
        if (moon == 1) angle += pi; // second moon == bottom half of circle
        F64 moon_x_offset = (moon == 0 ? -0.5 : 0.5); 
        F64 moon_y_offset = (moon == 0 ? -0.25 : 0.25);
        F64 r = 1.0;
        F64 sample_x = cos(angle)*r + moon_x_offset; // x1
        F64 sample_y = sin(angle)*r + moon_y_offset; // x2
        // add noise
        sample_x += sample_f64_in_range(-noise*r, noise*r);
        sample_y += sample_f64_in_range(-noise*r, noise*r);

        result.X[i*feat_count + 0] = sample_x;
        result.X[i*feat_count + 1] = sample_y;
        result.y[i] = (F64)moon;
    }
    return result;
}

void write_dataset_to_file(FILE *file, Dataset *dataset) {
    for (int i = 0; i < dataset->x_feature_count; ++i) {
        fprintf(file, "x%d,", i);
    }
    fprintf(file, "y\n");
    for (int i = 0; i < dataset->sample_count; ++i) {
        for (int f = 0; f < dataset->x_feature_count; ++f) {
            int x_idx = i*dataset->x_feature_count + f;
            fprintf(file, "%f,", dataset->X[x_idx]);
        }
        fprintf(file, "%f\n", dataset->y[i]);
    }
}

int main(void) {
    train_mlp();
    
    train_small_cnn();

    ArenaTemp scratch = scratch_begin(0,0);
    Dataset moons = make_moons_2d(scratch.arena, 100, 0.1);
    FILE *out_file = fopen("/home/kobedb/Dev/rommel/moons.csv", "w");
    if (!out_file) fprintf(stderr, "Error opening moons output file\n");
    else {
        write_dataset_to_file(out_file, &moons);
        fclose(out_file);
    }
    scratch_end(scratch);
}
